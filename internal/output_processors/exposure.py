from dataclasses import dataclass
import torch
from internal.utils.general_utils import inverse_sigmoid
from .output_processors import OutputProcessor


@dataclass
class ExposureProcessor(OutputProcessor):
    lr_init: float = 1e-2

    lr_final_factor: float = 0.1

    max_steps: int = 30_000

    max_gray_scale: float = 5.

    with_bias: bool = False

    max_gamma: float = 5.

    shade_correction: bool = False

    shade_correction_size: int = 128

    def instantiate(self, *args, **kwargs):
        # avoid exceptions when loading previous checkpoint
        if not hasattr(self, "with_bias"):
            self.config.with_bias = False
        if not hasattr(self, "shade_correction"):
            self.config.shade_correction = False

        # must > 1 or optimization will not work
        assert self.max_gray_scale > 1.
        assert self.max_gamma > 1.
        return ExposureProcessorModule(self)


class ExposureProcessorModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_exposures(self, n_appearances: int, device, n_cameras: int = 1):
        n_parameters = 4
        if self.config.with_bias:
            n_parameters += 3
        exposures = torch.full(
            (n_appearances, n_parameters),
            inverse_sigmoid(torch.tensor(1. / self.config.max_gray_scale, dtype=torch.float32, device=device)),
            dtype=torch.float32,
            device=device,
        )
        exposures[:, -1] = inverse_sigmoid(torch.tensor(1. / self.config.max_gamma, dtype=torch.float32, device=device))

        if self.config.with_bias:
            # bias = sigmoid(x) * 2. - 1.
            exposures[:, 3:6].fill_(inverse_sigmoid(torch.tensor(0.5, dtype=torch.float32, device=device)))
            assert torch.allclose(torch.sigmoid(exposures[:, 3:6]) * 2. - 1., torch.tensor(0., device=device))

        assert torch.allclose(torch.sigmoid(exposures[:, :3]) * self.config.max_gray_scale, torch.tensor(1., device=device))
        assert torch.allclose(torch.sigmoid(exposures[:, -1]) * self.config.max_gamma, torch.tensor(1., device=device))
        self.exposure_parameters = torch.nn.Parameter(exposures, requires_grad=True)

        if self.config.shade_correction:
            # TODO: multiple cameras
            self.shade_correction = torch.nn.Parameter(
                torch.full(
                    size=(n_cameras, self.config.shade_correction_size, self.config.shade_correction_size),
                    fill_value=inverse_sigmoid(torch.tensor(0.95, device=device)),
                    device=device,
                ),
                requires_grad=True,
            )

    def setup(self, stage: str, pl_module=None, *args, **kwargs):
        if pl_module is not None:
            max_input_id = 0
            appearance_group_ids = pl_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
            if appearance_group_ids is not None:
                for i in appearance_group_ids.values():
                    if i[0] > max_input_id:
                        max_input_id = i[0]
            n_appearances = max_input_id + 1
            assert n_appearances > 1

            self.init_exposures(n_appearances, pl_module.device)

            print("{} exposure groups".format(n_appearances))

    def load_state_dict(self, state_dict, strict=True):
        n_appearances = state_dict["exposure_parameters"].shape[0]
        self.init_exposures(n_appearances, state_dict["exposure_parameters"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, pl_module):
        params = [
            {"params": [self.exposure_parameters], "name": "exposure"},
        ]
        if self.config.shade_correction:
            params.append({"params": [self.shade_correction], "name": "shade"})

        optimizer = torch.optim.Adam(
            params=params,
            lr=self.config.lr_init,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.config.lr_final_factor ** min(iter / self.config.max_steps, 1),
            verbose=False,
        )

    def forward(self, camera, outputs) -> None:
        if "render" not in outputs:
            return

        adjustment = torch.sigmoid(self.exposure_parameters[camera.appearance_id])

        rendered_image = outputs["render"]  # [C, H, W]

        if self.config.shade_correction:
            # TODO: multiple cameras
            shade_correction_map = torch.sigmoid(self.shade_correction[0][None, None, :, :])
            shade_correction_map = shade_correction_map / shade_correction_map.detach().max().clamp_min_(1e-4)
            shade_correction_map = torch.nn.functional.interpolate(
                shade_correction_map,
                size=(rendered_image.shape[1], rendered_image.shape[2]),
                mode="bilinear",
                align_corners=True,
            )[0]  # [1, H, W]
            rendered_image = rendered_image * shade_correction_map

        rendered_image = adjustment[:3, None, None] * self.config.max_gray_scale * rendered_image
        if self.config.with_bias:
            bias = adjustment[3:6, None, None] * 2. - 1.
            rendered_image = rendered_image + bias

        with torch.no_grad():
            rendered_image_clamped = torch.clamp(rendered_image, min=0., max=1.)
            rendered_image_clamped_diff = rendered_image - rendered_image_clamped
        rendered_image = rendered_image - rendered_image_clamped_diff

        rendered_image = torch.pow(rendered_image + 1e-5, adjustment[-1] * self.config.max_gamma)
        outputs["render"] = rendered_image

    def training_forward(self, batch, outputs):
        return self(batch[0], outputs)

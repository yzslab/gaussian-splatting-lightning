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

    max_gamma: float = 5.

    def instantiate(self, *args, **kwargs):
        # must > 1 or optimization will not work
        assert self.max_gray_scale > 1.
        assert self.max_gamma > 1.
        return ExposureProcessorModule(self)


class ExposureProcessorModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def init_exposures(self, n_appearances: int, device):
        exposures = torch.full(
            (n_appearances, 4),
            inverse_sigmoid(torch.tensor(1. / self.config.max_gray_scale, dtype=torch.float32, device=device)),
            dtype=torch.float32,
            device=device,
        )
        exposures[:, -1] = inverse_sigmoid(torch.tensor(1. / self.config.max_gamma, dtype=torch.float32, device=device))
        assert torch.allclose(torch.sigmoid(exposures[:, :3]) * self.config.max_gray_scale, torch.tensor(1., device=device))
        assert torch.allclose(torch.sigmoid(exposures[:, -1]) * self.config.max_gamma, torch.tensor(1., device=device))
        self.exposure_parameters = torch.nn.Parameter(exposures, requires_grad=True)

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
        optimizer = torch.optim.Adam(
            params=[
                {"params": [self.exposure_parameters], "name": "exposure"},
            ],
            lr=self.config.lr_init,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.config.lr_final_factor ** min(iter / self.config.max_steps, 1),
            verbose=False,
        )

    def training_forward(self, batch, outputs):
        if "render" not in outputs:
            return

        camera = batch[0]
        adjustment = torch.sigmoid(self.exposure_parameters[camera.appearance_id])

        rendered_image = outputs["render"]  # [C, H, W]
        rendered_image = adjustment[:3, None, None] * self.config.max_gray_scale * rendered_image
        rendered_image = torch.pow(rendered_image + 1e-5, adjustment[-1] * self.config.max_gamma)
        rendered_image = torch.clamp_max(rendered_image, max=1.)
        outputs["render"] = rendered_image

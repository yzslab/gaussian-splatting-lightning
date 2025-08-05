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
        assert self.max_gray_scale > 1.
        assert self.max_gamma > 1.
        return ExposureProcessorModule(self)


class ExposureProcessorModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def training_setup(self, pl_module):
        max_input_id = 0
        appearance_group_ids = pl_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
        if appearance_group_ids is not None:
            for i in appearance_group_ids.values():
                if i[0] > max_input_id:
                    max_input_id = i[0]
        n_appearances = max_input_id + 1
        assert n_appearances > 1

        exposures = torch.full(
            (n_appearances, 4),
            inverse_sigmoid(torch.tensor(1. / self.config.max_gray_scale, dtype=torch.float32)),
            dtype=torch.float32,
            device=pl_module.device,
        )
        exposures[:, -1] = inverse_sigmoid(torch.tensor(1. / self.config.max_gamma, dtype=torch.float32, device=pl_module.device))
        assert torch.allclose(torch.sigmoid(exposures[:, :3]) * self.config.max_gray_scale, torch.tensor(1., device=pl_module.device))
        assert torch.allclose(torch.sigmoid(exposures[:, -1]) * self.config.max_gamma, torch.tensor(1., device=pl_module.device))
        self.exposure_parameters = torch.nn.Parameter(exposures, requires_grad=True)


        print("{} exposure groups".format(n_appearances))

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

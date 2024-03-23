import lightning
import torch
from typing import Any, Tuple, Optional
from internal.cameras.cameras import Camera
from internal.models.gaussian_model import GaussianModel


class Renderer(torch.nn.Module):
    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        pass

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            **kwargs,
        )

    def before_training_step(
            self,
            step: int,
            module,
    ):
        return

    def after_training_step(
            self,
            step: int,
            module,
    ):
        return

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_setup(self, module: lightning.LightningModule) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        return None, None

    def on_load_checkpoint(self, module, checkpoint):
        pass

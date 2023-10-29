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

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_setup(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        return None, None

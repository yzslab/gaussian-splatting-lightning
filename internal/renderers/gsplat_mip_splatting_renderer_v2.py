from dataclasses import dataclass

import torch

from .renderer import RendererConfig, Renderer
from .gsplat_renderer import GSPlatRenderer
from ..cameras import Camera
from ..models.mip_splatting import MipSplattingModel


@dataclass
class GSplatMipSplattingRendererV2(RendererConfig):
    filter_2d_kernel_size: float = 0.1

    def instantiate(self, *args, **kwargs) -> "GSplatMipSplattingRendererV2Module":
        return GSplatMipSplattingRendererV2Module(self)


class GSplatMipSplattingRendererV2Module(Renderer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: MipSplattingModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        opacities, scales = pc.get_3d_filtered_scales_and_opacities()
        return GSPlatRenderer.render(
            means3D=pc.get_xyz,
            opacities=opacities,
            scales=scales,
            rotations=pc.get_rotation,
            features=pc.get_features,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            anti_aliased=True,
            extra_projection_kwargs={
                "filter_2d_kernel_size": self.config.filter_2d_kernel_size,
            }
        )

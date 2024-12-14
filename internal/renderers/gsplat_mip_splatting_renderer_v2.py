from dataclasses import dataclass

import torch

from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1RendererModule
from ..cameras import Camera
from ..models.mip_splatting import MipSplattingModel


@dataclass
class GSplatMipSplattingRendererV2(GSplatV1Renderer):
    filter_2d_kernel_size: float = 0.1

    def instantiate(self, *args, **kwargs) -> "GSplatMipSplattingRendererV2Module":
        return GSplatMipSplattingRendererV2Module(self)


class GSplatMipSplattingRendererV2Module(GSplatV1RendererModule):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_scales_and_opacities(self, pc: MipSplattingModel):
        opacities, scales = pc.get_3d_filtered_scales_and_opacities()
        return scales, opacities

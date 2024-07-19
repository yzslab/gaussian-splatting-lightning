from dataclasses import dataclass

from . import Renderer
from .renderer import RendererConfig
from .gsplat_v1_renderer import GSplatV1Renderer

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


@dataclass
class GSPlatRenderer(RendererConfig):
    """
    This class is kept for the compatible purpose
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS

    def instantiate(self, *args, **kwargs) -> Renderer:
        return GSplatV1Renderer(
            tile_size=self.block_size,
            anti_aliased=self.anti_aliased,
        ).instantiate(*args, **kwargs)

from typing import Tuple, Any
from dataclasses import dataclass
import torch
from gsplat.sh import spherical_harmonics
from gsplat.sh_decomposed import spherical_harmonics_decomposed
from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1RendererModule


@dataclass
class GlossyRenderer(GSplatV1Renderer):
    def instantiate(self, *args, **kwargs):
        return GlossyRendererModule(self)


class GlossyRendererModule(GSplatV1RendererModule):
    def get_rgbs(self, camera, gaussian_model, projections: Tuple, visibility_filter, status: Any, **kwargs) -> Tuple[torch.Tensor, Any]:
        viewdirs = status  # (N, 3)
        if gaussian_model.is_pre_activated or not self.config.separate_sh:
            rgbs = spherical_harmonics(gaussian_model.active_sh_degree, viewdirs, gaussian_model.get_features, visibility_filter)
        else:
            rgbs = spherical_harmonics_decomposed(
                gaussian_model.active_sh_degree,
                viewdirs,
                gaussian_model.get_shs_dc(),
                gaussian_model.get_shs_rest(),
                visibility_filter,
            )
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        return rgbs

    def get_opacities(self, camera, gaussian_model, projections, visibility_filter, status, **kwargs):
        viewdirs = torch.nn.functional.normalize(gaussian_model.get_xyz.detach() - camera.camera_center, dim=-1)  # (N, 3)
        opacities = torch.clamp(gaussian_model.get_view_dep_opacities(dirs=viewdirs, masks=visibility_filter).squeeze(-1), min=0., max=1.)

        return opacities, viewdirs

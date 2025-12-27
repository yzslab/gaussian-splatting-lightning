"""
Revising Densification in Gaussian Splatting
https://arxiv.org/abs/2404.06109

Opacity correction only currently
"""

from dataclasses import dataclass
import torch
from .density_controller import DensityController, DensityControllerImpl
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl
from internal.utils.general_utils import inverse_sigmoid


@dataclass
class RevisingDensityController(VanillaDensityController):
    def instantiate(self, *args, **kwargs):
        return RevisingDensityControllerModule(self)


class RevisingDensityControllerModule(VanillaDensityControllerImpl):
    def _densify_and_clone(self, grads, gaussian_model, optimizers):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # NEW: Opacity correction
        current_opacity = gaussian_model.get_opacities()[selected_pts_mask]
        alpha_hat = 1. - torch.sqrt(1. - current_opacity)
        raw_alpha_hat = gaussian_model.opacity_inverse_activation(alpha_hat)
        gaussian_model.properties["opacities"][selected_pts_mask] = raw_alpha_hat
        new_properties["opacities"] = raw_alpha_hat

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

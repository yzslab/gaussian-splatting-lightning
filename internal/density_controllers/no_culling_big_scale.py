import torch
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl


class NoCullingBigScaleDC(VanillaDensityController):
    def instantiate(self, *args, **kwargs):
        return NoCullingBigScaleDCModel(self)


class NoCullingBigScaleDCModel(VanillaDensityControllerImpl):
    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        min_opacity = self.config.cull_opacity_threshold

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers)

        # prune
        if self.config.cull_by_max_opacity:
            # TODO: re-implement as a new density controller
            prune_mask = torch.logical_and(
                gaussian_model.get_opacity_max() >= 0.,
                gaussian_model.get_opacity_max() < min_opacity,
            )
            gaussian_model.reset_opacity_max()
        else:
            prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

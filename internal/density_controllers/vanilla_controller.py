from typing import Dict, Type, Tuple, Optional, Union, List
from dataclasses import field
import torch
from lightning import LightningModule

from .density_controller import DensityController, DensityControllerImpl


class VanillaDensityControllerImpl(DensityControllerImpl):
    def configure_optimizers(self, pl_module: LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        # Getting hparams here looks a little bit wired, but some hparams only can be obtained here.
        # TODO: refactor
        self.hparams = pl_module.hparams
        self.optimization_hparams = pl_module.optimization_hparams
        self.cameras_extent = pl_module.cameras_extent
        self.prune_extent = pl_module.prune_extent

        return None, None

    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        with torch.no_grad():
            # Densification
            if global_step < self.hparams["gaussian"].optimization.densify_until_iter:
                gaussians = gaussian_model
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                if self.hparams["absgrad"] is True:
                    viewspace_point_tensor.grad = viewspace_point_tensor.absgrad
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, scale=viewspace_points_grad_scale)

                if global_step > self.optimization_hparams.densify_from_iter and global_step % self.optimization_hparams.densification_interval == 0:
                    size_threshold = 20 if global_step > self.optimization_hparams.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        self.hparams["gaussian"].optimization.densify_grad_threshold,
                        self.hparams["gaussian"].optimization.cull_opacity_threshold,
                        extent=self.cameras_extent,
                        prune_extent=self.prune_extent,
                        max_screen_size=size_threshold,
                    )

                if global_step % self.hparams["gaussian"].optimization.opacity_reset_interval == 0 or \
                        (
                                torch.all(pl_module.background_color == 1.) and global_step == self.hparams[
                            "gaussian"].optimization.densify_from_iter
                        ):
                    gaussians.reset_opacity()


class VanillaDensityController(DensityController):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return VanillaDensityControllerImpl()

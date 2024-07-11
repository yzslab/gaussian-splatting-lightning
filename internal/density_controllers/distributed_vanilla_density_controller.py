from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import torch
from lightning import LightningModule

from .density_controller import DensityController, DensityControllerImpl


@dataclass
class DistributedVanillaDensityController(DensityController):
    percent_dense: float = 0.01

    densification_interval: int = 100

    opacity_reset_interval: int = 3000

    densify_from_iter: int = 500

    densify_until_iter: int = 15_000

    densify_grad_threshold: float = 0.0002

    cull_opacity_threshold: float = 0.005
    """threshold of opacity for culling gaussians."""

    camera_extent_factor: float = 1.

    absgrad: bool = False

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return DistributedVanillaDensityControllerImpl(self)


class DistributedVanillaDensityControllerImpl(DensityControllerImpl):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        if stage == "fit":
            self.cameras_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor

    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        with torch.no_grad():
            # Densification
            if global_step < self.config.densify_until_iter:
                member_data_list = outputs["member_data_list"]
                projection_results_list = outputs["projection_results_list"]
                # processing for each projection results
                for i in range(len(projection_results_list)):
                    # retrieve data
                    member_data = member_data_list[i]
                    xys, _, radii, _, _, _, _ = projection_results_list[i]

                    viewspace_point_tensor = xys
                    visibility_filter = radii > 0
                    viewspace_points_grad_scale = torch.ones((2, ), dtype=torch.float, device=xys.device)
                    if outputs["xys_grad_scale_required"] is True:
                        viewspace_points_grad_scale = 0.5 * torch.tensor([[member_data.width, member_data.height]], dtype=torch.float, device=xys.device)

                    # original operations
                    gaussians = gaussian_model
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                    if self.config.absgrad is True:
                        viewspace_point_tensor.grad = viewspace_point_tensor.absgrad
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, scale=viewspace_points_grad_scale)

                if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                    size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        self.config.densify_grad_threshold,
                        self.config.cull_opacity_threshold,
                        percent_dense=self.config.percent_dense,
                        extent=self.cameras_extent,
                        prune_extent=self.prune_extent,
                        max_screen_size=size_threshold,
                    )

                if global_step % self.config.opacity_reset_interval == 0 or \
                        (
                                torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                        ):
                    gaussians.reset_opacity()

from dataclasses import dataclass
import os
import torch
from lightning import LightningModule
from .taming_3dgs_density_controller import Taming3DGSDensityController, Taming3DGSDensityControllerModule, Taming3DGSUtils


@dataclass
class Taming3DGSDensityFFController(Taming3DGSDensityController):
    # partition: str = None
    """Partition data directory"""

    # partition_idx: int = None

    max_grad_decay_factor: float = 4

    max_radius_factor: float = 1.0
    """for those 'distance >= max_radius_factor * radius', 'grad = grad / max_grad_decay_factor'"""

    budget_auto_scaling: bool = True

    def instantiate(self, *args, **kwargs) -> "Taming3DGSDensityControllerFFModule":
        return Taming3DGSDensityControllerFFModule(self)


class Taming3DGSDensityControllerFFModule(Taming3DGSDensityControllerModule):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        partition_size = pl_module.store.partition_size
        default_partition_size = pl_module.store.default_partition_size
        if self.config.budget_auto_scaling and not torch.allclose(partition_size, torch.tensor(default_partition_size, dtype=partition_size.dtype, device=partition_size.device)):
            size_factor = torch.prod(partition_size / default_partition_size)
            if size_factor > 1.:
                # since the merged partition are background areas, no need to increase the budget too much
                # TODO: avoid hard-coded maximum
                actual_size_factor = torch.clamp_max(torch.sqrt(size_factor), max=2.).item()
                self.config.budget *= actual_size_factor
                print("budget {}x increased to {}".format(actual_size_factor, self.config.budget))

        super().setup(stage, pl_module)

        if stage == "fit":
            start_count = (self._get_grad_decay_factors(pl_module)[1] == 0.).sum().item()
            self.counts_array = Taming3DGSUtils.get_count_array(
                start_count=start_count,
                multiplier=self.config.budget,
                densify_until_iter=self.config.densify_until_iter,
                densify_from_iter=self.config.densify_from_iter,
                densification_interval=self.config.densification_interval,
                mode=self.config.mode,
            )
            print(self.counts_array)

    def _get_grad_decay_factors(self, pl_module):
        # decay grads based on distance (xy only)
        distances = pl_module.store.distances
        normalized_distances = torch.clamp_max(
            pl_module.store.distance_factors / self.config.max_radius_factor,
            max=1.,
        )
        decay_factors = (normalized_distances * (self.config.max_grad_decay_factor - 1)) + 1

        return decay_factors, distances

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze(-1)  # [N]

        grad_decay_factor, distances = self._get_grad_decay_factors(self.avoid_state_dict[0])
        if self.config.max_grad_decay_factor > 1.:
            grads = grads / grad_decay_factor

        n_inside_partition = (distances == 0.).sum()
        self.log_metric("n_inside_partition", n_inside_partition)

        pl_module = self.avoid_state_dict[0]

        # sample cameras
        n_cameras = len(pl_module.trainer.train_dataloader.cached)
        sample_cameras = []
        sample_edge_losses = []
        for camera_idx in torch.randperm(n_cameras, dtype=torch.int)[:self.config.n_sample_cameras].tolist():
            sample_cameras.append(pl_module.trainer.train_dataloader.cached[camera_idx])
            sample_edge_losses.append(self.all_edges[camera_idx])

        # calculate importance
        gaussian_importance = Taming3DGSUtils.compute_gaussian_score(
            gaussian_model=gaussian_model,
            renderer=pl_module.renderer,
            global_steps=pl_module.global_step,
            all_grads=grads,
            sample_cameras=sample_cameras,
            edge_losses=sample_edge_losses,
            bg_color=pl_module._fixed_background_color().to(device=pl_module.device),
            score_coeffs=self.config.score_coeffs,
            lambda_dssim=pl_module.metric.config.lambda_dssim,
        )

        # must < 2^24
        if gaussian_model.n_gaussians < 16_000_000:
            self._densify_and_prune_with_scores(
                grads=grads,
                scores=gaussian_importance,
                gaussian_model=gaussian_model,
                optimizers=optimizers,
                n_inside_partition=n_inside_partition,
            )

        # prune
        self._opacity_culling(gaussian_importance, max_screen_size, gaussian_model, optimizers)

        self.densify_iter_num = self.densify_iter_num + 1

        torch.cuda.empty_cache()

    def _densify_and_prune_with_scores(
            self,
            grads,
            scores,
            gaussian_model,
            optimizers,
            n_inside_partition,
    ):
        extent = self.cameras_extent
        scales = gaussian_model.get_scales()
        max_scales = torch.max(scales, dim=-1).values

        grad_qualifiers = grads >= self.config.densify_grad_threshold
        clone_qualifiers = max_scales <= self.config.percent_dense * extent
        split_qualifiers = max_scales > self.config.percent_dense * extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers)
        total_clones = torch.sum(all_clones).item()
        total_splits = torch.sum(all_splits).item()

        # Budget only apply inside partition
        curr_points = n_inside_partition
        upper_budget = self.counts_array[self.densify_iter_num]
        expected_budget = total_clones + total_splits + curr_points
        budget = min(upper_budget, expected_budget)

        self.log_metric("total_clones", total_clones)
        self.log_metric("total_splits", total_splits)
        self.log_metric("upper_budget", upper_budget)
        self.log_metric("expected_budget", expected_budget)
        self.log_metric("budget", budget)

        clone_budget = ((budget - curr_points) * total_clones) // (total_clones + total_splits)
        split_budget = ((budget - curr_points) * total_splits) // (total_clones + total_splits)

        self.log_metric("clone_budget", clone_budget)
        self._densify_and_clone(
            scores,
            clone_budget,
            all_clones,
            gaussian_model,
            optimizers,
        )

        self.log_metric("split_budget", split_budget)
        self._densify_and_split(
            scores,
            split_budget,
            all_splits,
            gaussian_model,
            optimizers,
        )

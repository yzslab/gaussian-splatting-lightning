from dataclasses import dataclass
import os
import torch
from lightning import LightningModule
from .taming_3dgs_density_controller import Taming3DGSDensityController, Taming3DGSDensityControllerModule, Taming3DGSUtils


@dataclass
class Taming3DGSDensityFFController(Taming3DGSDensityController):
    partition: str = None
    """Partition data directory"""

    partition_idx: int = None

    max_grad_decay_factor: float = 4

    max_radius_factor: float = 1.0
    """for those 'distance >= max_radius_factor * radius', 'grad = grad / max_grad_decay_factor'"""

    def instantiate(self, *args, **kwargs) -> "Taming3DGSDensityControllerFFModule":
        return Taming3DGSDensityControllerFFModule(self)


class Taming3DGSDensityControllerFFModule(Taming3DGSDensityControllerModule):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        # load partition data
        partition_data = torch.load(os.path.join(
            self.config.partition,
            "partitions.pt",
        ))
        default_partition_size = partition_data["scene_config"]["partition_size"]
        self.register_buffer(
            "default_partition_size",
            torch.tensor(default_partition_size, dtype=torch.float),
            persistent=False,
        )
        partition_size = partition_data["partition_coordinates"]["size"][self.config.partition_idx]

        if not torch.allclose(partition_size, torch.tensor(default_partition_size, dtype=partition_size.dtype, device=partition_size.device)):
            size_factor = torch.prod(partition_size / default_partition_size)
            if size_factor > 1.:
                # since the merged partition are background areas, no need to increase the budget too much
                actual_size_factor = torch.sqrt(size_factor).item()
                self.config.budget *= actual_size_factor
                print("budget {}x increased to {}".format(actual_size_factor, self.config.budget))

        # get transform matrix
        try:
            rotation_transform = partition_data["extra_data"]["rotation_transform"]
        except:
            print("FFDensityController: No orientation transform")
            rotation_transform = torch.eye(4, dtype=torch.float, device=pl_module.device)
        self.register_buffer(
            "rotation_transform",
            rotation_transform,
            persistent=False,
        )
        # get bounding box
        partition_bbox_min = partition_data["partition_coordinates"]["xy"][self.config.partition_idx]
        partition_bbox_max = partition_bbox_min + partition_size

        # partition_center = partition_data["partition_coordinates"]["xy"][self.config.partition_idx] + partition_size * 0.5
        # self.register_buffer(
        #     "partition_center",
        #     partition_center,
        #     persistent=False,
        # )
        # partition_bbox_min = partition_center - partition_size
        # partition_bbox_max = partition_center + partition_size
        self.register_buffer("partition_bbox_min", partition_bbox_min, persistent=False)
        self.register_buffer("partition_bbox_max", partition_bbox_max, persistent=False)

        print("partition_idx=#{}, id={}, transform={}, default_size={}, partition_size={}, bbox=(\n  {}, \n  {}\n)".format(
            self.config.partition_idx,
            partition_data["partition_coordinates"]["id"][self.config.partition_idx],
            self.rotation_transform.tolist(),
            default_partition_size,
            partition_size.tolist(),
            self.partition_bbox_min.tolist(),
            self.partition_bbox_max.tolist(),
        ))

        super().setup(stage, pl_module)

        if stage == "fit":
            start_count = (self._get_grad_decay_factors(pl_module.gaussian_model) == 1.).sum().item()
            self.counts_array = Taming3DGSUtils.get_count_array(
                start_count=start_count,
                multiplier=self.config.budget,
                densify_until_iter=self.config.densify_until_iter,
                densify_from_iter=self.config.densify_from_iter,
                densification_interval=self.config.densification_interval,
                mode=self.config.mode,
            )
            print(self.counts_array)

    def _get_normalized_distance_to_bounding_box(self, gaussian_model):
        # transform 3D means
        transformed_means = gaussian_model.get_means() @ self.rotation_transform[:2, :3].T + self.rotation_transform[:2, 3]  # [N, 2]

        dist_min2p = self.partition_bbox_min - transformed_means
        dist_p2max = transformed_means - self.partition_bbox_max
        dxy = torch.maximum(dist_min2p, dist_p2max)
        return torch.clamp_max(
            (torch.sqrt(torch.pow(dxy.clamp(min=0.), 2).sum(dim=-1)) / self.default_partition_size) / self.config.max_radius_factor,
            max=1.,
        ), transformed_means  # [N]

    def _get_grad_decay_factors(self, gaussian_model):
        # decay grads based on distance (xy only)
        normalized_distances, _ = self._get_normalized_distance_to_bounding_box(gaussian_model)
        decay_factors = (normalized_distances * (self.config.max_grad_decay_factor - 1)) + 1

        return decay_factors

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze(-1)  # [N]

        grad_decay_factor = self._get_grad_decay_factors(gaussian_model)
        grads = grads / grad_decay_factor

        n_inside_partition = (grad_decay_factor == 1.).sum()
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

        if clone_budget > 0:
            self.log_metric("clone_budget", clone_budget)
            self._densify_and_clone(
                scores,
                clone_budget,
                all_clones,
                gaussian_model,
                optimizers,
            )

        if split_budget > 0:
            self.log_metric("split_budget", split_budget)
            self._densify_and_split(
                scores,
                split_budget,
                all_splits,
                gaussian_model,
                optimizers,
            )

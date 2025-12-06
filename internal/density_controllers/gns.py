"""
Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting
https://xiaobin2001.github.io/GNS-web/

Below components from Improved-GS are not included yet:
    Recovery-Aware Pruning
    Muti-view Update
"""

from dataclasses import dataclass
import math
import torch
from internal.utils.general_utils import build_rotation
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl
from .taming_3dgs_density_controller import Taming3DGSUtils
from internal.renderers.gsplat_v1_renderer import GSplatV1
from .logger_mixin import LoggerMixin


@dataclass
class GNS(VanillaDensityController):
    budget: int = -1

    opacity_reg_interval: int = 50

    opacity_reg_from: int = 15_000

    opacity_reg_until: int = 23_000

    opacity_reg_weight: float = 2e-4

    opacity_reg_prior_free_steps: int = 1_000

    opacity_reg_restore_lr_after: int = 1_000

    natural_selection_min_opacity: float = 0.001

    n_sample_cameras: int = 10

    opacity_reduction: float = 0.6

    split_distance: float = 0.45

    def instantiate(self, *args, **kwargs):
        assert self.budget > 0, "The budget that represents the maximum number of Gaussians must be provided"

        return GNSModule(self)


class GNSModule(LoggerMixin, VanillaDensityControllerImpl):
    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        pl_module.on_train_start_hooks.append(self.get_edges)
        pl_module.extra_train_metrics.append(self.opacity_reg)
        pl_module.on_after_backward_hooks.append(self.natural_selection)
        pl_module.on_after_backward_hooks.append(self.restore_opacity_lr_without_pruning)
        # self.avoid_state_dict = (pl_module,)

        self.opacity_min = None

        self.register_buffer("current_opacity_reg_weight", torch.tensor(self.config.opacity_reg_weight, dtype=torch.float, device=pl_module.device))
        self.register_buffer("prune_iter", torch.tensor(9999999, dtype=torch.int32, device=pl_module.device))
        self.register_buffer("opacity_lr_factor", torch.tensor(1., device=pl_module.device))

    def get_edges(self, gaussian_model, pl_module):
        from tqdm.auto import tqdm
        train_set = pl_module.trainer.train_dataloader.cached
        all_edges = []
        for item in tqdm(train_set, leave=False, desc="Getting edges..."):
            image = item[1][1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.
            edges_loss = Taming3DGSUtils.get_edges(image).squeeze(0)
            edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
            all_edges.append(edges_loss_norm.cpu())
        self.all_edges = all_edges

    def get_budget_by_step(self, step: int):
        startI = self.config.densify_from_iter
        endI = self.config.densify_until_iter - startI
        rate = (step - startI) / (endI - startI)

        if rate >= 1:
            budget = int(self.config.budget * 3)
        else:
            budget = int(math.sqrt(rate) * self.config.budget * 3)

        return budget

    def get_gaussian_importance(self, gaussian_model):
        pl_module = self.avoid_state_dict["pl"]

        # sample cameras
        n_cameras = len(pl_module.trainer.train_dataloader.cached)
        sample_cameras = []
        sample_edge_losses = []
        for camera_idx in torch.randperm(n_cameras, dtype=torch.int)[:self.config.n_sample_cameras].tolist():
            sample_cameras.append(pl_module.trainer.train_dataloader.cached[camera_idx])
            sample_edge_losses.append(self.all_edges[camera_idx])

        # calculate importance
        num_points = gaussian_model.n_gaussians
        gaussian_importance = torch.zeros(
            (num_points,),
            device=gaussian_model.means.device,
            dtype=torch.float32,
        )

        for camera_idx in range(len(sample_cameras)):
            # TODO: mask
            # TODO: move camera to GPU
            camera, image_info, mask = sample_cameras[camera_idx]
            assert mask is None
            _, gt_image, masked_pixels = image_info

            scales, status = pl_module.renderer.get_scales(camera, gaussian_model)
            preprocessed_camera = GSplatV1.preprocess_camera(camera)
            projections = GSplatV1.project(
                preprocessed_camera,
                gaussian_model.get_means(),
                scales,
                gaussian_model.get_rotations(),
                eps2d=pl_module.renderer.config.filter_2d_kernel_size,
                anti_aliased=pl_module.renderer.config.anti_aliased,
                radius_clip=pl_module.renderer.runtime_options.radius_clip,
                # radius_clip_from=self.runtime_options.radius_clip_from,
                camera_model=pl_module.renderer.runtime_options.camera_model,
            )
            radii, means2d, depths, conics, compensations = projections
            radii_squeezed = radii.squeeze(0)
            visibility_filter = radii_squeezed > 0
            opacities, status = pl_module.renderer.get_opacities(
                camera,
                gaussian_model,
                projections,
                visibility_filter,
                status,
            )
            opacities = opacities.unsqueeze(0)  # [1, N]
            if pl_module.renderer.config.anti_aliased:
                opacities = opacities * compensations
            isects = pl_module.renderer.isect_encode(
                preprocessed_camera,
                projections,
                opacities,
                tile_size=pl_module.renderer.config.block_size,
            )

            all_depths, all_radii, loss_accum, reverse_counts, blending_weights, dist_accum = Taming3DGSUtils.rasterize_to_weights(
                opacities=opacities.squeeze(0),
                projections=projections,
                isects=isects,
                pixel_weights=self.all_edges[camera_idx].to(device=camera.device),
                viewpoint_camera=camera,
            )

            # In gsplat, only the visible Gaussians have valid depth values
            all_depths *= visibility_filter

            gaussian_importance += Taming3DGSUtils.normalize(1., loss_accum) * visibility_filter

        gaussian_importance = gaussian_importance / len(sample_cameras)

        return gaussian_importance

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers):
        min_opacity = self.config.cull_opacity_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # get the mask using the gradient threshold
        grad_norms = torch.norm(grads, dim=-1)
        del grads
        selected_pts_mask = torch.where(grad_norms >= self.config.densify_grad_threshold, True, False)
        # calculate the budget of current step
        n_current = gaussian_model.n_gaussians
        n_expected = selected_pts_mask.sum()
        step_budget = self.get_budget_by_step(self.avoid_state_dict["pl"].global_step)
        budget = min(step_budget, n_current + n_expected)
        n_addable = budget - n_current
        self.log_metric("n_expected", n_expected)
        self.log_metric("step_budget", step_budget)
        self.log_metric("n_addable", n_addable)

        if n_addable > 0:
            # sample based on the gradient norms
            grad_norms = grad_norms * selected_pts_mask
            gaussian_importance = self.get_gaussian_importance(gaussian_model) * selected_pts_mask
            sampled_indices = torch.multinomial(
                gaussian_importance,
                n_addable,
                replacement=False,
            )
            final_selected_pts_mask = torch.zeros_like(selected_pts_mask)
            del selected_pts_mask
            final_selected_pts_mask[sampled_indices] = True

            # densify
            # self._densify_and_clone(gaussian_model, optimizers, final_selected_pts_mask)
            # self._densify_and_split(gaussian_model, optimizers, final_selected_pts_mask)
            self._long_axis_split(gaussian_model, optimizers, final_selected_pts_mask)

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
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

    # def _densify_and_clone(self, gaussian_model, optimizers, selected_pts_mask):
    #     percent_dense = self.config.percent_dense
    #     scene_extent = self.cameras_extent

    #     # Exclude big Gaussians
    #     selected_pts_mask = torch.logical_and(
    #         selected_pts_mask,
    #         torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
    #     )

    #     self.log_metric("n_clone", selected_pts_mask.sum())

    #     # Copy selected Gaussians
    #     new_properties = {}
    #     for key, value in gaussian_model.properties.items():
    #         new_properties[key] = value[selected_pts_mask]

    #     # Update optimizers and properties
    #     self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _split_means_and_scales(self, gaussian_model, selected_pts_mask, N):
        scales = gaussian_model.get_scales()

        stds = scales[selected_pts_mask]

        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)

        samples = stds * mask * 3

        reduction = self.config.opacity_reduction
        rate = self.config.split_distance
        x1 = samples * rate
        rate_w = 1 - rate
        rate_h = math.sqrt(1 - rate * rate)
        x1 = torch.cat([x1, -x1], dim=0)

        rots = build_rotation(gaussian_model.get_property("rotations")[selected_pts_mask]).repeat(N, 1, 1)
        # Split means and scales, they are a little bit different
        new_means = torch.bmm(rots, x1.unsqueeze(-1)).squeeze(-1) + gaussian_model.get_means()[selected_pts_mask].repeat(N, 1)
        new_scales = gaussian_model.scale_inverse_activation(stds.scatter(1, max_indices, max_values * rate_w / rate_h).repeat(N, 1) * rate_h)
        new_opacities = gaussian_model.opacity_inverse_activation(gaussian_model.get_opacities()[selected_pts_mask] * reduction).repeat(N, 1)

        new_properties = {
            "means": new_means,
            "scales": new_scales,
            "opacities": new_opacities,
        }

        return new_properties

    def _long_axis_split(self, gaussian_model, optimizers, selected_pts_mask, N: int = 2):
        device = gaussian_model.get_property("means").device

        self.log_metric("n_split", selected_pts_mask.sum())

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def opacity_reg(
            self,
            outputs,
            batch,
            gaussian_model,
            global_step,
            pl_module,
            metrics,
            pbar,
    ):
        if global_step < self.config.opacity_reg_from:
            return
        if global_step > self.config.opacity_reg_until:
            return
        if ((global_step - 1) % self.config.opacity_reg_interval) != 0:
            return
        # already lower the budget?
        if gaussian_model.n_gaussians <= self.config.budget:
            return

        if self.opacity_min is None:
            sorted_values, _ = torch.sort(gaussian_model.get_opacities().flatten())
            # find the min opacity that should survive
            value = sorted_values[gaussian_model.n_gaussians - self.config.budget].item()
            self.opacity_min = value * 0.8
            print("opacity_min={}".format(self.opacity_min))
        elif ((global_step - 1) % 100) == 0:
            # select the goal based on current number of iteration, where the goal is proportional to the iteration
            opacity_goal = (1 - (global_step - self.config.opacity_reg_from) / (self.config.opacity_reg_until - self.config.opacity_reg_from - 1000)) * self.opacity_min
            if opacity_goal < 0:
                opacity_goal = 0
            sorted_values, _ = torch.sort(gaussian_model.get_opacities().flatten())
            # select opacity min using the goal
            value = sorted_values[gaussian_model.n_gaussians - self.config.budget].item()
            # update the lr
            if value < opacity_goal * 0.9:
                # Reduce the LR when the value is close to the goal
                self.current_opacity_reg_weight = self.current_opacity_reg_weight * 0.8
            elif value > opacity_goal * 1.1:
                # Otherwise increase the LR
                self.current_opacity_reg_weight = self.current_opacity_reg_weight * 1.2

            # print("current_opacity_reg_weight={}".format(self.current_opacity_reg_weight.item()))

        if global_step < self.config.opacity_reg_from + self.config.opacity_reg_prior_free_steps:
            rate_l = torch.max(torch.ones_like(gaussian_model.get_opacity) * 0.05, 1 - gaussian_model.get_opacity)  # (1 - opacity).min_clamp(0.05)
            reg_loss = self.current_opacity_reg_weight * (torch.mean((gaussian_model.opacities + 20) / rate_l)) ** 2  # the higher the opacity of a primitive, the more it weights the mean value, and the larger its decay rate
        else:
            reg_loss = 3 * self.current_opacity_reg_weight * (torch.mean(gaussian_model.opacities) + 20) ** 2  # uniform gradient, with a higher weight

        metrics["loss"] = metrics["loss"] + reg_loss
        metrics["opacity_reg"] = reg_loss
        pbar["opacity_reg"] = False

    def natural_selection(
            self,
            outputs,
            batch,
            gaussian_model,
            global_step,
            pl_module,
    ):
        if global_step < self.config.opacity_reg_from:
            return

        if global_step > self.config.opacity_reg_until:
            return

        # already lower the budget?
        if gaussian_model.n_gaussians <= self.config.budget:
            return

        # is close to the budget?
        if (global_step != self.config.opacity_reg_from and gaussian_model.n_gaussians < self.config.budget * 1.05) or global_step == self.config.opacity_reg_until:
            self.final_prune(gaussian_model, pl_module.gaussian_optimizers)
            self.prune_iter.fill_(global_step)
            return

        if global_step == self.config.opacity_reg_from:
            self.update_opacity_lr(4)

        if global_step % self.config.opacity_reg_interval == 0 and global_step >= self.config.opacity_reg_from + 1000:
            with torch.no_grad():
                prune_mask = gaussian_model.get_opacities().squeeze(-1) < self.config.natural_selection_min_opacity
            self.log_metric("natural_selection_prune", prune_mask.sum())
            self._prune_points(prune_mask, gaussian_model, pl_module.gaussian_optimizers)

    def restore_opacity_lr_without_pruning(
        self,
        outputs,
        batch,
        gaussian_model,
        global_step,
        pl_module,
    ):
        # restore LR if there are no pruning in 1000 steps
        if global_step == self.prune_iter + self.config.opacity_reg_restore_lr_after:
            self.update_opacity_lr(1)

    def final_prune(self, gaussian_model, optimizers):
        final_budget = self.config.budget

        imp_score = gaussian_model.get_opacity.squeeze()
        if imp_score.shape[0] <= final_budget:
            return
        sampled_indices = torch.multinomial(imp_score, final_budget, replacement=False)
        mask = imp_score > -99999999
        mask[sampled_indices] = False
        self._prune_points(mask, gaussian_model, optimizers)

        print("Final pruning activated")

    def _set_opacity_lr(self, factor):
        for opt in self.avoid_state_dict["pl"].gaussian_optimizers:
            for param_group in opt.param_groups:
                if param_group["name"] == "opacities":
                    param_group["lr"] = self.avoid_state_dict["pl"].gaussian_model.config.optimization.opacities_lr * factor
                    break

    def update_opacity_lr(self, factor):
        if factor == self.opacity_lr_factor:
            return

        self._set_opacity_lr(factor)

        # print("opacity_lr_factor={}".format(factor))

        self.opacity_lr_factor.fill_(factor)

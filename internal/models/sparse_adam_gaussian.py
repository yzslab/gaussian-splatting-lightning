from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel

from diff_t3dgs_rasterization import SparseGaussianAdam
import torch
from torch.optim.optimizer import _use_grad_for_differentiable


class SparseAdamAdapter(SparseGaussianAdam):
    def set_vis_and_N(self, vis, N):
        self.visibility = vis
        self.N = N

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        super().step(self.visibility, self.N)

        return loss


@dataclass
class VanillaGaussianWithSparseAdam(VanillaGaussian):
    def instantiate(self, *args, **kwargs) -> "VanillaGaussianWithSparseAdamModel":
        return VanillaGaussianWithSparseAdamModel(self)


class VanillaGaussianWithSparseAdamModel(VanillaGaussianModel):
    def _set_vis_and_N_hook(self, outputs, batch, gaussian_model, global_step, pl_module):
        visibility_filter = outputs["visibility_filter"]
        for i in self.sparse_adam_optimizers:
            i.set_vis_and_N(visibility_filter, visibility_filter.shape[0])

    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        # TODO: avoid code duplication

        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent
        assert spatial_lr_scale > 0

        optimization_config = self.config.optimization

        # the param name and property name must be identical

        # means
        means_lr_init = optimization_config.means_lr_init * spatial_lr_scale
        means_optimizer = SparseAdamAdapter(
            [{'params': [self.gaussians["means"]], "name": "means"}],
            lr=means_lr_init,
            eps=1e-15,
        )
        # TODO: other scheduler may not contain `lr_final`, but does not need to change scheduler currently
        optimization_config.means_lr_scheduler.lr_final *= spatial_lr_scale
        means_scheduler = optimization_config.means_lr_scheduler.instantiate().get_scheduler(
            means_optimizer,
            means_lr_init,
        )

        # the params with constant LR
        l = [
            {'params': [self.gaussians["shs_dc"]], 'lr': optimization_config.shs_dc_lr, "name": "shs_dc"},
            {'params': [self.gaussians["shs_rest"]], 'lr': optimization_config.shs_rest_lr, "name": "shs_rest"},
            {'params': [self.gaussians["opacities"]], 'lr': optimization_config.opacities_lr, "name": "opacities"},
            {'params': [self.gaussians["scales"]], 'lr': optimization_config.scales_lr, "name": "scales"},
            {'params': [self.gaussians["rotations"]], 'lr': optimization_config.rotations_lr, "name": "rotations"},
        ]
        constant_lr_optimizer = SparseAdamAdapter(l, lr=0.0, eps=1e-15)

        print("spatial_lr_scale={}, learning_rates=".format(spatial_lr_scale))
        print("  means={}->{}".format(means_lr_init, optimization_config.means_lr_scheduler.lr_final))
        for i in l:
            print("  {}={}".format(i["name"], i["lr"]))

        # store optimizers
        self.sparse_adam_optimizers = [
            means_optimizer,
            constant_lr_optimizer,
        ]
        # add a hook for the optimizer adapter
        module.on_after_backward_hooks.append(self._set_vis_and_N_hook)

        return [means_optimizer, constant_lr_optimizer], [means_scheduler]

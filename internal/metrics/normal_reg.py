from dataclasses import dataclass
import torch
from gsplat.utils import depth_to_normal

from .metric import Metric, MetricModule
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class NormalRegMixin:
    normal_reg_lambda: float = 0.05

    flatten_reg: float = 0.02

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError()


class NormalRegModuleMixin:
    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)
        if stage == "fit":
            with torch.no_grad():
                pl_module.gaussian_model.gaussians["rotations"].copy_(torch.rand_like(pl_module.gaussian_model.gaussians["rotations"]))
                pl_module.gaussian_model.gaussians["scales"][..., -1] -= torch.log(torch.tensor(5., dtype=torch.float, device=pl_module.gaussian_model.gaussians["scales"].device))

    def get_normal_reg_metrics(self, gaussian_model, outputs, pl_module, metrics, pbar):
        exp_depth = outputs["exp_depth"]
        w2c, K, _ = outputs["preprocessed_camera"]
        normals_from_depth = depth_to_normal(
            exp_depth.detach().permute(1, 2, 0),
            torch.linalg.inv(w2c[0]),
            K[0],
        ).permute(2, 0, 1) * outputs["alpha"].squeeze(0).detach()
        normal_error = (1 - (outputs["normal"] * normals_from_depth).sum(dim=0)).mean()

        normal_loss = normal_error * self.config.normal_reg_lambda
        flatten_loss = gaussian_model.get_scales()[..., -1].mean() * self.config.flatten_reg

        metrics["loss"] = metrics["loss"] + normal_loss + flatten_loss
        metrics["normal_loss"] = normal_loss
        metrics["flatten_loss"] = flatten_loss
        pbar["normal_loss"] = False
        pbar["flatten_loss"] = False

    def get_train_metrics(
            self,
            pl_module,
            gaussian_model,
            step: int,
            batch,
            outputs,
    ):
        metrics, pbar = super().get_train_metrics(
            pl_module,
            gaussian_model,
            step,
            batch,
            outputs,
        )

        self.get_normal_reg_metrics(gaussian_model, outputs, pl_module, metrics, pbar)

        return metrics, pbar


@dataclass
class VanillaWithNormalRegMetrics(NormalRegMixin, VanillaMetrics):
    def instantiate(self, *args, **kwargs):
        return VanillaWithNormalRegMetricsModule(self)


class VanillaWithNormalRegMetricsModule(NormalRegModuleMixin, VanillaMetricsImpl):
    pass

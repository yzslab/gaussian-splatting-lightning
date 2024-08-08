from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000


@dataclass
class HasInverseDepthMetrics(VanillaMetrics):
    depth_loss_type: Literal["l1", "l2", "kl"] = "l1"

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    def instantiate(self, *args, **kwargs) -> "HasInverseDepthMetricsModule":
        # TODO: implement other types
        assert self.depth_loss_type == "l1"

        return HasInverseDepthMetricsModule(self)


class HasInverseDepthMetricsModule(VanillaMetricsImpl):
    config: HasInverseDepthMetrics

    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        self._get_disparity_loss = self._l1_metric

    def _l1_metric(self, a, b):
        return torch.abs(a - b).mean()

    def get_disparity_metric(self, batch, outputs):
        camera, _, gt_disparity = batch

        if gt_disparity is None:
            return 0.

        return self._get_disparity_loss(gt_disparity, outputs["inverse_depth"].squeeze(0))

    def get_weight(self, step: int):
        return self.config.depth_loss_weight.init * (self.config.depth_loss_weight.final_factor ** min(step / self.config.depth_loss_weight.max_steps, 1))

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        d_reg_weight = self.get_weight(step)
        d_reg = self.get_disparity_metric(batch, outputs) * d_reg_weight

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        metrics["d_w"] = d_reg_weight
        pbar["d_reg"] = True
        pbar["d_w"] = True

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)

        d_reg = self.get_disparity_metric(batch, outputs)

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        pbar["d_reg"] = True

        return metrics, pbar

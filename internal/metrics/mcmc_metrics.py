"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch

from .metric import MetricImpl
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class MCMCMetricsMixin:
    mcmc_reg_until_iter: int = -1

    opacity_reg: float = 0.01

    scale_reg: float = 0.01

    reg_weight_decay: float = 1.
    """The actual values of the opacity_reg and scale_reg will be divided by reg_weight_decay"""


class MCMCMetricsModuleMixin:
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        reg_weight_decay = min(self.config.reg_weight_decay, 5.)

        # calculate the weigh
        self.opacity_reg_weight = self.config.opacity_reg / reg_weight_decay
        self.scale_reg_weight = self.config.scale_reg / reg_weight_decay

        print("[MCMC]opacity_reg_weight={}, scale_reg_weight={}".format(self.opacity_reg_weight, self.scale_reg_weight))

    def reg_loss(self, gaussian_model, basic_metrics: Tuple[Dict[str, Any], Dict[str, bool]]):
        # opacity
        opacity_reg_loss = 0.
        opacity_reg_loss = self.opacity_reg_weight * torch.abs(gaussian_model.get_opacity).mean()
        # scale
        scale_reg_loss = 0.
        if self.scale_reg_weight > 0:
            scale_reg_loss = self.scale_reg_weight * torch.abs(gaussian_model.get_scaling).mean()

        basic_metrics[0]["loss"] = basic_metrics[0]["loss"] + opacity_reg_loss + scale_reg_loss
        basic_metrics[0]["o_reg"] = opacity_reg_loss
        basic_metrics[0]["s_reg"] = scale_reg_loss

        basic_metrics[1]["o_reg"] = False
        basic_metrics[1]["s_reg"] = False

        return basic_metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        if self.config.mcmc_reg_until_iter >= 0 and step >= self.config.mcmc_reg_until_iter:
            return basic_metrics
        return self.reg_loss(gaussian_model, basic_metrics)

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        return self.reg_loss(gaussian_model, basic_metrics)


@dataclass
class MCMCMetrics(MCMCMetricsMixin, VanillaMetrics):
    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return MCMCMetricsImpl(self)


class MCMCMetricsImpl(MCMCMetricsModuleMixin, VanillaMetricsImpl):
    pass

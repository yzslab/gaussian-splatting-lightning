from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from .inverse_depth_metrics import HasInverseDepthMetrics, HasInverseDepthMetricsModule
from .ground_reg_metrics import GroundRegMetricConfig, GroundRegMetricModuleMixin


@dataclass
class ScaleRegularizationMetricsMixin:
    """Avoid large and highly anisotropic Gaussians"""

    scale_reg_from: int = 3100
    """Should be at least after the first opacity reset + a densify interval, i.e. 3100 by default"""

    max_scale: float = -1
    """Should be greater than `percent_dense * camera_extent`, or splitting of densification will not work"""

    max_scale_outside_partition: Optional[float] = None

    max_scale_ratio: float = 10

    scale_reg_lambda: float = 0.05

    scale_ratio_reg_lambda: float = 0.05

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        assert self.scale_reg_from >= 0
        # if self.scale_reg_lambda > 0.:
        # assert self.max_scale > 0
        if self.scale_ratio_reg_lambda > 0.:
            assert self.max_scale_ratio > 0

    def instantiate(self, *args, **kwargs) -> "ScaleRegularizationMetricsModuleMixin":
        raise NotImplementedError()


class ScaleRegularizationMetricsModuleMixin:
    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)
        if self.config.max_scale <= 0 and stage == "fit":
            self.config.max_scale = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * 1.1
            assert self.config.max_scale > 0.
            print("max_scale={}".format(self.config.max_scale))

    def get_scale_regularization_metrics(self, gaussian_model, pl_module, metrics, pbar):
        scales = gaussian_model.get_scales()
        sorted_scales = torch.sort(scales, dim=-1).values

        max_scales = sorted_scales[:, -1]
        mid_scales = sorted_scales[:, -2]

        n_over_scales = 0
        over_scale_loss = 0.
        is_over_scales = None
        if self.config.scale_reg_lambda > 0.:
            if self.config.max_scale_outside_partition is None:
                upper_scale = self.config.max_scale
            else:
                is_outside_partition = pl_module.store.distance_factors.to(device=pl_module.device) > 0.5
                upper_scale = torch.where(
                    is_outside_partition,
                    self.config.max_scale_outside_partition,
                    self.config.max_scale,
                ).unsqueeze(-1)
            is_over_scales = scales.detach() > upper_scale

            # is_over_scales = scales.detach() > self.config.max_scale
            # if self.config.max_scale_outside_partition is not None:
            #     is_outside_partition = pl_module.store.distance_factors > 0.5
            #     is_over_outside_scale = torch.logical_or(
            #         scales.detach() > self.config.max_scale_outside_partition,  # overscale is True
            #         torch.logical_not(is_outside_partition),  # inside all True
            #     )
            #     is_over_scales = torch.logical_and(is_over_scales, is_over_outside_scale)  # mask out not overscale outside

            n_over_scales = is_over_scales.sum().float()
            over_scale_loss = (scales * is_over_scales).sum() / (n_over_scales + 1) * self.config.scale_reg_lambda

        n_over_ratios = 0
        over_ratio_loss = 0.
        is_over_ratios = None
        if self.config.scale_ratio_reg_lambda > 0.:
            scale_ratios = max_scales / (mid_scales + 1e-8)
            is_over_ratios = scale_ratios.detach() > self.config.max_scale_ratio
            n_over_ratios = is_over_ratios.sum().float()
            over_ratio_loss = (scale_ratios * is_over_ratios).sum() / (n_over_ratios + 1) * self.config.scale_ratio_reg_lambda

        metrics["loss"] = metrics["loss"] + over_scale_loss + over_ratio_loss
        metrics["scale_reg"] = over_scale_loss
        metrics["scale_ratio_reg"] = over_ratio_loss
        metrics["n_over_scales"] = n_over_scales
        metrics["n_over_ratios"] = n_over_ratios
        with torch.no_grad():
            metrics["max_scale"] = max_scales.max()
            metrics["max_ratio"] = scale_ratios.max()
            metrics["mean_ratio"] = scale_ratios.mean()

        pbar["scale_reg"] = False
        pbar["scale_ratio_reg"] = False
        pbar["n_over_scales"] = False
        pbar["n_over_ratios"] = False
        pbar["max_scale"] = False
        pbar["max_ratio"] = False
        pbar["mean_ratio"] = False

        return sorted_scales, is_over_scales, is_over_ratios

    def get_train_metrics(
            self,
            pl_module,
            gaussian_model,
            step: int,
            batch,
            outputs,
    ) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(
            pl_module,
            gaussian_model,
            step,
            batch,
            outputs,
        )

        if step >= self.config.scale_reg_from:
            self.get_scale_regularization_metrics(gaussian_model, pl_module, metrics, pbar)

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        self.get_scale_regularization_metrics(gaussian_model, pl_module, metrics, pbar)

        return metrics, pbar


@dataclass
class ScaleRegularizationMetrics(ScaleRegularizationMetricsMixin, VanillaMetrics):
    def instantiate(self, *args, **kwargs) -> "ScaleRegularizationMetricsModuleMixin":
        return ScaleRegularizationMetricsModule(self)


class ScaleRegularizationMetricsModule(ScaleRegularizationMetricsModuleMixin, VanillaMetricsImpl):
    pass


@dataclass
class ScaleRegularizationWithDepthMetrics(ScaleRegularizationMetricsMixin, HasInverseDepthMetrics):
    def instantiate(self, *args, **kwargs) -> "ScaleRegularizationWithDepthMetricsModule":
        return ScaleRegularizationWithDepthMetricsModule(self)


class ScaleRegularizationWithDepthMetricsModule(ScaleRegularizationMetricsModuleMixin, HasInverseDepthMetricsModule):
    pass


@dataclass
class ScaleRegularizationWithGroundMetrics(
    ScaleRegularizationMetricsMixin,
    GroundRegMetricConfig,
    VanillaMetrics,
):
    def instantiate(self, *args, **kwargs) -> "ScaleRegularizationWithGroundMetricsModule":
        return ScaleRegularizationWithGroundMetricsModule(self)


class ScaleRegularizationWithGroundMetricsModule(
    ScaleRegularizationMetricsModuleMixin,
    GroundRegMetricModuleMixin,
    VanillaMetricsImpl,
):
    pass


@dataclass
class ScaleRegularizationWithGroundDepthMetrics(
    ScaleRegularizationMetricsMixin,
    GroundRegMetricConfig,
    HasInverseDepthMetrics,
):
    def instantiate(self, *args, **kwargs) -> "ScaleRegularizationWithGroundDepthMetricsModule":
        return ScaleRegularizationWithGroundDepthMetricsModule(self)


class ScaleRegularizationWithGroundDepthMetricsModule(
    ScaleRegularizationMetricsModuleMixin,
    GroundRegMetricModuleMixin,
    HasInverseDepthMetricsModule,
):
    pass

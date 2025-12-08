from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000


@dataclass
class HasInverseDepthMetrics(VanillaMetrics):
    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_median_normalization: bool = False

    depth_mean_normalization: bool = False

    depth_output_key: str = "inverse_depth"

    def instantiate(self, *args, **kwargs) -> "HasInverseDepthMetricsModule":
        return HasInverseDepthMetricsModule(self)


class HasInverseDepthMetricsModule(VanillaMetricsImpl):
    config: HasInverseDepthMetrics

    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        if self.config.depth_loss_type == "l1":
            self._get_inverse_depth_loss = self._depth_l1_loss
        elif self.config.depth_loss_type == "l1+ssim":
            self.depth_ssim = StructuralSimilarityIndexMeasure()
            self._get_inverse_depth_loss = self._depth_l1_and_ssim_loss
        elif self.config.depth_loss_type == "l2":
            self._get_inverse_depth_loss = self._depth_l2_loss
        # elif self.config.depth_loss_type == "kl":
        #     self._get_inverse_depth_loss = self._depth_kl_loss
        else:
            raise NotImplementedError()

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l1_and_ssim_loss(self, a, b):
        l1_loss = self._depth_l1_loss(a, b)
        ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])

        return (1 - self.config.depth_loss_ssim_weight) * l1_loss + self.config.depth_loss_ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass

    def get_inverse_depth_metric(self, batch, outputs):
        _, image_info, gt_inverse_depth_data = batch

        device = batch[0].device

        if gt_inverse_depth_data is None:
            return torch.tensor(0., device=device)
                
        gt_inverse_depth = gt_inverse_depth_data.get(device=device)

        predicted_inverse_depth = outputs[self.config.depth_output_key].squeeze(0)
        if self.config.depth_normalized:
            with torch.no_grad():
                max_depth = predicted_inverse_depth.max()
                min_depth = predicted_inverse_depth.min()
            predicted_inverse_depth = (predicted_inverse_depth - min_depth) / (max_depth - min_depth + 1e-8)
        elif gt_inverse_depth_data.median is not None:
            assert self.config.depth_median_normalization
            predicted_inverse_depth = predicted_inverse_depth / gt_inverse_depth_data.median
        elif self.config.depth_mean_normalization:
            mean_depth = torch.mean(gt_inverse_depth)
            gt_inverse_depth = gt_inverse_depth / mean_depth
            predicted_inverse_depth = predicted_inverse_depth / mean_depth

        if gt_inverse_depth_data.mask is not None:
            gt_inverse_depth = gt_inverse_depth * gt_inverse_depth_data.mask
            predicted_inverse_depth = predicted_inverse_depth * gt_inverse_depth_data.mask

        # mask
        if image_info[-1] is not None:
            mask = image_info[-1].to(torch.uint8)
            gt_inverse_depth = gt_inverse_depth * mask
            predicted_inverse_depth = predicted_inverse_depth * mask

        return self._get_inverse_depth_loss(gt_inverse_depth, predicted_inverse_depth)

    def get_weight(self, step: int):
        return self.config.depth_loss_weight.init * (self.config.depth_loss_weight.final_factor ** min(step / self.config.depth_loss_weight.max_steps, 1))

    def get_predicted_depth(self, pl_module, gaussian_model, batch, outputs):
        predicted_depth = outputs
        if outputs.get(self.config.depth_output_key, None) is None and batch[-1] is not None:
            assert batch[-1].camera is not None
            depth_camera = batch[-1].camera.to_device(pl_module.device)
            predicted_depth = pl_module.renderer(
                depth_camera,
                gaussian_model,
                torch.zeros((3, ), dtype=torch.float32, device=pl_module.device),
                render_types=[
                    self.config.depth_output_key,
                ],
            )

        return predicted_depth

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        d_reg_weight = self.get_weight(step)

        predicted_depth = self.get_predicted_depth(
            pl_module,
            gaussian_model,
            batch,
            outputs,
        )

        d_reg = self.get_inverse_depth_metric(batch, predicted_depth) * d_reg_weight

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        metrics["d_w"] = d_reg_weight
        pbar["d_reg"] = True
        pbar["d_w"] = True

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)

        predicted_depth = self.get_predicted_depth(
            pl_module,
            gaussian_model,
            batch,
            outputs,
        )

        d_reg = self.get_inverse_depth_metric(batch, predicted_depth)

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        pbar["d_reg"] = True

        return metrics, pbar

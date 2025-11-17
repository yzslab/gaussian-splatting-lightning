from dataclasses import dataclass
import torch
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class BackgroundRemoval(VanillaMetrics):
    background_removal_from: int = 7_000

    background_removal_depth_key: str = "hard_inverse_depth"

    background_removal_weight: float = 0.1

    def instantiate(self, *args, **kwargs):
        return BackgroundRemovalModule(self)


class BackgroundRemovalModule(VanillaMetricsImpl):
    def get_train_metrics(self, pl_module, gaussian_model, step, batch, outputs):
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        if step < self.config.background_removal_from:
            return metrics, pbar

        mask = batch[1][2]
        if mask is None:
            return metrics, pbar

        bkg_mask = torch.logical_not(mask)

        pred_bkg_depth = outputs[self.config.background_removal_depth_key] * bkg_mask[:1, ...]
        bkg_removal_loss = pred_bkg_depth.mean() * self.config.background_removal_weight

        metrics["loss"] = metrics["loss"] + bkg_removal_loss
        metrics["bkg_removal"] = bkg_removal_loss
        pbar["bkg_removal"] = False

        return metrics, pbar

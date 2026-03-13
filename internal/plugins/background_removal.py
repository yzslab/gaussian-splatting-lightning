from dataclasses import dataclass
import torch
from .plugin import Plugin, PluginModule


@dataclass
class BackgroundRemoval(Plugin):
    background_removal_from: int = 7_000

    background_removal_depth_key: str = "hard_inverse_depth"

    background_removal_weight: float = 0.1

    def instantiate(self, *args, **kwargs):
        return BackgroundRemovalModule(self)


class BackgroundRemovalModule(PluginModule):
    def setup(self, pl_module):
        pl_module.extra_train_metrics.append(self.remove_background)

    def remove_background(self, outputs, batch, gaussian_model, global_step, pl_module, metrics, pbar):
        if global_step < self.config.background_removal_from:
            return

        mask = batch[1][2]
        if mask is None:
            return

        bkg_mask = torch.logical_not(mask)

        pred_bkg_depth = outputs[self.config.background_removal_depth_key] * bkg_mask[:1, ...]
        bkg_removal_loss = pred_bkg_depth.mean() * self.config.background_removal_weight

        metrics["loss"] = metrics["loss"] + bkg_removal_loss
        metrics["bkg_removal"] = bkg_removal_loss
        pbar["bkg_removal"] = False

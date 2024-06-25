from typing import Tuple, Union, List, Dict, Optional
import torch
from lightning import LightningModule
from internal.configs.instantiate_config import InstantiatableConfig


class DensityControllerImpl(torch.nn.Module):
    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        pass

    def get_train_metrics(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> Tuple[Dict, Dict[str, bool]]:
        """
            return
                The fist dict contains metrics, the second one indicates whether show on progress bar.
                If `loss` key exists in the dict, it will be added to the total loss, other values are only for logging.
        """

        return {}, {}

    def get_validation_metrics(self, outputs: dict, batch, gaussian_model, pl_module: LightningModule) -> Tuple[Dict, Dict[str, bool]]:
        return {}, {}

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        pass

    def configure_optimizers(self, pl_module: LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        return None, None

    def on_load_checkpoint(self, module, checkpoint):
        pass


class DensityController(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        pass

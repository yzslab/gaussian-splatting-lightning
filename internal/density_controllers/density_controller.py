from dataclasses import dataclass
from typing import Tuple, Union, List, Dict, Optional
import torch
from lightning import LightningModule
from internal.configs.instantiate_config import InstantiatableConfig


class DensityControllerImpl(torch.nn.Module):
    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        pass

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


@dataclass
class DensityController(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        pass

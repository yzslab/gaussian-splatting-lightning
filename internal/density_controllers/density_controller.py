from typing import Tuple, Union, List, Dict, Optional, Type
import torch
from lightning import LightningModule
from internal.configs.instantiate_config import InstantiatableConfig


class DensityControllerImpl(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    # TODO: Gaussian Model should be provided on setup or here?
    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        pass

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        pass

    def on_load_checkpoint(self, module, checkpoint):
        pass


class DensityController(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        pass

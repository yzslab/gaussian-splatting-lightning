from dataclasses import dataclass
from lightning import LightningModule
from .density_controller import DensityController, DensityControllerImpl


@dataclass
class StaticDensityController(DensityController):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return StaticDensityControllerImpl(self)


class StaticDensityControllerImpl(DensityControllerImpl):
    def forward(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        return

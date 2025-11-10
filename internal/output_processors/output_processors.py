from typing import Tuple, List, Optional
from internal.configs.instantiate_config import InstantiatableConfig
from dataclasses import dataclass


@dataclass
class OutputProcessor(InstantiatableConfig):
    pass


@dataclass
class VanillaOutputProcessor(OutputProcessor):
    def instantiate(self):
        return VanillaOutputProcessorModule()


class VanillaOutputProcessorModule:
    def setup(self, stage: str, pl_module=None, *args, **kwargs) -> None:
        return

    def training_setup(self, pl_module) -> Tuple[Optional[List], Optional[List]]:
        return None, None

    def forward(self, camera, outputs) -> None:
        return

    def training_forward(self, batch, outputs) -> None:
        return

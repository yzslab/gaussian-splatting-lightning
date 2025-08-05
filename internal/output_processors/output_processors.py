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
    def training_setup(self, pl_module) -> Tuple[Optional[List], Optional[List]]:
        return None, None

    def training_forward(self, batch, outputs) -> None:
        return

from dataclasses import dataclass
from internal.configs.instantiate_config import InstantiatableConfig


@dataclass
class Store(InstantiatableConfig):
    pass


class StoreModule:
    def setup(self, stage: str, pl_module):
        pass

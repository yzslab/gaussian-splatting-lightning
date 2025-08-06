from dataclasses import dataclass
from . import Store, StoreModule


@dataclass
class VanillaStore(Store):
    def instantiate(self, *args, **kwargs):
        return VanillaStoreModule(self)


class VanillaStoreModule(StoreModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

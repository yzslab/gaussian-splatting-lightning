from typing import Type, Any
from dataclasses import dataclass


@dataclass
class InstantiatableConfig:
    def instantiate(self, *args, **kwargs) -> Any:
        pass

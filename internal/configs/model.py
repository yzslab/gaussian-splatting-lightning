from typing import Tuple
from dataclasses import dataclass
from internal.configs.optimization import OptimizationParams


@dataclass
class ModelParams:
    optimization: OptimizationParams
    sh_degree: int = 3

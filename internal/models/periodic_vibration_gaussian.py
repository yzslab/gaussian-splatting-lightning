from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union, List

import torch
from internal.schedulers import Scheduler, ExponentialDecayScheduler
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig as VanillaOptimizationConfig


@dataclass
class OptimizationConfig(VanillaOptimizationConfig):
    time_lr_init: float = 0.0008
    time_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.000008,
            "max_steps": 30_000,
        },
    })

    scale_time_lr: float = 0.002

    velocity_lr: float = 0.001


@dataclass
class PeriodicVibrationGaussian(VanillaGaussian):
    time_init: float = 0.2

    time_duration: Tuple[float, float] = (-0.5, 0.5)

    cycle: float = 0.2

    velocity_decay: float = 1.0

    optimization: OptimizationConfig = field(default_factory=lambda: VanillaOptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "PeriodicVibrationGaussianModel":
        return PeriodicVibrationGaussianModel(self)


class PeriodicVibrationGaussianModel(VanillaGaussianModel):
    config: PeriodicVibrationGaussian

    def get_extra_property_names(self):
        return [
            "time",  # [N, 1]
            "scale_time",  # [N, 1]
            "velocity",  # [N, 3]
        ]

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        # TODO: initialization
        return

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict["time"] = torch.empty((n, 1), dtype=torch.float)
        property_dict["scale_time"] = torch.empty((n, 1), dtype=torch.float)
        property_dict["velocity"] = torch.empty((n, 3), dtype=torch.float)

    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        # TODO: setup optimizers and schedulers
        return super().training_setup(module)

    def get_time(self):
        return self.gaussians["time"]

    def scale_time_activation(self, v):
        return torch.exp(v)

    def scale_time_inverse_activation(self, v):
        return torch.log(v)

    def get_scale_time(self):
        return self.scale_time_activation(self.gaussians["scale_time"])

    def get_velocity(self):
        return self.gaussians["velocity"]

    def get_mean_SHM(self, t):
        a = 1 / self.config.cycle * torch.pi * 2
        return self.get_means() + self.get_velocity() * torch.sin((t - self.get_time()) * a) / a

    def get_marginal_t(self, timestamp):
        return torch.exp(-0.5 * (self.get_time() - timestamp) ** 2 / self.get_scale_time() ** 2)

    def get_instant_velocity(self):
        return self.get_velocity() * torch.exp(-self.get_scale_time() / self.config.cycle / 2 * self.config.velocity_decay)

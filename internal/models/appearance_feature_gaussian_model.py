from typing import Literal, Dict, Union
from dataclasses import dataclass

import numpy as np
import torch

from .vanilla_gaussian_model import VanillaGaussian, VanillaGaussianModel


@dataclass
class AppearanceFeatureGaussian(VanillaGaussian):
    appearance_feature_dims: int = 64

    appearance_feature_init_type: Literal["zero", "rand", "normal"] = "zero"

    def instantiate(self, *args, **kwargs) -> "VanillaGaussianModel":
        return AppearanceFeatureGaussianModel(self)


class AppearanceFeatureGaussianModel(VanillaGaussianModel):
    config: AppearanceFeatureGaussian

    _appearance_feature_name = "appearance_features"

    def __init__(self, config: VanillaGaussian) -> None:
        super().__init__(config)
        self._names = tuple(list(self._names) + [self._appearance_feature_name])

    def create_appearance_features(self, n: int):
        appearance_features = torch.empty((n, self.config.appearance_feature_dims), dtype=torch.float)
        if self.config.appearance_feature_init_type == "zero":
            torch.nn.init.zeros_(appearance_features)
        elif self.config.appearance_feature_init_type == "rand":
            appearance_features.copy_(torch.rand_like(appearance_features))
        elif self.config.appearance_feature_init_type == "normal":
            torch.nn.init.normal_(appearance_features)

        return torch.nn.Parameter(appearance_features.requires_grad_(True))

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict[self._appearance_feature_name] = self.create_appearance_features(xyz.shape[0])

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict[self._appearance_feature_name] = self.create_appearance_features(n)

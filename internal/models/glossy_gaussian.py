from dataclasses import dataclass, field
import torch
from internal.utils.sh_utils import RGB2SH, eval_sh_decomposed
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig


@dataclass
class GlossyOptimizationConfig(OptimizationConfig):
    opacities_lr: float = 0.0025

    opacity_rest_lr: float = 0.0025 / 20


@dataclass
class GlossyGaussian(VanillaGaussian):
    optimization: GlossyOptimizationConfig = field(default_factory=lambda: GlossyOptimizationConfig())

    opacity_sh_degree: int = 3

    def instantiate(self, *args, **kwargs):
        return GlossyGaussianModel(self)


class GlossyGaussianModel(VanillaGaussianModel):
    _opacity_rest_name: str = "opacity_rest"

    _opacity_max_name: str = "opacity_max"

    def get_extra_property_names(self):
        return super().get_extra_property_names() + [self._opacity_rest_name, self._opacity_max_name]

    def before_setup_set_properties_from_pcd(self, xyz, rgb, property_dict, *args, **kwargs):
        super().before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)

        opacity_shs = torch.zeros((property_dict[self._mean_name].shape[0], 1, (self.config.opacity_sh_degree + 1) ** 2), dtype=torch.float)
        opacity_shs[..., 0] = RGB2SH(torch.full((
            opacity_shs.shape[0],
            1,
        ), self.config.init_opacity, dtype=torch.float))

        property_dict[self._opacity_name] = opacity_shs[:, :, :1].transpose(1, 2).contiguous()  # [N, N_DC, C]
        property_dict[self._opacity_rest_name] = opacity_shs[:, :, 1:].transpose(1, 2).contiguous()  # [N, N_REST, C]
        property_dict[self._opacity_max_name] = torch.zeros((property_dict[self._mean_name].shape[0], ), dtype=torch.float)

    def before_setup_set_properties_from_number(self, n, property_dict, *args, **kwargs):
        super().before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)

        opacity_shs = torch.zeros((property_dict[self._mean_name].shape[0], 1, (self.config.opacity_sh_degree + 1) ** 2), dtype=torch.float)

        property_dict[self._opacity_name] = opacity_shs[:, :, :1].transpose(1, 2).contiguous()
        property_dict[self._opacity_rest_name] = opacity_shs[:, :, 1:].transpose(1, 2).contiguous()
        property_dict[self._opacity_max_name] = torch.zeros((property_dict[self._mean_name].shape[0], ), dtype=torch.float)

    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        # TODO: some components call inverse_sigmoid directly
        return eval_sh_decomposed(
            0,
            shs_dc=opacities,
            shs_rest=None,
            dirs=None,
        ) + 0.5

    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        return RGB2SH(opacities).unsqueeze(-1)

    def get_opacity_features(self):
        return torch.cat((self.properties[self._opacity_name], self.properties[self._opacity_rest_name]), dim=1)

    def get_view_dep_opacities(self, dirs, masks=None):
        # TODO: reimplement in CUDA kernel
        # NOTE: dirs must be normalized

        opacities = eval_sh_decomposed(
            self.active_sh_degree,
            shs_dc=self.properties[self._opacity_name],
            shs_rest=self.properties[self._opacity_rest_name],
            dirs=dirs,
        ) + 0.5

        return opacities

    def training_setup(self, module):
        optimizers, schedulers = super().training_setup(module)

        opacity_rest_optimizer = torch.optim.Adam(
            params=[{"name": self._opacity_rest_name, "params": self.properties[self._opacity_rest_name]}],
            lr=self.config.optimization.opacity_rest_lr,
        )

        optimizers.append(opacity_rest_optimizer)

        return optimizers, schedulers

    def get_opacity_max(self) -> torch.Tensor:
        return self.gaussians[self._opacity_max_name]

    def update_opacity_max(self, new_opacities: torch.Tensor):
        with torch.no_grad():
            self.gaussians[self._opacity_max_name] = torch.maximum(
                self.get_opacities().squeeze(-1),
                new_opacities,
            )

    def reset_opacity_max(self):
        with torch.no_grad():
            self.gaussians[self._opacity_max_name].copy_(self.get_opacities().squeeze(-1))

    def pre_activate_all_properties(self):
        self.is_pre_activated = True

        self.scales = self.get_scales()
        self.rotations = self.get_rotations()

        # concat `shs_dc` and `shs_rest` and store it to dict, then remove `shs_dc` and `shs_rest`
        names = list(self._names)
        # concat
        self.gaussians["shs"] = self.get_shs()
        names.append("shs")
        # remove `shs_dc`
        del self.gaussians["shs_dc"]
        names.remove("shs_dc")
        # remove `shs_rest`
        del self.gaussians["shs_rest"]
        names.remove("shs_rest")
        # replace `get_shs` interface
        self.get_shs = self._get_shs_from_dict
        # replace `names`
        self._names = tuple(names)

        self.scale_activation = self._return_as_is
        self.scale_inverse_activation = self._return_as_is
        self.rotation_activation = self._return_as_is
        self.rotation_inverse_activation = self._return_as_is

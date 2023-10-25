import torch
from torch import nn
import internal.utils.gaussian_utils as gaussian_utils


class GaussianModelSimplified(nn.Module):
    def __init__(
            self,
            xyz: torch.Tensor,
            features_dc: torch.Tensor,
            features_rest: torch.Tensor,
            scaling: torch.Tensor,
            rotation: torch.Tensor,
            opacity: torch.Tensor,
            active_sh_degree: int,
            device,
    ) -> None:
        super().__init__()

        self._xyz = xyz.to(device)
        # self._features_dc = features_dc
        # self._features_rest = features_rest
        self._scaling = torch.exp(scaling).to(device)
        self._rotation = torch.nn.functional.normalize(rotation).to(device)
        self._opacity = torch.sigmoid(opacity).to(device)

        self._features = torch.cat([features_dc, features_rest], dim=1).to(device)

        self.active_sh_degree = active_sh_degree

    @classmethod
    def construct_from_state_dict(cls, state_dict, active_sh_degree, device):
        init_args = {
            "active_sh_degree": active_sh_degree,
            "device": device,
        }
        for i in state_dict:
            if i.startswith("gaussian_model._") is False:
                continue
            init_args[i[len("gaussian_model._"):]] = state_dict[i]
        return cls(**init_args)

    @classmethod
    def construct_from_ply(cls, ply_path: str, active_sh_degree, device):
        gaussians = gaussian_utils.Gaussian.load_from_ply(ply_path, active_sh_degree).to_parameter_structure()
        return cls(
            active_sh_degree=active_sh_degree,
            device=device,
            xyz=gaussians.xyz,
            opacity=gaussians.opacities,
            features_dc=gaussians.features_dc,
            features_rest=gaussians.features_extra,
            scaling=gaussians.scales,
            rotation=gaussians.rotations,
        )

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

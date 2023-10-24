#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from typing import Any, Tuple, Optional
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.cameras.cameras import Camera
from internal.configs.appearance import AppearanceModelParams
from internal.models.gaussian_model import GaussianModel
from internal.models.appearance_model import AppearanceModel
from internal.utils.sh_utils import eval_sh


class Renderer(torch.nn.Module):
    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        pass

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_setup(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        return None, None


class VanillaRenderer(Renderer):
    def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            # "viewspace_points": screenspace_points,
            # "visibility_filter": radii > 0,
            # "radii": radii,
        }


class AppearanceMLPRenderer(VanillaRenderer):

    def __init__(
            self,
            appearance: AppearanceModelParams,
            compute_cov3D_python: bool = False,
            convert_SHs_python: bool = False,
    ):
        super().__init__(compute_cov3D_python, convert_SHs_python)

        self.appearance_config = appearance

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            appearance: Tuple = None,
    ):
        outputs = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)
        rendered_image = outputs["render"]

        # appearance embedding
        if appearance is not None:
            grayscale_factors, gamma = appearance
        else:
            grayscale_factors, gamma = self.appearance_model.get_appearance(viewpoint_camera.appearance_embedding)

        # apply appearance transform
        rendered_image = torch.pow(rendered_image, gamma)
        rendered_image = rendered_image * grayscale_factors

        # store transformed result
        outputs["render"] = rendered_image

        return outputs

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        appearance = self.appearance_config
        self.appearance_model = AppearanceModel(
            n_input_dims=1,
            n_grayscale_factors=appearance.n_grayscale_factors,
            n_gammas=appearance.n_gammas,
            n_neurons=appearance.n_neurons,
            n_hidden_layers=appearance.n_hidden_layers,
            n_frequencies=appearance.n_frequencies,
            grayscale_factors_activation=appearance.grayscale_factors_activation,
            gamma_activation=appearance.gamma_activation,
        )

    def training_setup(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        appearance_optimizer = torch.optim.Adam(
            [
                {"params": list(self.appearance_model.parameters()), "name": "appearance"}
            ],
            lr=self.appearance_config.optimization.lr,
            eps=self.appearance_config.optimization.eps,
        )
        appearance_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=appearance_optimizer,
            lr_lambda=lambda iter: self.appearance_config.optimization.gamma ** min(iter / self.appearance_config.optimization.max_steps, 1),
            verbose=False,
        )

        return appearance_optimizer, appearance_scheduler


class RGBMLPRenderer(VanillaRenderer):
    def __init__(
            self,
            compute_cov3D_python: bool = False,
            n_neurons: int = 128,
            n_hidden_layers: int = 3,
            lr: float = 1e-4,
            gamma: float = 0.1,
            max_steps: int = 30_000,
    ):
        super().__init__(compute_cov3D_python, convert_SHs_python=False)

        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.gamma = gamma
        self.max_steps = max_steps

    def setup(self, stage: str, **kwargs):
        super().setup(stage, **kwargs)
        import tinycudann as tcnn

        self.rgb_network = tcnn.NetworkWithInputEncoding(
            n_input_dims=1 + 3 + 3 * ((3 + 1) ** 2),  # 1: appearance embedding, 3: view direction, others: SHs
            n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    # encoding appearance embedding
                    {
                        "n_dims_to_encode": 1,
                        "otype": "Frequency",
                        "degree": 6
                    },
                    {
                        "otype": "Identity"
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": self.n_neurons,
                "n_hidden_layers": self.n_hidden_layers,
            },
        )

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        # view directions
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        # spherical harmonics
        shs = pc.get_features
        shs = shs.transpose(1, 2).reshape((shs.shape[0], -1))

        override_color = self.rgb_network(torch.concatenate([
            viewpoint_camera.appearance_embedding.repeat(shs.shape[0]).unsqueeze(-1),
            dir_pp_normalized,
            shs,
        ], dim=-1)).to(torch.float)
        return super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)

    def training_setup(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(self.rgb_network.parameters()), "name": "mlp_renderer"},
            ],
            lr=self.lr,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.gamma ** min(iter / self.max_steps, 1),
            verbose=False,
        )

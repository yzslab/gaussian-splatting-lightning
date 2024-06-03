import math
import torch
from .renderer import Renderer, Camera, GaussianModel
from ds_splat import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from typing import Optional
from internal.utils.sh_utils import eval_sh


class OpenRenderer(Renderer):
    def __init__(
        self, precompute_cov_3d: bool = False, precompute_colors: bool = False
    ):
        super().__init__()

        self.precompute_cov_3d = precompute_cov_3d
        self.precompute_colors = precompute_colors

    def forward(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device=bg_color.device,
            )
            + 0
        )

        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

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
            max_sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means_3d = pc.get_xyz
        means_2d = screenspace_points
        opacity = pc.get_opacity

        scales, rotations, cov_3d_precomp, shs, colors_precomp = (
            self.get_cov_3d_and_colors(
                viewpoint_camera, pc, scaling_modifier, override_color
            )
        )

        rendered_image, radii = rasterizer(
            means_3d=means_3d,
            means_2d=means_2d,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov_3d_precomp=cov_3d_precomp,
        )

        grad_scale = 0.5 * max(
            raster_settings.image_height, raster_settings.image_width
        )

        return {
            "render": rendered_image.permute(2, 0, 1),
            "viewspace_points": screenspace_points,
            "viewspace_points_grad_scale": grad_scale,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_cov_3d_and_colors(
        self,
        viewpoint_camera: Camera,
        pc: GaussianModel,
        scaling_modifier=1.0,
        override_color=None,
    ):
        scales = None
        rotations = None
        cov_3d_precomp = None

        if self.precompute_cov_3d is True:
            cov_3d_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        shs = None
        colors_precomp = None
        if override_color is None:
            if self.precompute_colors is True:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2_to_rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)

                colors_precomp = torch.clamp_min(sh2_to_rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        return scales, rotations, cov_3d_precomp, shs, colors_precomp

    @staticmethod
    def render(
        means_3d: torch.Tensor,
        opacity: torch.Tensor,
        scales: Optional[torch.Tensor],
        rotations: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        active_sh_degree: int,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        colors_precomp: Optional[torch.Tensor] = None,
        cov_3d_precomp: Optional[torch.Tensor] = None,
    ):
        screenspace_points = torch.zeros_like(
            means_3d,
            dtype=means_3d.dtype,
            requires_grad=True,
            device=means_3d.device,
        )

        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=math.tan(viewpoint_camera.fov_x * 0.5),
            tanfovy=math.tan(viewpoint_camera.fov_y * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=active_sh_degree,
            max_sh_degree=3,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means_2d = screenspace_points

        rendered_image, radii = rasterizer(
            means_3d=means_3d,
            means_2d=means_2d,
            shs=features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov_3d_precomp=cov_3d_precomp,
        )

        return {
            "render": rendered_image.permute(2, 0, 1),
            "depth": None,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

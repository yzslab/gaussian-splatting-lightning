from dataclasses import dataclass
from typing import Any, Dict

import math
import torch
import torch.nn.functional as F
import kornia
from gsplat.sh import spherical_harmonics
from gsplat.rasterize import rasterize_gaussians

from . import RendererOutputInfo, RendererOutputTypes
from .renderer import Renderer, RendererConfig
from .gsplat_renderer import GSPlatRenderer, DEFAULT_BLOCK_SIZE
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from internal.model_components.envlight import EnvLight


@dataclass
class PeriodicVibrationGaussianRenderer(RendererConfig):
    env_map_res: int = 1024

    anti_aliased: bool = True

    def instantiate(self, *args, **kwargs) -> "PeriodicVibrationGaussianRendererModule":
        return PeriodicVibrationGaussianRendererModule(self)


class PeriodicVibrationGaussianRendererModule(Renderer):
    def __init__(self, config: PeriodicVibrationGaussianRenderer) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        self.env_map = None
        if self.config.env_map_res > 0:
            self.env_map = EnvLight(resolution=self.config.env_map_res)
        return super().setup(stage, *args, **kwargs)

    def vanilla_forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

        # TODO: `time_shift`
        means3D = pc.get_mean_SHM(viewpoint_camera.time)
        marginal_t = pc.get_marginal_t(viewpoint_camera.time)

        opacities = pc.get_opacities() * marginal_t

        # TODO: prefilter
        mask = marginal_t[:, 0] > 0.05
        masked_means3D = means3D[mask]
        masked_xyz_homo = torch.cat([masked_means3D, torch.ones_like(masked_means3D[:, :1])], dim=1)
        masked_depth = (masked_xyz_homo @ viewpoint_camera.world_to_camera[:, 2:3])
        depth_alpha = torch.zeros(means3D.shape[0], 2, dtype=torch.float32, device=means3D.device)
        depth_alpha[mask] = torch.cat([
            masked_depth,
            torch.ones_like(masked_depth)
        ], dim=1)
        features = depth_alpha

        # rasterize
        # Set up rasterization configuration
        if True:
            # we find that set fov as -1 slightly improves the results
            tanfovx = math.tan(-0.5)
            tanfovy = math.tan(-0.5)
        else:
            tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
            tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=viewpoint_camera.height.int().item(),
            image_width=viewpoint_camera.width.int().item(),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color if self.env_map is not None else torch.zeros(3, device=bg_color.device),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=bg_color.device)
        contrib, rendered_image, rendered_feature, radii = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=pc.get_shs(),
            colors_precomp=None,
            features=features,
            opacities=opacities,
            scales=pc.get_scales(),
            rotations=pc.get_rotations(),
            cov3D_precomp=None,
            mask=mask,
        )

        rendered_other, rendered_depth, rendered_opacity = rendered_feature.split([0, 1, 1], dim=0)
        rendered_image_before = rendered_image
        if self.env_map is not None:
            bg_color_from_envmap = self.env_map(self.get_world_directions(viewpoint_camera, self.training).permute(1, 2, 0)).permute(2, 0, 1)
            rendered_image = rendered_image + (1 - rendered_opacity) * bg_color_from_envmap

        return {
            "render": rendered_image,
            "depth": rendered_depth,
        }

    def gsplat_forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        # TODO: `time_shift`
        means3D = pc.get_mean_SHM(viewpoint_camera.time - 0.5)
        marginal_t = pc.get_marginal_t(viewpoint_camera.time - 0.5)

        opacities = pc.get_opacities() * marginal_t

        # TODO: prefilter

        # rasterize
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = GSPlatRenderer.project(
            means3D=means3D,
            scales=pc.get_scales(),
            rotations=pc.get_rotations(),
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
        )

        if self.config.anti_aliased is True:
            opacities = opacities * comp.unsqueeze(-1)

        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        rgb, alpha = rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=rgbs,
            opacity=opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=bg_color,
            return_alpha=True,
        )
        alpha = alpha.unsqueeze(-1)

        depth = rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=depths.unsqueeze(-1),
            opacity=opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=torch.zeros((1,), dtype=torch.float, device=xys.device),
            return_alpha=True,
        )

        rgb_without_envmap = rgb
        if self.env_map is not None:
            bg_color_from_envmap = self.env_map(self.get_world_directions(viewpoint_camera, self.training).permute(1, 2, 0))
            rgb = rgb + (1 - alpha) * bg_color_from_envmap

        return {
            "render": rgb.permute(2, 0, 1),
            "rgb_without_envmap": rgb_without_envmap.permute(2, 0, 1),
            "depth": depth,
            "alpha": alpha.permute(2, 0, 1),
        }

    def forward(self, *args, **kwargs):
        return self.gsplat_forward(*args, **kwargs)

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return {
            "rgb": RendererOutputInfo("render"),
            "rgb_without_envmap": RendererOutputInfo("rgb_without_envmap"),
            "depth": RendererOutputInfo("depth", type=RendererOutputTypes.GRAY),
            "alpha": RendererOutputInfo("alpha", type=RendererOutputTypes.GRAY),
        }

    @staticmethod
    def get_world_directions(camera, train=False):
        height, width = camera.height.int().item(), camera.width.int().item()

        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=camera.world_to_camera.device)[0]
        u, v = grid.unbind(-1)
        if train:
            directions = torch.stack([(u - camera.cx + torch.rand_like(u)) / camera.fx,
                                      (v - camera.cy + torch.rand_like(v)) / camera.fy,
                                      torch.ones_like(u)], dim=0)
        else:
            directions = torch.stack([(u - camera.cx + 0.5) / camera.fx,
                                      (v - camera.cy + 0.5) / camera.fy,
                                      torch.ones_like(u)], dim=0)
        directions = F.normalize(directions, dim=0)
        c2w = torch.linalg.inv(camera.world_to_camera.T)
        directions = (c2w[:3, :3] @ directions.reshape(3, -1)).reshape(3, height, width)
        return directions  # [3, H, W]

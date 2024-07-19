from typing import List, Literal, Dict, Tuple
from dataclasses import dataclass
import math

import lightning
import torch

from gsplat.rendering import (
    rasterization,
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)

from . import Renderer, RendererOutputTypes, RendererOutputInfo
from .renderer import RendererConfig, BatchRenderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel

DEFAULT_TILE_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


class GSplatV1RendererImpl(BatchRenderer):
    def __init__(self, config: "GSplatV1Renderer") -> None:
        super().__init__()
        self._config = config

        self.rasterize_mode = "antialiased" if config.anti_aliased is True else "classic"

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        render_outputs = self.batch_forward(
            cameras=[viewpoint_camera],
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=render_types,
        )

        return self.batch_output_to_single(render_outputs)

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_outputs = self.batch_forward(
            cameras=[viewpoint_camera],
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=render_types,
        )

        render_outputs["viewspace_points"].retain_grad()

        return {
            "render": render_outputs["render"].squeeze(0),
            "acc": render_outputs["acc"].squeeze(0),
            "acc_depth": render_outputs["acc_depth"].squeeze(0),
            "exp_depth": render_outputs["exp_depth"].squeeze(0),
            "viewspace_points": render_outputs["viewspace_points"],  # do not squeeze here, or grad will be None
            "viewspace_points_grad_scale": render_outputs["viewspace_points_grad_scale"],
            "visibility_filter": render_outputs["visibility_filter"].squeeze(0),
            "radii": render_outputs["radii"].squeeze(0),
        }

        # project_results = self.project(
        #     means=pc.get_xyz,
        #     scales=pc.get_scaling,
        #     rotations=pc.get_rotation,
        #     camera=viewpoint_camera,
        #     scaling_modifier=scaling_modifier,
        # )
        # radii, means2d, depths, conics, compensations = project_results
        # visibility_mask = radii > 0
        #
        # # can not use `self.batch_rasterization()` directly, since `retain_grad()` is required
        # means2d.retain_grad()
        #
        # colors = spherical_harmonics(
        #     pc.active_sh_degree,
        #     dirs=pc.get_xyz - viewpoint_camera.camera_center,
        #     coeffs=pc.get_features,
        #     masks=visibility_mask,
        # )
        # colors = torch.clamp_min(colors + 0.5, 0.0)
        #
        # width = viewpoint_camera.width.int().item()
        # height = viewpoint_camera.height.int().item()
        #
        # render_colors, render_alphas = self.rasterize_to_pixels(
        #     project_results,
        #     width=width,
        #     height=height,
        #     opacities=pc.get_opacity.squeeze(-1),
        #     colors=colors,
        #     bg_color=bg_color,
        #     tile_size=self._config.tile_size,
        #     absgrad=self._config.absgrad,
        # )
        #
        # return {
        #     "render": render_colors.permute(2, 0, 1),
        #     "viewspace_points": means2d,
        #     "viewspace_points_grad_scale": 0.5 * torch.tensor([width, height], dtype=torch.float, device=bg_color.device),
        #     "visibility_filter": visibility_mask,
        #     "radii": radii,
        # }

    def batch_forward(
            self,
            cameras: List[Camera],
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        render_mode, rgb_dims = self.render_types_to_mode(render_types)
        render_colors, render_alphas, meta = self.batch_rasterization(
            cameras=cameras,
            pc=pc,
            bg_color=bg_color,
            packed=self._config.packed,
            tile_size=self._config.tile_size,
            render_mode=render_mode,
            rasterize_mode=self.rasterize_mode,
            scaling_modifier=scaling_modifier,
        )

        render_colors = render_colors.permute(0, 3, 1, 2)  # [N_cameras, N_channels, H, W]
        render_alphas = render_alphas.permute(0, 3, 1, 2)  # [N_cameras, N_channels, H, W]

        return {
            "render": render_colors[:, :rgb_dims],
            "acc": render_alphas,
            "acc_depth": render_colors[:, rgb_dims:rgb_dims + 1],
            "exp_depth": render_colors[:, rgb_dims:rgb_dims + 1],
            "viewspace_points": meta["means2d"],
            "viewspace_points_grad_scale": 0.5 * torch.tensor([render_colors.shape[3], render_colors.shape[2]], dtype=torch.float, device=bg_color.device),
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
        }

    def batch_training_forward(self, step: int, module: lightning.LightningModule, cameras: List[Camera], pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        outputs = self.batch_forward(
            cameras=cameras,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=render_types,
            **kwargs,
        )
        # TODO: let density controller invoke retain_grad()
        outputs["viewspace_points"].retain_grad()
        return outputs

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return {
            "rgb": RendererOutputInfo("render"),
            # "acc": RendererOutputInfo("acc", type=RendererOutputTypes.GRAY),
            "acc_depth": RendererOutputInfo("acc_depth", type=RendererOutputTypes.GRAY),
            "exp_depth": RendererOutputInfo("exp_depth", type=RendererOutputTypes.GRAY),
        }

    @staticmethod
    def batch_output_to_single(batch_output: Dict):
        return {
            "render": batch_output["render"].squeeze(0),
            "acc": batch_output["acc"].squeeze(0),
            "acc_depth": batch_output["acc_depth"].squeeze(0),
            "exp_depth": batch_output["exp_depth"].squeeze(0),
            "viewspace_points": batch_output["viewspace_points"].squeeze(0),
            "viewspace_points_grad_scale": batch_output["viewspace_points_grad_scale"],
            "visibility_filter": batch_output["visibility_filter"].squeeze(0),
            "radii": batch_output["radii"].squeeze(0),
        }

    @staticmethod
    def render_types_to_mode(render_types: List, rgb_dims: int = 3) -> Tuple[str, int]:
        """Convert `render_types` list to gsplat's `render_mode` str"""

        if render_types is None:
            render_types = ["rgb"]

        render_mode_list = []
        has_rgb = False
        # TODO: RGB should always at first place; D and ED can not be used at the same time
        for i in render_types:
            if i == "rgb":
                has_rgb = True
                render_mode_list.append("RGB")
            elif i == "acc_depth":
                render_mode_list.append("D")
            elif i == "exp_depth":
                render_mode_list.append("ED")

        return "+".join(render_mode_list), rgb_dims if has_rgb is True else 0

    @staticmethod
    def batch_camera_preprocess(cameras: List[Camera]):
        """
        Convert camera list to intrinsics and extrinsics matrix

        Args:
            cameras: camera list
        Return:
            viewmats: camera_to_world matrix [N_cameras, 4, 4]
            Ks, intrinsics in [N_cameras, 3, 3]
            width: int
            height: int
            batch_size: i.e. N_cameras
        """

        batch_size = len(cameras)

        device = cameras[0].world_to_camera.device

        # TODO: may be should use maximum size?
        width = cameras[0].width.int().item()
        height = cameras[0].height.int().item()

        viewmats = torch.empty((batch_size, 4, 4), dtype=torch.float, device=device)
        Ks = torch.empty((batch_size, 3, 3), dtype=torch.float, device=device)
        for i in range(batch_size):
            camera = cameras[i]
            viewmats[i] = camera.world_to_camera.T
            Ks[i][0, 0] = camera.fx
            Ks[i][1, 1] = camera.fy
            Ks[i][0, 2] = camera.cx
            Ks[i][1, 2] = camera.cy
            Ks[i][2, 2] = 1.

        return viewmats, Ks, width, height, batch_size

    @classmethod
    def batch_rasterization(
            cls,
            cameras: List[Camera],
            pc: GaussianModel,
            bg_color: torch.Tensor,
            colors: torch.Tensor = None,
            packed: bool = False,
            tile_size: int = DEFAULT_TILE_SIZE,
            absgrad: bool = False,
            render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
            rasterize_mode: Literal["classic", "antialiased"] = "antialiased",
            scaling_modifier=1.0,
    ):
        if colors is None:
            colors = pc.get_features
            sh_degree = pc.active_sh_degree
        else:
            sh_degree = None

        viewmats, Ks, width, height, batch_size = cls.batch_camera_preprocess(cameras)
        bg_color = bg_color.unsqueeze(0).repeat(batch_size, 1)

        scales = pc.get_scaling
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        return rasterization(
            means=pc.get_xyz,
            quats=pc.get_rotation,
            scales=scales,
            opacities=pc.get_opacity.squeeze(-1),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=packed,
            tile_size=tile_size,
            backgrounds=bg_color,
            render_mode=render_mode,
            absgrad=absgrad,
            rasterize_mode=rasterize_mode,
        )

    @classmethod
    def project(
            cls,
            means: torch.Tensor,  # xyz
            scales: torch.Tensor,
            rotations: torch.Tensor,  # remember to normalize them yourself
            camera: Camera,
            scaling_modifier=1.0,
            extra_projection_kwargs: dict = {},
    ):
        batch_project_results = cls.batch_project(
            means=means,
            scales=scales,
            rotations=rotations,
            batch_camera_data=cls.batch_camera_preprocess([camera]),
            scaling_modifier=scaling_modifier,
            extra_projection_kwargs=extra_projection_kwargs,
        )
        new_project_results = []
        for i in batch_project_results:
            new_project_results.append(i.squeeze(0))

        return new_project_results

    @classmethod
    def batch_project(
            cls,
            means: torch.Tensor,  # xyz
            scales: torch.Tensor,
            rotations: torch.Tensor,  # remember to normalize them yourself
            batch_camera_data: Tuple[torch.Tensor, torch.Tensor, int, int, int],
            scaling_modifier=1.0,
            extra_projection_kwargs: dict = {},
    ):
        """
        Return:
            A tuple: radii, means2d, depths, conics, compensations, isect_offsets
        """

        viewmats, Ks, width, height, batch_size = batch_camera_data
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        proj_results = fully_fused_projection(
            means,
            None,  # covars,
            rotations,
            scales,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=True,
            **extra_projection_kwargs,
        )

        return proj_results

    @classmethod
    def rasterize_to_pixels(
            cls,
            project_results,
            width: int,
            height: int,
            opacities: torch.Tensor,
            colors: torch.Tensor,
            bg_color: torch.Tensor,
            tile_size: int = DEFAULT_TILE_SIZE,
            absgrad: bool = False,
    ):
        new_project_results = []
        for i in project_results:
            new_project_results.append(i.unsqueeze(0))

        rasterize_results = cls.batch_rasterize_to_pixels(
            project_results=new_project_results,
            width=width,
            height=height,
            opacities=opacities,
            colors=colors.unsqueeze(0),
            bg_color=bg_color,
            tile_size=tile_size,
            absgrad=absgrad,
        )
        new_rasterize_results = []
        for i in rasterize_results:
            new_rasterize_results.append(i.squeeze(0))

        return new_rasterize_results

    @staticmethod
    def batch_rasterize_to_pixels(
            project_results,
            width: int,
            height: int,
            opacities: torch.Tensor,  # [N, ], TODO: may be camera dependent opacity should be supported?
            colors: torch.Tensor,  # [N_cameras, N_gaussians, N_channels]
            bg_color: torch.Tensor,  # [N_channels]
            tile_size: int = DEFAULT_TILE_SIZE,
            absgrad: bool = False,
            channel_chunk: int = 32,
    ):
        """
        Only non-packed input is supported
        """

        radii, means2d, depths, conics, compensations = project_results
        batch_size = radii.shape[0]
        opacities = opacities.repeat(batch_size, 1)  # [N_cameras, N_gaussians]

        # anti aliased
        if compensations is not None:
            opacities = opacities * compensations

        # Identify intersecting tiles
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
        )
        isect_offsets = isect_offset_encode(isect_ids, batch_size, tile_width, tile_height)

        if colors.shape[-1] > channel_chunk:
            # slice into chunks
            n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
            render_colors, render_alphas = [], []
            for i in range(n_chunks):
                colors_chunk = colors[..., i * channel_chunk: (i + 1) * channel_chunk]
                backgrounds_chunk = bg_color[..., i * channel_chunk: (i + 1) * channel_chunk]

                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=False,
                    absgrad=absgrad,
                )
                render_colors.append(render_colors_)
                render_alphas.append(render_alphas_)
            render_colors = torch.cat(render_colors, dim=-1)
            render_alphas = render_alphas[0]  # discard the rest
        else:
            render_colors, render_alphas = rasterize_to_pixels(
                means2d=means2d,
                conics=conics,
                colors=colors,
                opacities=opacities,
                image_width=width,
                image_height=height,
                tile_size=tile_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                backgrounds=bg_color.unsqueeze(0).repeat(batch_size, 1),
                packed=False,
                absgrad=absgrad,
            )

        return render_colors, render_alphas


@dataclass
class GSplatV1Renderer(RendererConfig):
    tile_size: int = DEFAULT_TILE_SIZE

    anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS

    packed: bool = False

    absgrad: bool = False

    def instantiate(self, *args, **kwargs) -> Renderer:
        assert self.packed is False
        return GSplatV1RendererImpl(self)

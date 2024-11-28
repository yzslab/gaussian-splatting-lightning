from dataclasses import dataclass
from typing import Union
import math
import torch
from .renderer import RendererConfig, Renderer, RendererOutputInfo, RendererOutputTypes

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

try:
    from gsplat.sh_decomposed import spherical_harmonics_decomposed
except:
    print("[ERROR] Incompatible gsplat found")
    print("Please install the latest version:")
    print("  pip uninstall gsplat")
    print("  pip install git+https://github.com/yzslab/gsplat.git@v1-with_v0_interfaces")
    exit()

from gsplat.v0_interfaces import rasterize_to_pixels


@dataclass
class GSplatV1Renderer(RendererConfig):
    block_size: int = 16

    anti_aliased: bool = True

    filter_2d_kernel_size: float = 0.3

    def instantiate(self, *args, **kwargs) -> "GSplatV1RendererModule":
        return GSplatV1RendererModule(self)


class GSplatV1RendererModule(Renderer):
    _RGB_REQUIRED = 1
    _ALPHA_REQUIRED = 1 << 1
    _ACC_DEPTH_REQUIRED = 1 << 2
    _ACC_DEPTH_INVERTED_REQUIRED = 1 << 3
    _EXP_DEPTH_REQUIRED = 1 << 4
    _EXP_DEPTH_INVERTED_REQUIRED = 1 << 5
    _INVERSE_DEPTH_REQUIRED = 1 << 6
    _HARD_DEPTH_REQUIRED = 1 << 7
    _HARD_INVERSE_DEPTH_REQUIRED = 1 << 8

    RENDER_TYPE_BITS = {
        "rgb": _RGB_REQUIRED,
        "alpha": _ALPHA_REQUIRED | _ACC_DEPTH_REQUIRED,
        "acc_depth": _ACC_DEPTH_REQUIRED,
        "acc_depth_inverted": _ACC_DEPTH_REQUIRED | _ACC_DEPTH_INVERTED_REQUIRED,
        "exp_depth": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED,
        "exp_depth_inverted": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _EXP_DEPTH_INVERTED_REQUIRED,
        "inverse_depth": _INVERSE_DEPTH_REQUIRED,
        "hard_depth": _HARD_DEPTH_REQUIRED,
        "hard_inverse_depth": _HARD_INVERSE_DEPTH_REQUIRED,
    }

    def __init__(self, config: GSplatV1Renderer):
        super().__init__()
        self.config = config

    def parse_render_types(self, render_types: list) -> int:
        if render_types is None:
            return self._RGB_REQUIRED
        else:
            bits = 0
            for i in render_types:
                bits |= self.RENDER_TYPE_BITS[i]
            return bits

    @staticmethod
    def is_type_required(bits: int, type: int) -> bool:
        return bits & type != 0

    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_type_bits = self.parse_render_types(render_types)

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        scales = pc.get_scales()
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        radii, means2d, depths, conics, compensations, isects = GSplatV1.project(
            means3d=pc.get_means(),
            scales=scales,
            quats=pc.get_rotations(),
            viewmat=viewpoint_camera.world_to_camera.T,
            fx=viewpoint_camera.fx,
            fy=viewpoint_camera.fy,
            cx=viewpoint_camera.cx,
            cy=viewpoint_camera.cy,
            img_height=img_height,
            img_width=img_width,
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
            tile_size=self.config.block_size,
        )

        opacities = pc.get_opacities()
        if self.config.anti_aliased is True:
            opacities = opacities * compensations[0, :, None]

        project_results_for_rasterization = radii, means2d, depths, conics, None, isects  # set the `compensations` to None, since the `opacities` have alredy been applied compensations

        depths = depths.squeeze(0)
        def rasterize(input_features: torch.Tensor, background, return_alpha: bool = False):
            rendered_colors, rendered_alphas = GSplatV1.rasterize(
                project_results_for_rasterization,
                opacities=opacities,
                colors=input_features,
                background=background,
                img_height=img_height,
                img_width=img_width,
                tile_size=self.config.block_size,
            )

            if return_alpha:
                return rendered_colors, rendered_alphas.squeeze(0).squeeze(-1)
            return rendered_colors

        # rgb
        rgb = None
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
            if pc.is_pre_activated:
                rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
            else:
                rgbs = spherical_harmonics_decomposed(pc.active_sh_degree, viewdirs, pc.get_shs_dc(), pc.get_shs_rest())
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

            rgb = rasterize(rgbs, bg_color).permute(2, 0, 1)

        alpha = None
        acc_depth_im = None
        acc_depth_inverted_im = None
        exp_depth_im = None
        exp_depth_inverted_im = None
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            acc_depth_im, alpha = rasterize(depths.unsqueeze(-1), torch.zeros((1,), device=bg_color.device), True)
            alpha = alpha[..., None]

            # acc depth inverted
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1. / acc_depth_im, acc_depth_im.detach().max())
                acc_depth_inverted_im = acc_depth_inverted_im.permute(2, 0, 1)

            # exp depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                exp_depth_im = torch.where(alpha > 0, acc_depth_im / alpha, acc_depth_im.detach().max())

                exp_depth_im = exp_depth_im.permute(2, 0, 1)

            # alpha
            if self.is_type_required(render_type_bits, self._ALPHA_REQUIRED):
                alpha = alpha.permute(2, 0, 1)
            else:
                alpha = None

            # permute acc depth
            acc_depth_im = acc_depth_im.permute(2, 0, 1)

            # exp depth inverted
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1. / exp_depth_im, exp_depth_im.detach().max())

        # inverse depth
        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths.clamp_min(0.) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)).permute(2, 0, 1)

        # hard depth
        hard_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            hard_depth_im, _ = GSplatV1.rasterize(
                project_results_for_rasterization,
                opacities=opacities + (1 - opacities.detach()),
                colors=depths.unsqueeze(-1),
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                img_height=img_height,
                img_width=img_width,
                tile_size=self.config.block_size,
            )
            hard_depth_im = hard_depth_im.permute(2, 0, 1)

        # hard inverse depth
        hard_inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths.clamp_min(0.) + 1e-8).unsqueeze(-1)

            hard_inverse_depth_im, _ = GSplatV1.rasterize(
                project_results_for_rasterization,
                opacities=opacities + (1 - opacities.detach()),
                colors=inverse_depth,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                img_height=img_height,
                img_width=img_width,
                tile_size=self.config.block_size,
            )

            hard_inverse_depth_im = hard_inverse_depth_im.permute(2, 0, 1)

        radii = radii.squeeze(0)

        return {
            "render": rgb,
            "alpha": alpha,
            "acc_depth": acc_depth_im,
            "acc_depth_inverted": acc_depth_inverted_im,
            "exp_depth": exp_depth_im,
            "exp_depth_inverted": exp_depth_inverted_im,
            "inverse_depth": inverse_depth_im,
            "hard_depth": hard_depth_im,
            "hard_inverse_depth": hard_inverse_depth_im,
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_width, img_height]]).to(means2d),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self):
        return {
            "rgb": RendererOutputInfo("render"),
            "alpha": RendererOutputInfo("alpha", type=RendererOutputTypes.GRAY),
            "acc_depth": RendererOutputInfo("acc_depth", type=RendererOutputTypes.GRAY),
            "acc_depth_inverted": RendererOutputInfo("acc_depth_inverted", type=RendererOutputTypes.GRAY),
            "exp_depth": RendererOutputInfo("exp_depth", type=RendererOutputTypes.GRAY),
            "exp_depth_inverted": RendererOutputInfo("exp_depth_inverted", type=RendererOutputTypes.GRAY),
            "inverse_depth": RendererOutputInfo("inverse_depth", type=RendererOutputTypes.GRAY),
            "hard_depth": RendererOutputInfo("hard_depth", type=RendererOutputTypes.GRAY),
            "hard_inverse_depth": RendererOutputInfo("hard_inverse_depth", type=RendererOutputTypes.GRAY),
        }


class GSplatV1:
    @classmethod
    def project(
        cls,
        means3d: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        viewmat: torch.Tensor,  # [4, 4]
        fx: Union[float, torch.Tensor],
        fy: Union[float, torch.Tensor],
        cx: Union[float, torch.Tensor],
        cy: Union[float, torch.Tensor],
        img_height: int,
        img_width: int,
        eps2d: float = 0.3,
        anti_aliased: bool = True,
        tile_size: int = 16,
        **kwargs,
    ):
        """
        Returns:
            A tuple:

            - **radii**. [1, N]
            - **means2d**. [1, N, 2]
            - **depths**. [1, N]
            - **conics**. [1, N, 3]
            - **compensations**. [1, N]
            - **A tuple**:
            -   **tiles_per_gauss**. [1, N]
            -   **isect_ids**. [n_isects]
            -   **flatten_ids**. [n_isects]
        """

        K = cls.get_intrinsics_matrix(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            device=means3d.device,
        )
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means3d,
            None,
            quats,
            scales,
            viewmats=viewmat[None, ...],
            Ks=K[None, ...],
            width=img_width,
            height=img_height,
            eps2d=eps2d,
            calc_compensations=anti_aliased,
            packed=False,
            **kwargs,
        )

        tile_width = math.ceil(img_width / float(tile_size))
        tile_height = math.ceil(img_height / float(tile_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_cameras=1,
            camera_ids=None,
            gaussian_ids=None,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)

        return radii, means2d.squeeze(0), depths, conics, compensations, (
            tiles_per_gauss,
            isect_ids,
            flatten_ids,
            isect_offsets,
        )

    @classmethod
    def rasterize(
        cls,
        project_results,  # then tuple returned by `cls.project()`
        opacities: torch.Tensor,  # [N, 1]
        colors: torch.Tensor,  # [N, n_color_dims]
        background: torch.Tensor,  # [n_color_dims]
        img_height: int,
        img_width: int,
        tile_size: int = 16,
        absgrad: bool = True,
        **kwargs,
    ):
        radii, means2d, depths, conics, compensations, (
            tiles_per_gauss,
            isect_ids,
            flatten_ids,
            isect_offsets,
        ) = project_results

        opacities = opacities.squeeze(-1).unsqueeze(0)
        colors = colors.unsqueeze(0)
        background = background.unsqueeze(0)

        if compensations is not None:
            opacities = opacities * compensations

        rendered_colors, rendered_alphas = rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=colors,
            opacities=opacities,
            image_width=img_width,
            image_height=img_height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=background,
            absgrad=absgrad,
            **kwargs,
        )

        return rendered_colors.squeeze(0), rendered_alphas.squeeze(0)

    @staticmethod
    def get_intrinsics_matrix(fx, fy, cx, cy, device):
        K = torch.eye(3, device=device)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

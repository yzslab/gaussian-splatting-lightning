from dataclasses import dataclass
from typing import Union, Tuple, Literal, Any
import math
import torch
from .renderer import RendererConfig, Renderer, RendererOutputInfo, RendererOutputTypes

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from gsplat.sh_decomposed import spherical_harmonics_decomposed
from gsplat.cuda.isect_tiles_tile_based_culling import (
    isect_tiles_tile_based_culling,
    isect_offset_encode_tile_based_culling,
)
from gsplat.v0_interfaces import rasterize_to_pixels


@dataclass
class GSplatV1Renderer(RendererConfig):
    block_size: int = 16

    anti_aliased: bool = True

    filter_2d_kernel_size: float = 0.3

    separate_sh: bool = False
    """Accelerated SH, from Taming 3DGS [Mallick and Goel, et al. 2024]"""

    tile_based_culling: bool = False
    """Tile-based culling, from StopThePop [Radl et al. 2024]"""

    max_viewspace_grad_scale: float = 65535.
    """ 1600 is recommended """

    def instantiate(self, *args, **kwargs) -> "GSplatV1RendererModule":
        return GSplatV1RendererModule(self)


@dataclass
class RuntimeOptions:
    radius_clip: float = 0.

    # radius_clip_from: float = 0.

    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


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
    _DEPTH_ALTERNATIVE = 1 << 9
    _NORMAL_REQUIRED = 1 << 10

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
        "inv_depth_alt": _DEPTH_ALTERNATIVE,
        "normal": _NORMAL_REQUIRED,
    }

    def __init__(self, config: GSplatV1Renderer):
        super().__init__()
        self.config = config
        self.runtime_options = RuntimeOptions()

        self.isect_encode = GSplatV1.isect_encode_with_unused_opacities
        if self.config.tile_based_culling:
            self.isect_encode = GSplatV1.isect_encode_tile_based_culling

        self._inv_depth_alt_state = 0
        self._inv_depth_alt = [
            self.RENDER_TYPE_BITS["inverse_depth"],
            self.RENDER_TYPE_BITS["hard_inverse_depth"],
        ]

    def parse_render_types(self, render_types: list) -> int:
        if render_types is None:
            return self._RGB_REQUIRED
        else:
            bits = 0
            for i in render_types:
                bits |= self.RENDER_TYPE_BITS[i]

            if self.is_type_required(bits, self._DEPTH_ALTERNATIVE):
                bits |= self._inv_depth_alt[self._inv_depth_alt_state]
                self._inv_depth_alt_state = int(not self._inv_depth_alt_state)

            return bits

    @staticmethod
    def is_type_required(bits: int, type: int) -> bool:
        return bits & type != 0

    def get_scales(self, camera, gaussian_model, **kwargs) -> Tuple[torch.Tensor, Any]:
        return gaussian_model.get_scales(), None

    def get_opacities(self, camera, gaussian_model, projections: Tuple, visibility_filter, status: Any, **kwargs) -> Tuple[torch.Tensor, Any]:
        return gaussian_model.get_opacities().squeeze(-1), status

    def get_rgbs(self, camera, gaussian_model, projections: Tuple, visibility_filter, status: Any, **kwargs) -> Tuple[torch.Tensor, Any]:
        viewdirs = gaussian_model.get_xyz.detach() - camera.camera_center  # (N, 3)
        if gaussian_model.is_pre_activated or not self.config.separate_sh:
            rgbs = spherical_harmonics(gaussian_model.active_sh_degree, viewdirs, gaussian_model.get_features, visibility_filter)
        else:
            rgbs = spherical_harmonics_decomposed(
                gaussian_model.active_sh_degree,
                viewdirs,
                gaussian_model.get_shs_dc(),
                gaussian_model.get_shs_rest(),
                visibility_filter,
            )
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        return rgbs

    def forward(self, viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_type_bits = self.parse_render_types(render_types)

        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)

        # 1. get scales and then project
        scales, status = self.get_scales(viewpoint_camera, pc, **kwargs)
        if scaling_modifier != 1.:
            scales = scales * scaling_modifier

        projections = GSplatV1.project(
            preprocessed_camera,
            pc.get_means(),
            scales,
            pc.get_rotations(),
            eps2d=self.config.filter_2d_kernel_size,
            anti_aliased=self.config.anti_aliased,
            radius_clip=self.runtime_options.radius_clip,
            # radius_clip_from=self.runtime_options.radius_clip_from,
            camera_model=self.runtime_options.camera_model,
        )
        radii, means2d, depths, conics, compensations = projections

        radii_squeezed = radii.squeeze(0)
        visibility_filter = radii_squeezed > 0

        # 2. get opacities and then isect encoding
        opacities, status = self.get_opacities(
            viewpoint_camera,
            pc,
            projections,
            visibility_filter,
            status,
            **kwargs,
        )

        opacities = opacities.unsqueeze(0)  # [1, N]
        if self.config.anti_aliased:
            opacities = opacities * compensations

        isects = self.isect_encode(
            preprocessed_camera,
            projections,
            opacities,
            tile_size=self.config.block_size,
        )

        # 3. rasterization
        means2d = means2d.squeeze(0)
        projection_for_rasterization = radii, means2d, depths, conics, compensations

        def rasterize(input_features: torch.Tensor, background, return_alpha: bool = False):
            rendered_colors, rendered_alphas = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities,
                colors=input_features,
                background=background,
                tile_size=self.config.block_size,
            )

            if return_alpha:
                return rendered_colors, rendered_alphas.squeeze(0).squeeze(-1)
            return rendered_colors

        outputs = {
            "render": None,
            "alpha": None,
            "acc_depth": None,
            "acc_depth_inverted": None,
            "exp_depth": None,
            "exp_depth_inverted": None,
            "inverse_depth": None,
            "hard_depth": None,
            "hard_inverse_depth": None,
            "normal": None,
            "inv_depth_alt": None,
            "viewspace_points": means2d,
            "viewspace_points_grad_scale": 0.5 * torch.tensor([preprocessed_camera[-1]]).to(means2d).clamp_(max=self.config.max_viewspace_grad_scale),
            "visibility_filter": visibility_filter,
            "acc_vis": None,
            "radii": radii_squeezed,
            "scales": scales,
            "opacities": opacities[0],
            "projections": projections,
            "isects": isects,
            "camera": viewpoint_camera,
            "preprocessed_camera": preprocessed_camera,
        }

        input_feature_list = []
        bg_color_list = []
        rasterization_output_indices = {}  # = (start, end + 1)
        n_input_dims = 0
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            rgbs = self.get_rgbs(
                viewpoint_camera,
                pc,
                projections,
                visibility_filter,
                status,
                **kwargs,
            )
            input_feature_list.append(rgbs)
            bg_color_list.append(bg_color)
            rasterization_output_indices["render"] = (n_input_dims, n_input_dims + 3)
            n_input_dims += 3

        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            input_feature_list.append(depths[0].unsqueeze(-1))
            bg_color_list.append(torch.zeros((1,), device=bg_color.device))
            rasterization_output_indices["acc_depth"] = (n_input_dims, n_input_dims + 1)
            n_input_dims += 1

        # normals
        if self.is_type_required(render_type_bits, self._NORMAL_REQUIRED):
            # TODO: implement in CUDA
            from internal.utils.general_utils import build_scaling_rotation
            scales_2d = torch.clone(scales)
            scales_2d[..., -1] = 1.
            normals = build_scaling_rotation(scales_2d, pc.get_rotations())[:, :3, -1]
            dirs = pc.get_means() - viewpoint_camera.camera_center
            is_point_to_the_view = torch.einsum("ij,ij->i", normals, dirs) > 0
            normal_multiplers = torch.where(is_point_to_the_view, -1., 1.)
            normals = normals * normal_multiplers.unsqueeze(-1)

            input_feature_list.append(normals)
            bg_color_list.append(torch.zeros((3,), device=bg_color.device))
            rasterization_output_indices["normal"] = (n_input_dims, n_input_dims + 3)
            n_input_dims += 3

        if n_input_dims > 0:
            if len(input_feature_list) == 1:
                input_features = input_feature_list[0]
                input_bg_colors = bg_color_list[0]
            else:
                input_features = torch.concat(input_feature_list, dim=-1)
                input_bg_colors = torch.concat(bg_color_list, dim=-1)
            render_features, render_alpha = rasterize(
                input_features,
                background=input_bg_colors,
                return_alpha=True,
            )
            render_features = render_features.permute(2, 0, 1)
            render_alpha = render_alpha.unsqueeze(0)

            for k, slice_index in rasterization_output_indices.items():
                outputs[k] = render_features[slice_index[0]:slice_index[1]]
            outputs["alpha"] = render_alpha
            # avoid overriding by hard depth
            outputs["acc_vis"] = means2d.has_hit_any_pixels

            # acc depth inverted
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_im = outputs["acc_depth"]
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1. / acc_depth_im, acc_depth_im.detach().max())
                outputs["acc_depth_inverted"] = acc_depth_inverted_im

            # exp depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                acc_depth_im = outputs["acc_depth"]
                exp_depth_im = torch.where(render_alpha > 0, acc_depth_im / render_alpha, acc_depth_im.detach().max())
                outputs["exp_depth"] = exp_depth_im

            # exp depth inverted
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1. / exp_depth_im, exp_depth_im.detach().max())
                outputs["exp_depth_inverted"] = exp_depth_inverted_im

        # inverse depth
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)).permute(2, 0, 1)
            outputs["inverse_depth"] = inverse_depth_im
            inv_depth_alt = inverse_depth_im
            outputs["inv_depth_alt"] = inv_depth_alt

        # hard depth
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            hard_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=depths[0].unsqueeze(-1),
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )
            hard_depth_im = hard_depth_im.permute(2, 0, 1)
            outputs["hard_depth"] = hard_depth_im

        # hard inverse depth
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths[0].clamp_min(0.) + 1e-8).unsqueeze(-1)
            hard_inverse_depth_im, _ = GSplatV1.rasterize(
                preprocessed_camera,
                projection_for_rasterization,
                isects,
                opacities=opacities + (1 - opacities.detach()),
                colors=inverse_depth,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                tile_size=self.config.block_size,
            )

            hard_inverse_depth_im = hard_inverse_depth_im.permute(2, 0, 1)
            outputs["hard_inverse_depth"] = hard_inverse_depth_im
            inv_depth_alt = hard_inverse_depth_im
            outputs["inv_depth_alt"] = inv_depth_alt

        return outputs

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        with tabs.add_tab("gsplat"):
            self._viewer_options = GSplatV1ViewerOptions(viewer, server, self.runtime_options)

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
            "normal": RendererOutputInfo("normal", type=RendererOutputTypes.NORMAL_MAP),
            # "inv_depth_alt": RendererOutputInfo("inv_depth_alt", type=RendererOutputTypes.GRAY),
        }


class GSplatV1:
    @classmethod
    def preprocess_camera(cls, viewpoint_camera):
        viewmats = viewpoint_camera.world_to_camera.T.unsqueeze(0)

        Ks = torch.tensor([[
            [viewpoint_camera.fx, 0., viewpoint_camera.cx],
            [0., viewpoint_camera.fy, viewpoint_camera.cy],
            [0., 0., 1.],
        ]], dtype=torch.float, device=viewpoint_camera.R.device)

        img_width = int(viewpoint_camera.width.item())
        img_height = int(viewpoint_camera.height.item())

        return viewmats, Ks, (img_width, img_height)

    @classmethod
    def project(
        cls,
        preprocessed_camera: Tuple,
        means3d: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        eps2d: float = 0.3,
        anti_aliased: bool = True,
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
        """

        return fully_fused_projection(
            means3d,
            None,
            quats,
            scales,
            viewmats=preprocessed_camera[0],
            Ks=preprocessed_camera[1],
            width=preprocessed_camera[2][0],
            height=preprocessed_camera[2][1],
            eps2d=eps2d,
            calc_compensations=anti_aliased,
            packed=False,
            **kwargs,
        )

    @classmethod
    def isect_encode(
        cls,
        preprocessed_camera: Tuple,
        projection_results,
        tile_size: int = 16,
    ):
        """
        Returns:
            A tuple:

            -   **tiles_per_gauss**. [1, N]
            -   **isect_ids**. [n_isects]
            -   **flatten_ids**. [n_isects]
            -   **isect_offsets**. [1, tile_height, tile_width]
        """

        img_width, img_height = preprocessed_camera[-1]

        radii, means2d, depths, _, _ = projection_results

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

        return tiles_per_gauss, isect_ids, flatten_ids, isect_offsets

    @classmethod
    def isect_encode_with_unused_opacities(
        cls,
        preprocessed_camera: Tuple,
        projection_results,
        opacities: torch.Tensor,  # [1, N]
        tile_size: int = 16,
    ):
        return cls.isect_encode(
            preprocessed_camera,
            projection_results,
            tile_size,
        )

    @classmethod
    def isect_encode_tile_based_culling(
        cls,
        preprocessed_camera: Tuple,
        projection_results,
        opacities: torch.Tensor,  # [1, N]
        tile_size: int = 16,
    ):
        """
        Returns:
            A tuple:

            -   **tiles_per_gauss**. [1, N]
            -   **isect_ids**. [n_isects]
            -   **flatten_ids**. [n_isects]
            -   **isect_offsets**. [1, tile_height, tile_width]
        """

        img_width, img_height = preprocessed_camera[-1]

        radii, means2d, depths, conics, _ = projection_results

        tile_width = math.ceil(img_width / float(tile_size))
        tile_height = math.ceil(img_height / float(tile_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_tile_based_culling(
            means2d,
            radii,
            depths,
            conics,
            opacities.detach(),
            tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_cameras=1,
            camera_ids=None,
            gaussian_ids=None,
        )
        isect_offsets, flatten_ids = isect_offset_encode_tile_based_culling(
            isect_ids,
            flatten_ids,
            1,
            tile_width,
            tile_height,
        )

        return tiles_per_gauss, isect_ids, flatten_ids, isect_offsets

    @classmethod
    def preprocess(
        cls,
        preprocessed_camera: Tuple,
        means3d: torch.Tensor,  # [N, 3]
        scales: torch.Tensor,  # [N, 3]
        quats: torch.Tensor,  # [N, 4]
        eps2d: float = 0.3,
        anti_aliased: bool = True,
        tile_size: int = 16,
        tile_based_culling: bool = False,
        opacities: torch.Tensor = None,  # [N, 1]
    ):
        projections = cls.project(
            preprocessed_camera,
            means3d=means3d,
            scales=scales,
            quats=quats,
            eps2d=eps2d,
            anti_aliased=anti_aliased,
        )

        opacities = opacities.unsqueeze(0).squeeze(-1)  # [1, N]
        if anti_aliased:
            opacities = opacities * projections[-1]

        if tile_based_culling:
            isects = cls.isect_encode_tile_based_culling(
                preprocessed_camera,
                projections,
                opacities,
                tile_size=tile_size,
            )
        else:
            isects = cls.isect_encode(
                preprocessed_camera,
                projections,
                tile_size=tile_size,
            )

        radii, means2d, depths, conics, compensations = projections

        return (radii, means2d.squeeze(0), depths, conics, compensations), isects, opacities

    @classmethod
    def rasterize(
        cls,
        preprocessed_camera: Tuple,
        projections,  # NOTE: the means2D must be [N, 2]
        isects,
        opacities: torch.Tensor,  # [1, N]
        colors: torch.Tensor,  # [N, n_color_dims]
        background: torch.Tensor,  # [n_color_dims]
        tile_size: int = 16,
        absgrad: bool = True,
        **kwargs,
    ):
        img_width, img_height = preprocessed_camera[-1]
        _, means2d, _, conics, _ = projections
        _, _, flatten_ids, isect_offsets = isects

        colors = colors.unsqueeze(0)
        background = background.unsqueeze(0)

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


from viser import ViserServer


class GSplatV1ViewerOptions:
    def __init__(self, viewer, server: ViserServer, options: RuntimeOptions):
        self.viewer = viewer
        self.server = server
        self.options = options

        # radius clip
        self.radius_clip_number = server.gui.add_number(
            label="Radius Clip",
            initial_value=options.radius_clip,
            step=0.1,
            min=0.,
            max=65535.,
        )

        @self.radius_clip_number.on_update
        def _(_):
            options.radius_clip = self.radius_clip_number.value
            viewer.rerender_for_all_client()

        # # radius clip from
        # self.radius_clip_from_number = server.gui.add_number(
        #     label="Radius Clip From",
        #     initial_value=options.radius_clip_from,
        #     step=0.01,
        #     min=0.,
        #     max=65535.,
        # )

        # @self.radius_clip_from_number.on_update
        # def _(_):
        #     options.radius_clip_from = self.radius_clip_from_number.value
        #     viewer.rerender_for_all_client()

        # camera model
        self.camera_model_dropdown = server.gui.add_dropdown(
            label="Camera Model",
            options=["pinhole", "ortho", "fisheye"],
            initial_value=options.camera_model,
        )

        @self.camera_model_dropdown.on_update
        def _(_):
            options.camera_model = self.camera_model_dropdown.value
            viewer.rerender_for_all_client()

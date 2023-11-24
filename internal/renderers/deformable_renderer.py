from dataclasses import dataclass
from typing import Tuple, Optional, Any

import math
import lightning
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh
from .renderer import Renderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel
from ..models.deform_model import DeformModel
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.rigid_utils import from_homogenous, to_homogenous


@dataclass
class DeformNetworkConfig:
    """
    Args:
        tcnn: whether use tiny-cuda-nn as network implementation
    """

    tcnn: bool = True
    n_layers: int = 8
    n_neurons: int = 256
    is_6dof: bool = False
    chunk: int = -1  # avoid CUDA oom


@dataclass
class XYZEncodingConfig:
    n_frequencies: int = 10


@dataclass
class TimeEncodingConfig:
    n_frequencies: int = 6
    n_layers: int = 0
    n_neurons: int = 0
    n_output_dim: int = 30


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 0.0008
    max_steps: int = 40_000
    lr_delay_mult: float = 0.01
    eps: float = 1e-15
    warm_up: int = 3_000
    enable_ast: bool = True


class DeformableRenderer(Renderer):
    def __init__(
            self,
            deform_network: DeformNetworkConfig,
            xyz_encoding: XYZEncodingConfig,
            time_encoding: TimeEncodingConfig,
            optimization: DeformableRendererOptimizationConfig,
    ) -> None:
        super().__init__()

        self.deform_network_config = deform_network
        self.xyz_encoding_config = xyz_encoding
        self.time_encoding_config = time_encoding
        self.optimization_config = optimization

        self.compute_cov3D_python = False
        self.convert_SHs_python = False

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        # without ast noise
        N = pc.get_xyz.shape[0]
        time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input)

        return self._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        if step >= self.optimization_config.warm_up:
            N = pc.get_xyz.shape[0]
            time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
            ast_noise = 0
            if self.optimization_config.enable_ast is True:
                time_interval = 1 / ((step % self.train_set_length) + 1)
                ast_noise = torch.randn(1, 1, device=pc.get_xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
            d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input + ast_noise)
            torch.cuda.empty_cache()  # avoid CUDA OOM

        return self._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )

    def _render(
            self,
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        override_color = None

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

        if self.deform_network_config.is_6dof is True:
            if torch.is_tensor(d_xyz) is False:
                means3D = pc.get_xyz
            else:
                means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = pc.get_xyz + d_xyz
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
            scales = pc.get_scaling + d_scaling
            rotations = pc.get_rotation + d_rotation

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
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)

        network_factory = NetworkFactory(tcnn=self.deform_network_config.tcnn)

        self.deform_model = DeformModel(
            network_factory=network_factory,
            D=self.deform_network_config.n_layers,
            W=self.deform_network_config.n_neurons,
            input_ch=3,
            multires=self.xyz_encoding_config.n_frequencies,
            t_D=self.time_encoding_config.n_layers,
            t_W=self.time_encoding_config.n_neurons,
            t_multires=self.time_encoding_config.n_frequencies,
            t_output_ch=self.time_encoding_config.n_output_dim,
            is_6dof=self.deform_network_config.is_6dof,
            chunk=self.deform_network_config.chunk,
        )
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    def training_setup(self) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            [{
                "params": list(self.deform_model.parameters()),
                "name": "deform",
            }],
            lr=self.optimization_config.lr,
            eps=self.optimization_config.eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.optimization_config.lr_delay_mult ** min(iter / self.optimization_config.max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler

import torch
from gsplat.sh import spherical_harmonics
from gsplat.rasterize import rasterize_gaussians
from gsplat._torch_impl import project_gaussians_forward
from internal.utils.gaussian_projection import project_gaussians
from internal.utils.sh_utils import eval_gaussian_model_sh
from .renderer import *


class PythonPreprocessGSplatRenderer(Renderer):
    block_size: int

    anti_aliased: bool

    def __init__(self, block_size: int = 16, anti_aliased: bool = True) -> None:
        super().__init__()
        self.block_size = block_size
        self.anti_aliased = anti_aliased

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        # xys, depths, radii, conics, comp, num_tiles_hit, cov3d, mask = project_gaussians(
        #     means_3d=pc.get_xyz,
        #     scales=pc.get_scaling,
        #     scale_modifier=scaling_modifier,
        #     quaternions=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
        #     world_to_camera=viewpoint_camera.world_to_camera,
        #     full_ndc_projection=viewpoint_camera.full_projection,
        #     fx=viewpoint_camera.fx,
        #     fy=viewpoint_camera.fy,
        #     cx=viewpoint_camera.cx,
        #     cy=viewpoint_camera.cy,
        #     img_height=viewpoint_camera.height,
        #     img_width=viewpoint_camera.width,
        #     block_width=self.block_size,
        # )
        cov_3d, cov_2d, xys, depths, radii, conics, comp, num_tiles_hit, mask = project_gaussians_forward(
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=scaling_modifier,
            quats=pc.get_rotation,
            viewmat=viewpoint_camera.world_to_camera.T,
            fullmat=viewpoint_camera.full_projection.T,
            intrins=(viewpoint_camera.fx, viewpoint_camera.fy, viewpoint_camera.cx, viewpoint_camera.cy),
            img_size=(viewpoint_camera.width, viewpoint_camera.height),
            block_width=self.block_size,
        )

        try:
            xys.retain_grad()
        except:
            pass

        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        # rgbs = eval_gaussian_model_sh(viewpoint_camera, pc)

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None].detach()

        rgb = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore

        return {
            "render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": mask,
            "radii": radii,
        }

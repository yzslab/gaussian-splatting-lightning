import os.path
from argparse import Namespace
from dataclasses import dataclass
from typing import Tuple, Optional, Any

import math
import lightning
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh
from .renderer import Renderer
from .deformable_renderer import DeformableRenderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel
from ..models.vanilla_deform_model import VanillaDeformNetwork
from ..utils.rigid_utils import from_homogenous, to_homogenous


class VanillaDeformableRenderer(Renderer):
    _render = DeformableRenderer._render

    def __init__(
            self,
            model_path: str,
            load_iteration: int,
            device,
    ) -> None:
        super().__init__()

        cfg_args = self._parse_cfg_args(model_path)

        self.deform_model = VanillaDeformNetwork(
            is_blender=cfg_args.is_blender,
            is_6dof=cfg_args.is_6dof,
        )

        checkpoint = torch.load(os.path.join(model_path, "deform", "iteration_{}".format(load_iteration), "deform.pth"))
        self.deform_model.load_state_dict(checkpoint)

        self.deform_model = self.deform_model.to(device)

        self.deform_network_config = Namespace(is_6dof=cfg_args.is_6dof)

        self.compute_cov3D_python = False
        self.convert_SHs_python = False

    @classmethod
    def _parse_cfg_args(cls, model_path):
        with open(os.path.join(model_path, "cfg_args"), "r") as f:
            cfg_args = f.read()
        return eval(cfg_args)

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

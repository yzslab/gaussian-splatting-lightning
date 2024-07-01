from typing import Literal, Any, Tuple, Optional, Union, List, Dict

import lightning
import torch
from .renderer import Renderer
from .gsplat_renderer import GSPlatRenderer
from gsplat.sh import spherical_harmonics
import sklearn
import sklearn.decomposition
import numpy as np
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel


class NoFeatureDecoder(torch.nn.Module):
    def forward(self, i):
        return i


class CNNDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1).cuda()

    def forward(self, x):
        x = self.conv(x)
        return x


class Feature3DGSRenderer(Renderer):
    def __init__(
            self,
            speedup: bool,
            n_feature_dims: int,
            feature_lr: float = 0.001,
            feature_decoder_lr: float = 0.0001,
            rasterize_batch: int = 32,
    ):
        super().__init__()
        self.speedup = speedup
        self.n_feature_dims = n_feature_dims
        self.feature_lr = feature_lr
        self.feature_decoder_lr = feature_decoder_lr
        self.rasterize_batch = rasterize_batch

        # update this when feature updated
        self.pca_projected_color = None

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        n_actual_feature_dims = self.n_feature_dims

        # setup feature decoder
        self.feature_decoder = NoFeatureDecoder()
        if self.speedup is True:
            n_actual_feature_dims = n_actual_feature_dims // 2
            self.feature_decoder = CNNDecoder(n_actual_feature_dims, self.n_feature_dims)

        self.n_actual_feature_dims = n_actual_feature_dims

        # initialize features
        feature_tensor = torch.zeros(
            (kwargs["lightning_module"].gaussian_model._xyz.shape[0], n_actual_feature_dims),
            dtype=torch.float,
            device=kwargs["lightning_module"].device,
            requires_grad=True,
        )
        self.features = torch.nn.Parameter(feature_tensor)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        optimizer = torch.optim.Adam(
            params=[
                {"name": "features", "params": [self.features], "lr": self.feature_lr},
                {"name": "feature_decoder", "params": self.feature_decoder.parameters(), "lr": self.feature_decoder_lr},
            ]
        )

        return optimizer, None

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        if render_types is None:
            render_types = ["features"]

        with torch.no_grad():
            project_results = GSPlatRenderer.project(
                means3D=pc.get_xyz,
                scales=pc.get_scaling,
                rotations=pc.get_rotation,
                viewpoint_camera=viewpoint_camera,
                scaling_modifier=scaling_modifier,
            )
            # anti-aliased
            comp = project_results[4]
            opacities = pc.get_opacity * comp[:, None]

        outputs = {}

        if "rgb" in render_types:
            with torch.no_grad():
                viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
                rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
                rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
                outputs["render"] = GSPlatRenderer.rasterize_simplified(
                    project_results,
                    viewpoint_camera=viewpoint_camera,
                    colors=rgbs,
                    bg_color=bg_color,
                    opacities=opacities,
                    anti_aliased=False,
                )
        if "features" in render_types or "features_vanilla_pca_2d" in render_types:
            rendered_features_list = []
            feature_bg_color = torch.zeros((self.rasterize_batch,), dtype=torch.float, device=self.features.device)
            for i in range(self.n_actual_feature_dims // self.rasterize_batch):
                start = i * self.rasterize_batch
                rendered_features_list.append(GSPlatRenderer.rasterize_simplified(
                    project_results,
                    viewpoint_camera=viewpoint_camera,
                    colors=self.features[..., start:start + self.rasterize_batch],
                    bg_color=feature_bg_color,
                    opacities=opacities,
                    anti_aliased=False,
                ))
            outputs["features"] = self.feature_decoder(torch.concat(rendered_features_list, dim=0))
        if "features_vanilla_pca_2d" in render_types:
            outputs["features_vanilla_pca_2d"] = self.feature_visualize(outputs["features"])
        if "features_pca_3d" in render_types:
            if getattr(self, "pca_projected_color", None) is None:
                from internal.utils.seganygs import SegAnyGSUtils
                normalized_features = torch.nn.functional.normalize(self.features, dim=-1)
                self.pca_projected_color = SegAnyGSUtils.get_pca_projected_colors(
                    semantic_features=normalized_features,
                    pca_projection_matrix=SegAnyGSUtils.get_pca_projection_matrix(semantic_features=normalized_features),
                )
            features_pca_3d = GSPlatRenderer.rasterize_simplified(
                project_results,
                viewpoint_camera=viewpoint_camera,
                colors=self.pca_projected_color,
                bg_color=bg_color,
                opacities=opacities,
                anti_aliased=False,
            )
            view_shape = (3, -1)
            features_pca_3d = features_pca_3d - torch.min(features_pca_3d.view(view_shape), dim=1, keepdim=True).values.unsqueeze(-1)
            features_pca_3d = features_pca_3d / (torch.max(features_pca_3d.view(view_shape), dim=1, keepdim=True).values.unsqueeze(-1) + 1e-9)
            outputs["features_pca_3d"] = features_pca_3d

        return outputs

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            render_types=["features"],
            **kwargs,
        )

    def get_available_output_types(self) -> Dict:
        return {
            "rgb": "render",
            "features": "features",
            "features_vanilla_pca_2d": "features_vanilla_pca_2d",
            "features_pca_3d": "features_pca_3d",
        }

    def is_type_feature_map(self, t: str) -> bool:
        return t == "features"

    # @staticmethod
    # def feature_visualize(image: torch.Tensor):
    #     from internal.utils.seganygs import SegAnyGSUtils
    #
    #     feature_map = image.permute(1, 2, 0).view((-1, image.shape[0]))
    #     normalized_feature_map = torch.nn.functional.normalize(feature_map, dim=-1)
    #     pca_projection_matrix = SegAnyGSUtils.get_pca_projection_matrix(normalized_feature_map)
    #     pca_projected_colors = SegAnyGSUtils.get_pca_projected_colors(normalized_feature_map, pca_projection_matrix)
    #     pca_feature_map = pca_projected_colors.view(*image.shape[1:], pca_projected_colors.shape[-1]).permute(2, 0, 1)
    #     return pca_feature_map

    @staticmethod
    def feature_visualize(feature):
        fmap = feature[None, :, :, :]  # torch.Size([1, C, H, W])
        fmap = torch.nn.functional.normalize(fmap, dim=1)
        pca = sklearn.decomposition.PCA(3, random_state=42)
        f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
        transformed = pca.fit_transform(f_samples)
        feature_pca_mean = torch.tensor(f_samples.mean(0), dtype=torch.float, device=feature.device)
        feature_pca_components = torch.tensor(pca.components_, dtype=torch.float, device=feature.device)
        q1, q99 = np.percentile(transformed, [1, 99])
        feature_pca_postprocess_sub = q1
        feature_pca_postprocess_div = (q99 - q1)
        del f_samples
        vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
        vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
        vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).permute(2, 0, 1)
        return vis_feature

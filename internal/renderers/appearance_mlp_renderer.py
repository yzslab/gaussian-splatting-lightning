from typing import Any, Tuple, Optional
import torch
from internal.configs.appearance import AppearanceModelParams
from internal.cameras.cameras import Camera
from internal.models.gaussian_model import GaussianModel
from internal.models.appearance_model import AppearanceModel
from .vanilla_renderer import VanillaRenderer


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

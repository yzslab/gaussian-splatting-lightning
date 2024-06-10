from collections import namedtuple
from typing import Any, Optional, Union
from typing_extensions import Self

import lightning
import torch.nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from internal.configs.semantic_splatting import Optimization as OptimizationConfig
from internal.renderers.gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
from internal.renderers.contrastive_feature_renderer import ContrastiveFeatureRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader


class SemanticSplatting(lightning.LightningModule):
    def __init__(
            self,
            initialize_from: str,
            optimization: OptimizationConfig,
            n_feature_dims: int = 32,
            sh_degree: int = -1,
            save_iterations: list = None,
            output_path: str = None,
            web_viewer: bool = False,
            save_val_output: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.n_feature_dims = n_feature_dims

        # avoid storing state_dict
        self.models = namedtuple("Models", ["gaussian"])

        # self.renderer = ContrastiveFeatureRenderer()
        self.renderer = GSplatContrastiveFeatureRenderer()

    def setup(self, stage: str) -> None:
        super().setup(stage)

        initialize_from = self.hparams["initialize_from"]
        sh_degree = self.hparams["sh_degree"]

        gaussian_model, _ = GaussianModelLoader.search_and_load(initialize_from, sh_degree, self.device)
        self.models.gaussian = gaussian_model

        self.gaussian_semantic_features = torch.nn.Parameter(torch.zeros(
            (gaussian_model.get_xyz.shape[0], self.n_feature_dims),
            dtype=torch.float,
        ), requires_grad=True)
        # torch.nn.init.normal_(self.gaussian_semantic_features)

        self.sam_proj = torch.nn.Sequential(
            torch.nn.Linear(256, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, self.n_feature_dims, bias=True)
        )

    def forward(self, camera) -> Any:
        return self.renderer(
            viewpoint_camera=camera,
            pc=self.models.gaussian,
            semantic_features=self.gaussian_semantic_features,
        )

    def forward_with_loss_calculation(self, cameras, image_info, semantic):
        outputs = self(cameras)
        rendered_features = outputs["render"]

        sam_features = semantic[1]

        H, W = sam_features.shape[-2:]

        # N_mask, H, W
        sam_masks = torch.nn.functional.interpolate(semantic[0].unsqueeze(0), size=sam_features.shape[-2:], mode='nearest').squeeze()
        nonzero_masks = sam_masks.sum(dim=(1, 2)) > 0
        sam_masks = sam_masks[nonzero_masks, :, :]
        full_resolution_sam_masks = semantic[0]
        full_resolution_sam_masks = full_resolution_sam_masks[nonzero_masks, :, :]

        low_dim_sam_features = self.sam_proj(
            sam_features.reshape(-1, H * W).permute([1, 0])
        ).permute([1, 0]).reshape(self.n_feature_dims, H, W)

        # NHW, NCHW
        prototypes = (sam_masks.unsqueeze(1) * low_dim_sam_features).sum(dim=(2, 3))
        prototypes /= sam_masks.sum(dim=(1, 2)).unsqueeze(-1)

        pp = torch.einsum('NC, CHW -> NHW', prototypes, rendered_features)

        prob = torch.sigmoid(pp)

        full_resolution_sam_masks = torch.nn.functional.interpolate(full_resolution_sam_masks.unsqueeze(0), size=prob.shape[-2:], mode='bilinear').squeeze()
        full_resolution_sam_masks[full_resolution_sam_masks <= 0.5] = 0

        bce_contrastive_loss = full_resolution_sam_masks * torch.log(prob + 1e-8) + ((1 - full_resolution_sam_masks) * torch.log(1 - prob + 1e-8))
        bce_contrastive_loss = -bce_contrastive_loss.mean()

        rands = torch.rand(self.gaussian_semantic_features.shape[0], device=prob.device)
        reg_loss = torch.relu(torch.einsum('NC,KC->NK', self.gaussian_semantic_features[rands > 0.9, :], prototypes)).mean()
        loss = bce_contrastive_loss + 0.1 * reg_loss

        NHW = sam_masks
        N, H, W = NHW.shape
        NL = NHW.view(N, -1)
        intersection = torch.einsum('NL,NC->LC', NL, NL)
        union = NL.sum(dim=0, keepdim=True) + NL.sum(dim=0, keepdim=True).T - intersection
        similarity = intersection / (union + 1e-5)
        HWHW = similarity.view(H, W, H, W)
        HWHW[HWHW == 0] = -1
        norm_rendered_feature = torch.nn.functional.normalize(torch.nn.functional.interpolate(rendered_features.unsqueeze(0), (H, W), mode='bilinear').squeeze(), dim=0, p=2)
        correspondence = torch.relu(torch.einsum('CHW,CJK->HWJK', norm_rendered_feature, norm_rendered_feature))
        corr_loss = -HWHW * correspondence
        corr_loss_mean = corr_loss.mean()

        loss = loss + corr_loss_mean

        return outputs, loss, bce_contrastive_loss, corr_loss_mean

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        camera, image_info, semantic = batch
        _, loss, loss_3d, loss_corr = self.forward_with_loss_calculation(camera, image_info, semantic)
        self.log("val/loss", loss, prog_bar=False, on_epoch=True, batch_size=1)
        self.log("val/3d_loss", loss_3d, prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/corr_loss", loss_corr, prog_bar=True, on_epoch=True, batch_size=1)
        return

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        camera, image_info, semantic = batch
        _, loss, loss_3d, loss_corr = self.forward_with_loss_calculation(camera, image_info, semantic)

        self.log("train/loss", loss, prog_bar=False, on_step=True, batch_size=1)
        self.log("train/3d_loss", loss_3d, prog_bar=True, on_step=True, batch_size=1)
        self.log("train/corr_loss", loss_corr, prog_bar=True, on_step=True, batch_size=1)

        if self.trainer.global_step % 100 == 0:
            metrics_to_log = {}
            for opt_idx, opt in enumerate([self.optimizer]):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics_to_log["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]
            self.logger.log_metrics(
                metrics_to_log,
                step=self.trainer.global_step,
            )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.optimizer = torch.optim.Adam(
            params=[
                {
                    "params": [self.gaussian_semantic_features],
                    "name": "gaussian_semantic_features",
                    "lr": self.hparams["optimization"].lr,
                },
                {
                    "params": self.sam_proj.parameters(),
                    "name": "sam_proj",
                    "lr": self.hparams["optimization"].lr,
                }
            ],
            lr=0.,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer=self.optimizer,
                    lr_lambda=lambda iter: 0.1 ** min(iter / self.trainer.max_steps, 1),
                ),
                "interval": "step",
                "frequency": 1,
            },
        }

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self.models.gaussian = self.models.gaussian.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        return_value = super().cuda(device)
        self.models.gaussian = self.models.gaussian.to(device=return_value.device)
        return return_value



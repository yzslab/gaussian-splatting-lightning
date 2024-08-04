from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl, ssim
from internal.density_controllers.density_controller import Utils as DensityControllerUtils


class SpotLessModule(torch.nn.Module):
    """SpotLess mask MLP predictor class."""

    def __init__(self, num_classes: int, num_features: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.mlp = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid(),
        )

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def get_regularizer(self):
        return torch.max(abs(self.mlp[0].weight.data)) * torch.max(
            abs(self.mlp[2].weight.data)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@dataclass
class SpotLessMetrics(VanillaMetrics):
    lower_bound: float = 0.5
    upper_bound: float = 0.9
    """ Thresholds for mlp mask supervision """

    bin_size: int = 10000
    """ bin size for the error hist for robust threshold  """

    cluster: bool = False
    """ enable clustering of semantic features """

    schedule: bool = True
    """ schedule sampling of the mask based on alpha """

    schedule_beta: float = -3e-3
    """ alpha sampling schedule rate (higher more robust) """

    reset_sh: int = 8002
    """ Reset SH specular coefficients once """

    robust_percentile: float = 0.7
    """ Robust loss percentile for threshold """

    densify_until_iter: int = 15_000

    n_feature_dims: int = 1280

    def instantiate(self, *args, **kwargs) -> "SpotLessMetricsModule":
        assert self.rgb_diff_loss == "l1"
        assert self.cluster is False

        return SpotLessMetricsModule(self)


class SpotLessMetricsModule(VanillaMetricsImpl):
    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)
        self.register_buffer("_hist_err", torch.zeros((self.config.bin_size,)), persistent=True)
        self.register_buffer("_avg_err", torch.tensor(1., dtype=torch.float), persistent=True)
        self.register_buffer("_lower_err", torch.tensor(0., dtype=torch.float), persistent=True)
        self.register_buffer("_upper_err", torch.tensor(1., dtype=torch.float), persistent=True)

        # currently using positional encoding of order 20 (4*20 = 80)
        self.spotless_module = SpotLessModule(
            num_classes=1, num_features=self.config.n_feature_dims + 80
        )

    def training_setup(self, pl_module):
        # register spotless module training hook
        pl_module.on_after_backward_hooks.append(self.spotless_module_loss_backward)
        # register the hook to reset shs_rest
        pl_module.on_after_backward_hooks.append(self.reset_shs_rest)
        # register the hook to update states
        pl_module.on_after_backward_hooks.append(self.update_running_stats)

        self.spotless_loss = lambda p, minimum, maximum: torch.mean(
            torch.nn.ReLU()(p - minimum) + torch.nn.ReLU()(maximum - p)
        )

        # setup spotless module optimizer
        spotless_optimizers = [
            torch.optim.Adam(
                self.spotless_module.parameters(),
                lr=1e-3,
            )
        ]

        return spotless_optimizers, None

    def spotless_module_loss_backward(self, outputs, batch, gaussian_model, step, pl_module):
        pred_mask_up = outputs["pred_mask_up"]
        upper_mask = outputs["upper_mask"]
        lower_mask = outputs["lower_mask"]

        self.spotless_module.train()
        spot_loss = self.spotless_loss(
            pred_mask_up.flatten(), upper_mask.flatten(), lower_mask.flatten()
        )
        reg = 0.5 * self.spotless_module.get_regularizer()
        spot_loss = spot_loss + reg
        pl_module.manual_backward(spot_loss)

        pl_module.log("train/spot", spot_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=pl_module.batch_size)

    def reset_shs_rest(self, outputs, batch, gaussian_model, step, pl_module):
        if step != self.config.reset_sh:
            return

        with torch.no_grad():
            gaussian_model.shs_rest.clamp_(max=0.001)
            gaussian_model.update_properties(DensityControllerUtils.replace_tensors_to_optimizers({"shs_rest": gaussian_model.shs_rest}, pl_module.gaussian_optimizers))

    def update_running_stats(self, outputs, batch, gaussian_model, step, pl_module):
        if step > self.config.densify_until_iter:
            return

        # [1, H, W, C]
        predict = outputs["render"].permute(1, 2, 0).unsqueeze(0)
        gt_image = batch[1][1].permute(1, 2, 0).unsqueeze(0)

        # update hist_err
        err = torch.histogram(
            torch.mean(torch.abs(predict - gt_image), dim=-3).clone().detach().cpu(),  # mean alone H?
            bins=self.config.bin_size,
            range=(0.0, 1.0),
        )[0].to(predict.device)
        self.hist_err = 0.95 * self.hist_err + err

        # update avg_err
        mid_err = torch.sum(self.hist_err) * self.config.robust_percentile
        self.avg_err = torch.linspace(0, 1, self.config.bin_size + 1)[
            torch.where(torch.cumsum(self.hist_err, 0) >= mid_err)[0][
                0
            ]
        ]

        # update lower_err and upper_err
        lower_err = torch.sum(self.hist_err) * self.config.lower_bound
        upper_err = torch.sum(self.hist_err) * self.config.upper_bound

        self.lower_err = torch.linspace(0, 1, self.config.bin_size + 1)[
            torch.where(torch.cumsum(self.hist_err, 0) >= lower_err)[
                0
            ][0]
        ]
        self.upper_err = torch.linspace(0, 1, self.config.bin_size + 1)[
            torch.where(torch.cumsum(self.hist_err, 0) >= upper_err)[
                0
            ][0]
        ]

        # TODO: UBP

    @property
    def hist_err(self) -> torch.Tensor:
        return self._hist_err

    @hist_err.setter
    def hist_err(self, v):
        self._hist_err.copy_(v)

    @property
    def avg_err(self) -> float:
        return self._avg_err.item()

    @avg_err.setter
    def avg_err(self, v):
        self._avg_err.fill_(v)

    @property
    def lower_err(self) -> float:
        return self._lower_err.item()

    @lower_err.setter
    def lower_err(self, v):
        self._lower_err.fill_(v)

    @property
    def upper_err(self):
        return self._upper_err.item()

    @upper_err.setter
    def upper_err(self, v):
        self._upper_err.fill_(v)

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        camera, image_info, sd_feature = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]

        # calculate loss
        if masked_pixels is not None:
            gt_image = gt_image.clone()
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.

        # spotless
        error_per_pixel = torch.abs(outputs["render"] - gt_image)  # [C, H, W]
        pred_mask_up, pred_mask, lower_mask, upper_mask = self.mlp_mask(
            sd_feature.unsqueeze(0),  # [1, C, H, W]
            error_per_pixel.permute(1, 2, 0).unsqueeze(0),  # [1, H, W, C]
            height=image.shape[1],
            width=image.shape[2],
        )  # [N_pixels, 1], [1, H, W, 1], ...
        if self.config.schedule is True:
            # schedule sampling of the mask based on alpha
            alpha = np.exp(self.config.schedule_beta * np.floor((1 + step) / 1.5))
            pred_mask = torch.bernoulli(
                torch.clip(
                    alpha + (1 - alpha) * pred_mask.clone().detach(),
                    min=0.0,
                    max=1.0,
                )
            )
        rgb_diff_loss = (pred_mask.squeeze(-1).clone().detach() * error_per_pixel).mean()

        # add predicted masks to output dict, which is required by the on_after_backward hook
        outputs["pred_mask_up"] = pred_mask_up
        outputs["pred_mask"] = pred_mask
        outputs["lower_mask"] = lower_mask
        outputs["upper_mask"] = upper_mask

        # TODO: apply mask to SSIM metric
        ssim_metric = ssim(outputs["render"], gt_image)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

        return {
            "loss": loss,
            "rgb_diff": rgb_diff_loss,
            "ssim": ssim_metric,
        }, {
            "loss": True,
            "rgb_diff": True,
            "ssim": True,
        }

    def cluster_mask(self, sf, colors, error_per_pixel):
        pred_mask = self.robust_mask(
            error_per_pixel, self.avg_err,
        )

        # cluster the semantic feature and mask based on cluster voting
        sf = nn.Upsample(
            size=(colors.shape[1], colors.shape[2]),
            mode="nearest",
        )(sf).squeeze(0)
        pred_mask = self.robust_cluster_mask(pred_mask, semantics=sf)

        return pred_mask

    def mlp_mask(self, sf, error_per_pixel, height: int, width: int):
        """
        use spotless mlp to predict the mask

        Args:
            sf: stable diffusion feature map, in [1, C, H, W]
            error_per_pixel: in [1, H, W, C]
            height: mask height
            width: mask width
        """

        sf = nn.Upsample(
            size=(height, width),
            mode="bilinear",
        )(sf).squeeze(0)  # [C, H, W]
        # TODO: smaller image to reduce memory consumption
        pos_enc = self.get_positional_encodings(
            height, width, 20
        ).permute((2, 0, 1))  # [C, H, W]
        sf = torch.cat([sf, pos_enc], dim=0)
        sf_flat = sf.reshape(sf.shape[0], -1).permute((1, 0))  # [N_pixels, N_features]
        self.spotless_module.eval()
        pred_mask_up = self.spotless_module(sf_flat)  # [N_pixels, 1]
        pred_mask = pred_mask_up.reshape(
            1, height, width, 1
        )  # [1, H, W, 1]
        # calculate lower and upper bound masks for spotless mlp loss
        lower_mask = self.robust_mask(
            error_per_pixel, self.lower_err
        )
        upper_mask = self.robust_mask(
            error_per_pixel, self.upper_err
        )

        return pred_mask_up, pred_mask, lower_mask, upper_mask

    @classmethod
    def robust_mask(
            cls,
            error_per_pixel: torch.Tensor,
            loss_threshold: float,
    ) -> torch.Tensor:
        epsilon = 1e-3
        error_per_pixel = error_per_pixel.mean(axis=-1, keepdims=True)
        error_per_pixel = error_per_pixel.squeeze(-1).unsqueeze(0)
        is_inlier_pixel = (error_per_pixel < loss_threshold).float()
        window_size = 3
        channel = 1
        window = torch.ones((1, 1, window_size, window_size), dtype=torch.float) / (
                window_size * window_size
        )
        if error_per_pixel.is_cuda:
            window = window.cuda(error_per_pixel.get_device())
        window = window.type_as(error_per_pixel)
        has_inlier_neighbors = F.conv2d(
            is_inlier_pixel, window, padding=window_size // 2, groups=channel
        )
        has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()
        is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > epsilon).float()
        pred_mask = is_inlier_pixel.squeeze(0).unsqueeze(-1)
        return pred_mask

    @staticmethod
    def get_positional_encodings(
            height: int, width: int, num_frequencies: int, device: str = "cuda"
    ) -> torch.Tensor:
        """Generates positional encodings for a given image size and frequency range.

        Args:
          height: height of the image
          width: width of the image
          num_frequencies: number of frequencies
          device: device to use

        Returns:

        """
        # Generate grid of (x, y) coordinates
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )

        # Normalize coordinates to the range [0, 1]
        y = y / (height - 1)
        x = x / (width - 1)

        # Create frequency range [1, 2, 4, ..., 2^(num_frequencies-1)]
        frequencies = (
                torch.pow(2, torch.arange(num_frequencies, device=device)).float() * torch.pi
        )

        # Compute sine and cosine of the frequencies multiplied by the coordinates
        y_encodings = torch.cat(
            [torch.sin(frequencies * y[..., None]), torch.cos(frequencies * y[..., None])],
            dim=-1,
        )
        x_encodings = torch.cat(
            [torch.sin(frequencies * x[..., None]), torch.cos(frequencies * x[..., None])],
            dim=-1,
        )

        # Combine the encodings
        pos_encodings = torch.cat([y_encodings, x_encodings], dim=-1)

        return pos_encodings

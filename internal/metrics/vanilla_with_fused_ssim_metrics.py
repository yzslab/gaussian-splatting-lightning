from dataclasses import dataclass
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl
from fused_ssim import fused_ssim


@dataclass
class VanillaWithFusedSSIMMetrics(VanillaMetrics):
    def instantiate(self, *args, **kwargs) -> "VanillaWithFusedSSIMMetricsModule":
        return VanillaWithFusedSSIMMetricsModule(self)


class VanillaWithFusedSSIMMetricsModule(VanillaMetricsImpl):
    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]

        # calculate loss
        if masked_pixels is not None:
            gt_image = gt_image.clone()
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.
        rgb_diff_loss = self.rgb_diff_loss_fn(outputs["render"], gt_image)
        ssim_metric = fused_ssim(outputs["render"].unsqueeze(0), gt_image.unsqueeze(0))
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

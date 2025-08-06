from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any
import torch
from .metric import Metric, MetricModule


@dataclass
class GroundRegMetricConfig:
    up_direction: Tuple[float, float, float] = None

    ground_alt: Union[float, Tuple[float, float, float]] = None

    ground_reg_lambda: float = 1.

    ground_reg_interval: int = 10


class GroundRegMetricCalculator(torch.nn.Module):
    def __init__(self, config: GroundRegMetricConfig):
        super().__init__()

        self.config = config

        self.register_buffer("up_direction", torch.tensor(config.up_direction, dtype=torch.float), persistent=False)
        self.up_direction /= torch.linalg.norm(self.up_direction)

        if isinstance(config.ground_alt, tuple):
            ground_alt = torch.dot(self.up_direction, torch.tensor(ground_alt, dtype=torch.float))
            print("ground_alt={}".format(ground_alt))
        else:
            ground_alt = torch.tensor(config.ground_alt, dtype=torch.float)

        self.register_buffer("ground_alt", ground_alt, persistent=False)

    def forward(self, gaussian_model):
        means = gaussian_model.get_means()
        z = torch.inner(means, self.up_direction.to(device=means.device))
        alt = self.ground_alt.to(device=z.device) - z
        reg_mask = alt.detach() > 0
        reg_loss = alt[reg_mask].sum() / (reg_mask.sum() + 1)

        return reg_loss * self.config.ground_reg_lambda, reg_loss, alt, reg_mask


class GroundRegMetricModuleMixin:
    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        if stage == "fit":
            self.ground_reg_metric_calculator = GroundRegMetricCalculator(self.config).to(device=pl_module.device)
            _, _, alt, mask = self.ground_reg_metric_calculator(pl_module.gaussian_model)
            with torch.no_grad():
                pl_module.gaussian_model.gaussians["means"][mask] += alt[mask].unsqueeze(-1) * self.ground_reg_metric_calculator.up_direction.to(device=alt.device).unsqueeze(0)
                pl_module.gaussian_model.gaussians["opacities"][mask].fill_(pl_module.gaussian_model.opacity_inverse_activation(torch.tensor(0., dtype=torch.float, device=mask.device)))
                pl_module.gaussian_model.gaussians["scales"][mask].fill_(pl_module.gaussian_model.scale_inverse_activation(torch.tensor(0.0001, dtype=torch.float, device=mask.device)))
                print("{} points reset to ground".format(mask.sum()))

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(
            pl_module,
            gaussian_model,
            step,
            batch,
            outputs,
        )

        if step % self.config.ground_reg_interval == 0:
            ground_reg_loss = self.ground_reg_metric_calculator(gaussian_model)[0]
            metrics["loss"] = metrics["loss"] + ground_reg_loss
            metrics["ground"] = ground_reg_loss
            pbar["ground"] = False

        return metrics, pbar

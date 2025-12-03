from typing import Tuple, List, Optional, Any
from internal.configs.instantiate_config import InstantiatableConfig
from dataclasses import dataclass
import torch
from .output_processors import OutputProcessor


@dataclass
class Bilagrid:
    BilateralGrid: Any

    slice: Any

    total_variation_loss: Any


@dataclass
class BilagridProcessor(OutputProcessor):
    lr_init: float = 2e-3

    lr_final_factor: float = 0.01

    max_steps: int = 30_000

    grid_x: int = 16

    grid_y: int = 16

    grid_w: int = 8

    tv_loss_weight: float = 10.

    fused: bool = True

    def instantiate(self, *args, **kwargs):
        return BilagridProcessorModule(self)

    def import_lib(self):
        if self.fused:
            from fused_bilagrid import BilateralGrid, slice, total_variation_loss
        else:
            from internal.utils.lib_bilagrid import BilateralGrid, slice, total_variation_loss
        return Bilagrid(BilateralGrid, slice, total_variation_loss)


class BilagridProcessorModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bilagrid_interfaces = self.config.import_lib()

    def init_bilagrid(self, n_grids: int, device):
        self.bgrid = self.bilagrid_interfaces.BilateralGrid(
            num=n_grids,
            grid_X=self.config.grid_x,
            grid_Y=self.config.grid_y,
            grid_W=self.config.grid_w,
        ).to(device=device)

    def setup(self, stage: str, pl_module=None, *args, **kwargs):
        if pl_module is not None:
            max_input_id = 0
            appearance_group_ids = pl_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
            if appearance_group_ids is not None:
                for i in appearance_group_ids.values():
                    if i[0] > max_input_id:
                        max_input_id = i[0]
            n_appearances = max_input_id + 1
            assert n_appearances > 1

            self.init_bilagrid(n_appearances, pl_module.device)

            print("{} appearance groups".format(n_appearances))

    def load_state_dict(self, state_dict, strict=True):
        n_appearances = state_dict["bgrid.grids"].shape[0]
        self.init_bilagrid(n_appearances, state_dict["bgrid.grids"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, pl_module):
        params = [
            {"params": self.bgrid.parameters(), "name": "bgrid"},
        ]

        optimizer = torch.optim.Adam(
            params=params,
            lr=self.config.lr_init,
            eps=1e-15,
        )

        pl_module.extra_train_metrics.append(self.tv_loss)

        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.config.lr_final_factor ** min(iter / self.config.max_steps, 1),
            verbose=False,
        )

    def tv_loss(self, outputs, batch, gaussian_model, global_step, pl_module, metrics, pbar):
        tv_loss = self.config.tv_loss_weight * self.bilagrid_interfaces.total_variation_loss(self.bgrid.grids)
        metrics["loss"] = metrics["loss"] + tv_loss
        metrics["tv"] = tv_loss
        pbar["tv"] = True

    def build_grid_xy(self, camera):
        device = camera.device
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1., camera.height, device=device),
            torch.linspace(0, 1., camera.width, device=device),
            indexing='ij',
        )
        return torch.stack([grid_x, grid_y], dim=-1)  # (img_h, img_w, 2)

    def forward(self, camera, outputs) -> None:
        if "render" not in outputs:
            return

        outputs["render"] = self.bilagrid_interfaces.slice(
            self.bgrid,
            xy=self.build_grid_xy(camera).unsqueeze(0),
            rgb=outputs["render"].permute(1, 2, 0).unsqueeze(0),
            grid_idx=camera.appearance_id[None, None],
        )["rgb"].squeeze(0).permute(2, 0, 1)

    def training_forward(self, batch, outputs):
        return self(batch[0], outputs)

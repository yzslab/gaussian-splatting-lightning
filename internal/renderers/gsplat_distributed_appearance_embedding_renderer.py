from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List, Any

import lightning
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from gsplat.sh import spherical_harmonics
from . import Renderer
from .gsplat_distributed_renderer import GSplatDistributedRenderer, GSplatDistributedRendererImpl, MemberData
from .gsplat_appearance_embedding_renderer import Model as AppearanceEmbeddingModel, ModelConfig, OptimizationConfig, GSplatAppearanceEmbeddingRendererModule as GSplatAppearanceEmbeddingRenderer


class GSplatDistributedAppearanceEmbeddingRendererImpl(GSplatDistributedRendererImpl):
    config: "GSplatDistributedAppearanceEmbeddingRenderer"

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        self.appearance_model = AppearanceEmbeddingModel(self.config.appearance)

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
        super().training_setup(module)

        # check `n_appearances`
        max_input_id = 0
        appearance_group_ids = module.trainer.datamodule.dataparser_outputs.appearance_group_ids
        if appearance_group_ids is not None:
            for i in appearance_group_ids.values():
                if i[0] > max_input_id:
                    max_input_id = i[0]
        n_appearances = max_input_id + 1
        assert self.config.appearance.n_appearances >= n_appearances, "`n_appearances` must be >= {}".format(n_appearances)

        print(self.appearance_model)
        self.appearance_model = DDP(self.appearance_model, device_ids=[module.device.index])

        embedding_optimizer, embedding_scheduler = GSplatAppearanceEmbeddingRenderer._create_optimizer_and_scheduler(
            self.appearance_model.module.embedding.parameters(),
            "embedding",
            lr_init=self.config.appearance_optimization.embedding_lr_init,
            lr_final_factor=self.config.appearance_optimization.lr_final_factor,
            max_steps=self.config.appearance_optimization.max_steps,
            eps=self.config.appearance_optimization.eps,
            warm_up=self.config.appearance_optimization.warm_up,
        )
        network_optimizer, network_scheduler = GSplatAppearanceEmbeddingRenderer._create_optimizer_and_scheduler(
            self.appearance_model.module.network.parameters(),
            "embedding_network",
            lr_init=self.config.appearance_optimization.lr_init,
            lr_final_factor=self.config.appearance_optimization.lr_final_factor,
            max_steps=self.config.appearance_optimization.max_steps,
            eps=self.config.appearance_optimization.eps,
            warm_up=self.config.appearance_optimization.warm_up,
        )

        return [embedding_optimizer, network_optimizer], [embedding_scheduler, network_scheduler]

    def get_rgbs(self, pc, member_data: MemberData, projection_results) -> torch.Tensor:
        radii = projection_results[2]
        is_gaussian_visible = radii > 0

        detached_xyz = pc.get_xyz.detach()
        view_directions = detached_xyz[is_gaussian_visible] - member_data.camera_center  # (N, 3)
        view_directions = view_directions / view_directions.norm(dim=-1, keepdim=True)
        base_rgb = spherical_harmonics(pc.active_sh_degree, view_directions, pc.get_features[is_gaussian_visible]) + 0.5
        rgb_offset = self.appearance_model(pc.get_appearance_features()[is_gaussian_visible], member_data.appearance_id, view_directions) * 2 - 1.
        rgbs = torch.zeros((radii.shape[0], 3), dtype=projection_results[0].dtype, device=radii.device)
        rgbs[is_gaussian_visible] = torch.clamp(base_rgb + rgb_offset, min=0., max=1.)

        return rgbs


@dataclass
class GSplatDistributedAppearanceEmbeddingRenderer(GSplatDistributedRenderer):
    appearance: ModelConfig = field(default_factory=lambda: ModelConfig())

    appearance_optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> Renderer:
        # TODO: enable warm up
        assert self.appearance_optimization.warm_up == 0, "`warm_up` must be `0` currently"
        return GSplatDistributedAppearanceEmbeddingRendererImpl(self)

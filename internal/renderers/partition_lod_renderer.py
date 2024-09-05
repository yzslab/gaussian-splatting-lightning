from dataclasses import dataclass
from typing import Any, Dict
import os

import torch
from tqdm.auto import tqdm

from . import RendererOutputInfo
from .gsplat_renderer import GSPlatRenderer
from .renderer import RendererConfig, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel


@dataclass
class PartitionLoDRenderer(RendererConfig):
    data: str
    names: list[str]  # from finest to coarsest
    min_images: int = 32

    def instantiate(self, *args, **kwargs) -> "PartitionLoDRendererModule":
        return PartitionLoDRendererModule(self)


class PartitionLoDRendererModule(Renderer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.gsplat_renderer = GSPlatRenderer()

        self.on_render_hooks = []
        self.on_model_updated_hooks = []

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "utils",
        ))

        device = torch.device("cuda")

        # load partition data
        from internal.utils.partitioning_utils import PartitionCoordinates
        partitions = torch.load(os.path.join(self.config.data, "partitions.pt"))
        self.orientation_transform = torch.eye(3)
        if partitions["extra_data"] is not None:
            self.orientation_transform = partitions["extra_data"]["rotation_transform"]
        self.orientation_transform = self.orientation_transform.T.to(device=device)[:3, :3]

        partition_size = partitions["scene_config"]["partition_size"]
        self.partition_size = partition_size

        self.partition_coordinates = PartitionCoordinates(**partitions["partition_coordinates"])
        self.partition_bounding_boxes = self.partition_coordinates.get_bounding_boxes(partition_size).to(device=device)

        # load partitions' models
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        from utils.train_partitions import PartitionTraining, PartitionTrainingConfig
        lods = []  # [N_lods, N_partitions]

        for lod in tqdm(self.config.names):
            models = []

            partition_training = PartitionTraining(PartitionTrainingConfig(
                partition_dir=self.config.data,
                project_name=lod,
                min_images=self.config.min_images,
                n_processes=1,
                process_id=1,
                dry_run=False,
                extra_epoches=0,
            ))
            trainable_partition_idx_list = partition_training.get_trainable_partition_idx_list(
                min_images=self.config.min_images,
                n_processes=1,
                process_id=1,
            )

            for partition_idx in tqdm(trainable_partition_idx_list, leave=False):
                ckpt = torch.load(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "outputs",
                    lod,
                    partition_training.get_experiment_name(partition_idx),
                    "preprocessed.ckpt"
                ), map_location="cpu")
                gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)
                gaussian_model.pre_activate_all_properties()
                gaussian_model.freeze()
                gaussian_model.to(device=device)
                models.append(gaussian_model)

            lods.append(models)
        self.lods = lods

        # set default LoD distance thresholds
        self.lod_thresholds = (torch.arange(1, len(self.lods)) * 0.25 * partition_size).to(device=device)  # [N_lods - 1]

        # initialize partition lod states
        self.n_partitions = len(self.lods[0])
        self.partition_lods = torch.empty(self.n_partitions, dtype=torch.int8, device=device).fill_(127)  # [N_partitions]

        # setup empty Gaussian Model
        self.gaussian_model = gaussian_model.config.instantiate()
        self.gaussian_model.setup_from_number(0)
        self.gaussian_model.pre_activate_all_properties()
        self.gaussian_model.freeze()
        self.gaussian_model.active_sh_degree = gaussian_model.active_sh_degree
        self.gaussian_model.to(device=device)

        # setup gsplat renderer
        self.gsplat_renderer.setup(stage)

    def get_partition_distances(self, p: torch.Tensor):
        p = (p @ self.orientation_transform)[:2]
        dist_min2p = self.partition_bounding_boxes.min - p
        dist_p2max = p - self.partition_bounding_boxes.max
        dxy = torch.maximum(dist_min2p, dist_p2max)
        return torch.sqrt(torch.pow(dxy.clamp(min=0.), 2).sum(dim=-1))  # [N_partitions]

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        partition_distances = self.get_partition_distances(viewpoint_camera.camera_center)
        self.partition_distances = partition_distances
        partition_lods = torch.ones_like(self.partition_lods).fill_(-1)

        # set lods by distances
        for i in range(len(self.lods) - 2, -1, -1):
            partition_lods[partition_distances < self.lod_thresholds[i]] = i

        if not torch.all(torch.eq(partition_lods, self.partition_lods)):
            # update stored lods
            self.partition_lods = partition_lods
            # update model
            properties = {}
            for partition_idx, lod in enumerate(partition_lods):
                for k, v in self.lods[lod.item()][partition_idx].properties.items():
                    properties.setdefault(k, []).append(v)

            for k in list(properties.keys()):
                properties[k] = torch.concat(properties[k], dim=0).to(device="cuda")

            self.gaussian_model.properties = properties

            for i in self.on_model_updated_hooks:
                i()

        for i in self.on_render_hooks:
            i()

        return self.gsplat_renderer(
            viewpoint_camera,
            self.gaussian_model,
            bg_color,
            scaling_modifier,
            render_types,
            **kwargs,
        )

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return self.gsplat_renderer.get_available_outputs()

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        with tabs.add_tab("LoD"):
            self.viewer_options = ViewerOptions(self, viewer, server)


import viser
from viser import GuiMarkdownHandle
from viser import ViserServer


class ViewerOptions:
    def __init__(self, renderer: PartitionLoDRendererModule, viewer, server):
        super().__init__()
        self.renderer = renderer
        self.viewer = viewer
        self.server: ViserServer = server

        self.show_label = False

        self.setup_status_folder()
        self.setup_labels()
        self.setup_settings_folder()

        self.renderer.on_model_updated_hooks.append(self.update_number_of_gaussians)

    def setup_status_folder(self):
        with self.server.gui.add_folder("Status"):
            self.markdown: GuiMarkdownHandle = self.server.gui.add_markdown("")

    def setup_settings_folder(self):
        def get_lod_threshold_updater(idx):
            def updater(v: viser.GuiEvent):
                self.renderer.lod_thresholds[idx] = v.target.value

            return updater

        with self.server.gui.add_folder("Settings"):
            # threshold
            with self.server.gui.add_folder("Max Distance"):
                for idx, i in enumerate(self.renderer.lod_thresholds):
                    lod_threshold_number = self.server.gui.add_number(label="LoD {}".format(idx), initial_value=i.item())
                    lod_threshold_number.on_update(get_lod_threshold_updater(idx))
                    self.viewer.rerender_for_all_client()

            # labels
            show_label_checkbox = self.server.gui.add_checkbox("Show Labels", initial_value=self.show_label)

            @show_label_checkbox.on_update
            def _(_):
                self.show_label = show_label_checkbox.value
                self.update_labels()

    def setup_labels(self):
        self.label_updaters = []

        def get_label_updater(name, text, position):
            label = None

            def updater(distance, lod):
                nonlocal label
                if label is not None:
                    label.remove()
                    label = None
                if not self.show_label:
                    return
                label = self.server.scene.add_label(
                    name=name,
                    text="{} - {:.2f} - {}".format(text, distance, lod),
                    position=position,
                )

            return updater

        for idx, (id, xy) in enumerate(self.renderer.partition_coordinates):
            self.label_updaters.append(get_label_updater(
                name="part/{}".format(idx),
                text="({},{})".format(idx, *id.tolist()),
                position=(torch.concat([xy + 0.5 * self.renderer.partition_size, torch.tensor([0.])]).cuda() @ self.renderer.orientation_transform.T).cpu().numpy(),
            ))

        self.renderer.on_render_hooks.append(self.update_labels)

    def update_number_of_gaussians(self):
        self.markdown.content = "Gaussians: {}".format(
            self.renderer.gaussian_model.n_gaussians,
        )

    def update_labels(self):
        for idx, i in enumerate(self.label_updaters):
            i(self.renderer.partition_distances[idx].item(), self.renderer.partition_lods[idx].item())

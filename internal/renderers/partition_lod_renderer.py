import random
from dataclasses import dataclass
from typing import Any, Dict
import os
import time
import numpy as np
import torch
from pytorch3d.ops.iou_box3d import _check_coplanar, _check_nonzero, _box3d_overlap
from tqdm.auto import tqdm
from threading import Lock

from . import RendererOutputInfo
from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1, spherical_harmonics
from .renderer import RendererConfig, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.utils.sh_utils import RGB2SH, C0


@dataclass
class PartitionLoDRenderer(RendererConfig):
    data: str
    names: list[str]  # from finest to coarsest
    min_images: int = 32
    visibility_filter: bool = False
    freeze: bool = False
    drop_shs_rest: bool = False

    lod_distances: list[int] = None
    """ i_distance = lod_distances[i] * default_partition_size, from finest to coarsest """

    ckpt_name: str = "preprocessed.ckpt"

    def instantiate(self, *args, **kwargs) -> "PartitionLoDRendererModule":
        if self.ckpt_name.endswith(".ckpt"):
            self.ckpt_name = self.ckpt_name[:self.ckpt_name.rfind(".")]

        return PartitionLoDRendererModule(self)


class PartitionLoDRendererModule(Renderer):
    MODEL_PROPERTIES = [
        "means",
        "shs",
        "opacities",
        "scales",
        "rotations",
    ]

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.synchronized = False

        self.on_render_hooks = []
        self.on_model_updated_hooks = []

        self.has_appearance_model = False
        self.previous_appearance_id = torch.tensor(-1, dtype=torch.long)
        self.appearance_lock = Lock()

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

        default_partition_size = partitions["scene_config"]["partition_size"]
        self.default_partition_size = default_partition_size
        if "size" not in partitions["partition_coordinates"]:
            print("Use default size")
            partitions["partition_coordinates"]["size"] = torch.full(
                (partitions["partition_coordinates"]["xy"].shape[0], 2),
                default_partition_size,
                dtype=torch.float,
                device=partitions["partition_coordinates"]["xy"].device,
            )

        self.partition_coordinates = PartitionCoordinates(**partitions["partition_coordinates"])
        self.partition_bounding_boxes = self.partition_coordinates.get_bounding_boxes().to(device=device)

        # load partitions' models
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        from utils.train_partitions import PartitionTraining, PartitionTrainingConfig
        lods = []  # [N_lods, N_partitions]
        lods_appearance_models = []

        for lod in tqdm(self.config.names):
            models = []
            appearance_models = []

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

            loaded_partition_idx_list = []
            for partition_idx in tqdm(trainable_partition_idx_list, leave=False):
                try:
                    ckpt = torch.load(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "outputs",
                        lod,
                        partition_training.get_experiment_name(partition_idx),
                        "{}.ckpt".format(self.config.ckpt_name),
                    ), map_location="cpu")
                except:
                    tqdm.write("[WARNING]Checkpoint '{}.ckpt' of partition {} of not found in {}".format(self.config.ckpt_name, partition_training.get_partition_id_str(partition_idx), lod))
                    continue

                loaded_partition_idx_list.append(partition_idx)
                gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device)
                if gaussian_model.__class__.__name__ == "AppearanceFeatureGaussianModel":
                    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, "validate", device)
                    appearance_models.append(renderer.model)
                    gaussian_model.shs_dc_backup = gaussian_model.get_shs_dc()
                    if renderer.model.config.with_opacity:
                        gaussian_model.opacities_backup = gaussian_model.get_opacities()
                    self.has_appearance_model = True

                if self.config.drop_shs_rest:
                    gaussian_model.config.sh_degree = 0
                    gaussian_model.active_sh_degree = 0
                    gaussian_model.shs_rest = torch.empty((gaussian_model.n_gaussians, 0, 3), device=gaussian_model.shs_dc.device)
                gaussian_model.pre_activate_all_properties()
                gaussian_model.freeze()
                gaussian_model.to(device=device)
                models.append(gaussian_model)

            lods.append(models)
            lods_appearance_models.append(appearance_models)
        self.lods = lods
        self.lods_appearance_models = lods_appearance_models

        # retain trainable partitions only
        trainable_partition_idx = torch.tensor(loaded_partition_idx_list)
        del trainable_partition_idx_list
        self.partition_coordinates.id = self.partition_coordinates.id[trainable_partition_idx]
        self.partition_coordinates.xy = self.partition_coordinates.xy[trainable_partition_idx]
        self.partition_coordinates.size = self.partition_coordinates.size[trainable_partition_idx]
        self.partition_bounding_boxes.min = self.partition_bounding_boxes.min[trainable_partition_idx]
        self.partition_bounding_boxes.max = self.partition_bounding_boxes.max[trainable_partition_idx]

        # get partition 3D bounding boxes
        partition_min_max_z = []
        for partition_model in self.lods[0]:
            partition_z_value = partition_model.get_means() @ self.orientation_transform[:, -1:]
            partition_min_max_z.append(torch.stack([torch.quantile(partition_z_value, 0.005), torch.quantile(partition_z_value, 0.995)]))
        partition_min_max_z = torch.stack(partition_min_max_z)
        # print(partition_min_max_z)
        partition_full_2d_bounding_box = []
        for _, partition_xy, partition_size in self.partition_coordinates:
            partition_xy = partition_xy.to(device=device)
            partition_size = partition_size.to(device=device)
            left_down = partition_xy
            left_top = partition_xy + torch.tensor([0., partition_size[1]], device=device)
            right_top = partition_xy + partition_size
            right_down = partition_xy + torch.tensor([partition_size[0], 0.], device=device)
            # clockwise
            partition_full_2d_bounding_box.append(torch.stack([
                left_down,
                left_top,
                right_top,
                right_down,
            ]))
        partition_full_2d_bounding_box = torch.stack(partition_full_2d_bounding_box)  # [N_partitions, 4, 2]
        partition_min_max_z = partition_min_max_z[:, None, :].repeat(1, 4, 1)
        # from top to bottom
        self.partition_full_3d_bounding_box = torch.concat([
            torch.concat([partition_full_2d_bounding_box, partition_min_max_z[..., 1:2]], dim=-1),
            torch.concat([partition_full_2d_bounding_box, partition_min_max_z[..., 0:1]], dim=-1),
        ], dim=1) @ self.orientation_transform.T  # [N_partitions, 8, 3], in world space
        # print(self.partition_full_3d_bounding_box)

        # set default LoD distance thresholds
        self.lod_thresholds = (torch.arange(1, len(self.lods)) * 0.25 * default_partition_size).to(device=device)  # [N_lods - 1]
        if self.config.lod_distances is not None:
            assert len(self.config.lod_distances) == len(self.lods) - 1
            previous = 0
            for i in self.config.lod_distances:
                assert i >= previous
                previous = i

            self.lod_thresholds = torch.tensor(self.config.lod_distances, dtype=torch.float, device=device) * self.default_partition_size
            

        # initialize partition lod states
        self.n_partitions = len(self.lods[0])
        self.partition_lods = torch.empty(self.n_partitions, dtype=torch.int8, device=device).fill_(127)  # [N_partitions]
        self.is_partition_visible = torch.ones(self.n_partitions, dtype=torch.bool, device=device)

        # setup empty Gaussian Model
        self.gaussian_model = VanillaGaussian(sh_degree=gaussian_model.config.sh_degree).instantiate()
        self.gaussian_model.setup_from_number(0)
        self.gaussian_model.pre_activate_all_properties()
        self.gaussian_model.freeze()
        self.gaussian_model.active_sh_degree = gaussian_model.active_sh_degree
        self.gaussian_model.to(device=device)

        # setup gsplat renderer
        renderer_config = GSplatV1Renderer(
            block_size=getattr(ckpt["hyper_parameters"]["renderer"], "block_size", 16),
            anti_aliased=getattr(ckpt["hyper_parameters"]["renderer"], "anti_aliased", True),
            filter_2d_kernel_size=getattr(ckpt["hyper_parameters"]["renderer"], "filter_2d_kernel_size", 0.3),
            separate_sh=getattr(ckpt["hyper_parameters"]["renderer"], "separate_sh", True),
            tile_based_culling=getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
        )
        print(renderer_config)
        self.gsplat_renderer = renderer_config.instantiate()
        self.gsplat_renderer.setup(stage)

    def cuda_synchronize(self):
        if self.synchronized:
            return torch.cuda.synchronize()

    @staticmethod
    def box3d_overlap(boxes1, boxes2):
        """
        Computes the intersection of 3D boxes1 and boxes2.

        Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
        (where B doesn't have to be the same for boxes1 and boxes2),
        containing the 8 corners of the boxes, as follows:

            (4) +---------+. (5)
                | ` .     |  ` .
                | (0) +---+-----+ (1)
                |     |   |     |
            (7) +-----+---+. (6)|
                ` .   |     ` . |
                (3) ` +---------+ (2)


        NOTE: Throughout this implementation, we assume that boxes
        are defined by their 8 corners exactly in the order specified in the
        diagram above for the function to give correct results. In addition
        the vertices on each plane must be coplanar.
        As an alternative to the diagram, this is a unit bounding
        box which has the correct vertex ordering:

        box_corner_vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

        Args:
            boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
            boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
        Returns:
            vol: (N, M) tensor of the volume of the intersecting convex shapes
            iou: (N, M) tensor of the intersection over union which is
                defined as: `iou = vol / (vol1 + vol2 - vol)`
        """
        if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
            raise ValueError("Each box in the batch must be of shape (8, 3)")

        _check_coplanar(boxes1, 1e-2)
        _check_coplanar(boxes2, 1e-2)
        _check_nonzero(boxes1, 1e-5)
        _check_nonzero(boxes2, 1e-5)

        vol, iou = _box3d_overlap.apply(boxes1, boxes2)

        return vol, iou

    def get_partition_distances(self, p: torch.Tensor):
        p = (p @ self.orientation_transform)[:2]
        dist_min2p = self.partition_bounding_boxes.min - p
        dist_p2max = p - self.partition_bounding_boxes.max
        dxy = torch.maximum(dist_min2p, dist_p2max)
        return torch.sqrt(torch.pow(dxy.clamp(min=0.), 2).sum(dim=-1))  # [N_partitions]

    @torch.no_grad()
    def update_shs_dc(self, appearance_id):
        if appearance_id == self.previous_appearance_id:
            return

        for lod_idx in range(len(self.lods)):
            gaussian_model_list = self.lods[lod_idx]
            appearance_model_list = self.lods_appearance_models[lod_idx]
            for gaussian_model, appearance_model in zip(gaussian_model_list, appearance_model_list):
                predicts = appearance_model(gaussian_model.get_appearance_features().cuda(), appearance_id, None).float()

                rgb_offsets = predicts[..., :3] * 2. - 1.
                shs_dc = RGB2SH(rgb_offsets)

                gaussian_model.gaussians["shs"][:, 0] = gaussian_model.shs_dc_backup.squeeze(1) + shs_dc + 0.5 / C0
                if appearance_model.config.with_opacity:
                    gaussian_model.gaussians["opacities"] = torch.clamp_max(gaussian_model.opacities_backup + predicts[..., 3:], max=1.)

        self.previous_appearance_id = appearance_id

        # force update model
        self.is_partition_visible.fill_(0)

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        self.cuda_synchronize()

        pipeline_started_at = time.time()
        if self.has_appearance_model:
            if viewpoint_camera.appearance_id != self.previous_appearance_id:
                with self.appearance_lock:
                    self.update_shs_dc(viewpoint_camera.appearance_id)
            self.cuda_synchronize()

        appearance_updated_at = time.time()

        partition_distances = self.get_partition_distances(viewpoint_camera.camera_center)
        # print((partition_distances == 0.).nonzero())
        self.partition_distances = partition_distances
        partition_lods = torch.ones_like(self.partition_lods).fill_(-1)

        # set lods by distances
        for i in range(len(self.lods) - 2, -1, -1):
            partition_lods[partition_distances < self.lod_thresholds[i]] = i

        # visibility
        # TODO: optimize visibility based filter, it does not correctly know whether a partition partly in front of the camera is invisible
        is_partition_visible = torch.ones_like(self.is_partition_visible)
        if self.config.visibility_filter:
            # build view frustum
            width = viewpoint_camera.width.item()
            height = viewpoint_camera.height.item()
            # clockwise
            four_corners = torch.tensor([
                # near panel
                [0., 0., 1.,],  # left top
                [width, 0., 1.,],  # right top
                [width, height, 1.,],  # right down
                [0., height, 1.,],  # right down
                # far panel
                [0., 0., 1.,],  # left top
                [width, 0., 1.,],  # right top
                [width, height, 1.,],  # right down
                [0., height, 1.,],  # right down
            ], dtype=torch.float, device=bg_color.device)
            K = GSplatV1.get_intrinsics_matrix(
                fx=viewpoint_camera.fx,
                fy=viewpoint_camera.fy,
                cx=viewpoint_camera.cx,
                cy=viewpoint_camera.cy,
                device=bg_color.device,
            )
            view_frustum = torch.matmul(four_corners, torch.linalg.inv(K).T)
            view_frustum[:4] *= 0.1  # near
            view_frustum[4:] *= 10 * torch.max(partition_distances)  # far, the `1.2` can be other number that > 1.

            # transform partition bounding box to camera space
            world2camera = viewpoint_camera.world_to_camera  # already transposed
            partition_bounding_box_in_camera_spaces = self.partition_full_3d_bounding_box @ world2camera[:3, :3] + world2camera[3, :3]  # [N_partitions, 8]

            # calculate intersections
            iset_vol, iou_3d = self.box3d_overlap(
                view_frustum.unsqueeze(0),
                partition_bounding_box_in_camera_spaces,
            )

            is_partition_visible = iset_vol[0] > 1e-8
            # the closest one always visible
            is_partition_visible[torch.argmin(partition_distances)] = True

        if not self.config.freeze and (not torch.all(torch.eq(self.partition_lods, partition_lods)) or not torch.all(torch.eq(self.is_partition_visible, is_partition_visible))):
            # update stored lods
            self.partition_lods = partition_lods
            self.is_partition_visible = is_partition_visible
            # update model
            properties = {}
            for partition_idx, lod in enumerate(partition_lods):
                if not is_partition_visible[partition_idx]:
                    continue

                lod_model_properties = self.lods[lod.item()][partition_idx].properties
                for k in self.MODEL_PROPERTIES:
                    v = lod_model_properties[k]
                    properties.setdefault(k, []).append(v)

            for k in list(properties.keys()):
                properties[k] = torch.concat(properties[k], dim=0).to(device="cuda")

            self.gaussian_model.properties = properties

            for i in self.on_model_updated_hooks:
                i()

        for i in self.on_render_hooks:
            i()

        self.cuda_synchronize()

        render_started_at = time.time()
        outputs = self.gsplat_renderer(
            viewpoint_camera,
            self.gaussian_model,
            bg_color,
            scaling_modifier,
            render_types,
            **kwargs,
        )

        self.cuda_synchronize()

        finished_at = time.time()
        # if self.config.visibility_filter:
        #     for idx, corner in enumerate(partition_corners_in_image_space[0]):
        #         red_value = 1.
        #         if partition_corners_in_image_space_in_homogeneous[0, idx, 2] <= 0.:
        #             red_value = 0.
        #         box_min = torch.clamp(corner - 16, min=0).to(torch.int)
        #         box_max = corner + 16
        #         box_max = torch.minimum(box_max, max_pixel_coor).to(torch.int)
        #         outputs["render"][1:3, box_min[1]:box_max[1], box_min[0]:box_max[0]] = (1 - red_value)
        #         outputs["render"][0, box_min[1]:box_max[1], box_min[0]:box_max[0]] = red_value

        outputs["n_gaussians"] = self.gaussian_model.n_gaussians
        outputs["time"] = finished_at - pipeline_started_at
        outputs["render_time_with_lod_preprocess"] = finished_at - appearance_updated_at
        outputs["render_time"] = finished_at - render_started_at

        return outputs

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return self.gsplat_renderer.get_available_outputs()

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        self.gsplat_renderer.setup_web_viewer_tabs(viewer, server, tabs)
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

        # self.draw_3d_boxes()

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

            # visibility filter
            visibility_filter_checkbox = self.server.gui.add_checkbox("Visibility Filter", initial_value=self.renderer.config.visibility_filter)

            @visibility_filter_checkbox.on_update
            def _(_):
                self.renderer.config.visibility_filter = visibility_filter_checkbox.value

            # visibility filter
            freeze_checkbox = self.server.gui.add_checkbox("Freeze", initial_value=self.renderer.config.freeze)

            @freeze_checkbox.on_update
            def _(_):
                self.renderer.config.freeze = freeze_checkbox.value

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
                    text="{}, {:.2f}, {}".format(text, distance, lod),
                    position=position,
                )

            return updater

        for idx, (id, xy, size) in enumerate(self.renderer.partition_coordinates):
            self.label_updaters.append(get_label_updater(
                name="part/{}".format(idx),
                text="#{}({},{})".format(idx, *id.tolist()),
                position=(torch.concat([xy + 0.5 * self.renderer.default_partition_size, torch.tensor([0.])]).cuda() @ self.renderer.orientation_transform.T).cpu().numpy(),
            ))

        self.renderer.on_render_hooks.append(self.update_labels)

    def draw_3d_boxes(self):
        from matplotlib.pyplot import cm
        colors = list(iter(cm.rainbow(np.linspace(0, 1, self.renderer.partition_full_3d_bounding_box.shape[0]))))
        random.shuffle(colors)

        for idx, box in enumerate(self.renderer.partition_full_3d_bounding_box):
            for cornerIdx, corner in enumerate(box):
                self.server.scene.add_icosphere(
                    name="box/{}-{}".format(idx, cornerIdx),
                    radius=0.01 * self.renderer.default_partition_size,
                    color=colors[idx][:3],
                    position=corner.cpu().numpy(),
                )
            # break

    def update_number_of_gaussians(self):
        self.markdown.content = "Gaussians: {}".format(
            self.renderer.gaussian_model.n_gaussians,
        )

    def update_labels(self):
        for idx, i in enumerate(self.label_updaters):
            i(self.renderer.partition_distances[idx].item(), self.renderer.partition_lods[idx].item())

import os
import math
import glob
import time
import json
import argparse
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
from internal.renderers import VanillaRenderer
from internal.models.gaussian_model_simplified import GaussianModelSimplified
from internal.models.simplified_gaussian_model_manager import SimplifiedGaussianModelManager
from internal.viewer import ClientThread, ViewerRenderer
from internal.utils.rotation import rotation_matrix

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"


class Viewer:
    def __init__(
            self,
            model_paths: list[str],
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
            enable_transform: bool = False,
            show_cameras: bool = False,
    ):
        self.device = torch.device("cuda")

        self.host = host
        self.port = port
        self.image_format = image_format

        self.sh_degree = sh_degree

        self.enable_transform = enable_transform

        self.show_cameras = show_cameras

        load_from = self._search_load_file(model_paths[0])

        # TODO: load multiple models more elegantly
        # load and create models
        model, renderer, training_output_base_dir, dataset_type = self._load_model_from_file(load_from)

        # reorient the scene
        cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)
        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)

        self.available_appearance_options = None

        self.loaded_model_count = 1
        addition_models = model_paths[1:]
        if len(addition_models) == 0:
            # load appearance groups
            appearance_group_filename = os.path.join(training_output_base_dir, "appearance_group_ids.json")
            if os.path.exists(appearance_group_filename) is True:
                with open(appearance_group_filename, "r") as f:
                    self.available_appearance_options = json.load(f)
            # self.available_appearance_options["@Disabled"] = None

            if enable_transform is True:
                model = SimplifiedGaussianModelManager(
                    [model.to_device(torch.device("cpu"))],
                    enable_transform=True,
                    device=self.device,
                )
        else:
            # switch to vanilla renderer
            model_list = [model.to_device(torch.device("cpu"))]
            renderer = VanillaRenderer()
            for model_path in addition_models:
                load_from = self._search_load_file(model_path)
                if load_from.endswith(".ckpt"):
                    load_results = self._do_initialize_models_from_checkpoint(load_from, sh_degree, device=torch.device("cpu"))
                else:
                    load_results = self._do_initialize_models_from_point_cloud(load_from, sh_degree, device=torch.device("cpu"))
                model_list.append(load_results[0])

            model = SimplifiedGaussianModelManager(model_list, enable_transform=enable_transform, device=self.device)

            self.loaded_model_count += len(addition_models)

        # create renderer
        self.viewer_renderer = ViewerRenderer(
            model,
            renderer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

    @staticmethod
    def _search_load_file(model_path: str) -> str:
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path) is False:
            return model_path
        # search checkpoint
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        # find checkpoint with max iterations
        load_from = None
        previous_checkpoint_iteration = -1
        for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
            try:
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue
            if checkpoint_iteration > previous_checkpoint_iteration:
                previous_checkpoint_iteration = checkpoint_iteration
                load_from = i

        # not a checkpoint can be found, search point cloud
        if load_from is None:
            previous_point_cloud_iteration = -1
            for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
                try:
                    point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
                except Exception as err:
                    print("error occurred when parsing iteration from {}: {}".format(i, err))
                    continue

                if point_cloud_iteration > previous_point_cloud_iteration:
                    load_from = os.path.join(i, "point_cloud.ply")

        assert load_from is not None, "not a checkpoint or point cloud can be found"

        return load_from

    def _reorient(self, cameras_json_path: str, mode: str, dataset_type: str = None):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))

        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        transform[:3, :3] = rotation
        transform = torch.linalg.inv(transform)

        return transform

    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)

    def add_cameras_to_scene(self, viser_server):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []

        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        for camera in self.camera_poses:
            name = camera["img_name"]
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)

            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=R.wxyz,
                position=c2w[:3, 3],
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = True

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        # def update_camera_scale(_):
        #     with viser_server.atomic():
        #         for i in self.camera_handles:
        #             i.scale = self.camera_scale_slider.value

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
            # self.camera_scale_slider = viser_server.add_gui_slider(
            #     "Camera Scale",
            #     min=0.,
            #     max=1.,
            #     step=0.01,
            #     initial_value=0.1,
            # )
        self.toggle_camera_button.on_click(toggle_camera_visibility)
        # self.camera_scale_slider.on_update(update_camera_scale)

    @staticmethod
    def _do_initialize_models_from_checkpoint(checkpoint_path: str, sh_degree, device):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]

        # initialize gaussian and renderer model
        model = GaussianModelSimplified.construct_from_state_dict(checkpoint["state_dict"], sh_degree, device)
        # extract state dict of renderer
        renderer = hparams["renderer"]
        renderer_state_dict = {}
        for i in checkpoint["state_dict"]:
            if i.startswith("renderer."):
                renderer_state_dict[i[len("renderer."):]] = checkpoint["state_dict"][i]
        # load state dict of renderer
        renderer.load_state_dict(renderer_state_dict)
        renderer = renderer.to(device)

        return model, renderer, checkpoint

    def _initialize_models_from_checkpoint(self, checkpoint_path: str):
        return self._do_initialize_models_from_checkpoint(checkpoint_path, self.sh_degree, self.device)

    @staticmethod
    def _do_initialize_models_from_point_cloud(point_cloud_path: str, sh_degree, device):
        model = GaussianModelSimplified.construct_from_ply(ply_path=point_cloud_path, active_sh_degree=sh_degree, device=device)
        renderer = VanillaRenderer()
        renderer.setup(stage="val")
        renderer = renderer.to(device)

        return model, renderer

    def _initialize_models_from_point_cloud(self, point_cloud_path: str):
        return self._do_initialize_models_from_point_cloud(point_cloud_path, self.sh_degree, self.device)

    def _load_model_from_file(self, load_from: str):
        print("load model from {}".format(load_from))
        dataset_type = None
        if load_from.endswith(".ckpt") is True:
            model, renderer, checkpoint = self._initialize_models_from_checkpoint(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(load_from))
            dataset_type = checkpoint["datamodule_hyper_parameters"]["type"]
        elif load_from.endswith(".ply") is True:
            model, renderer = self._initialize_models_from_point_cloud(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(load_from)))
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer, training_output_base_dir, dataset_type

    def start(self):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.configure_theme(control_layout="collapsible")
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        tabs = server.add_gui_tab_group()

        with tabs.add_tab("General"):
            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(server)

            # add render options
            with server.add_gui_folder("Render"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.add_gui_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                )
                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

            with server.add_gui_folder("Model"):
                self.scaling_modifier = server.add_gui_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                self.active_sh_degree_slider = server.add_gui_slider(
                    "Active SH Degree",
                    min=0,
                    max=self.viewer_renderer.gaussian_model.max_sh_degree,
                    step=1,
                    initial_value=self.viewer_renderer.gaussian_model.max_sh_degree,
                )
                self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                if self.available_appearance_options is not None:
                    # find max appearance id
                    max_input_id = 0
                    available_option_values = list(self.available_appearance_options.values())
                    if isinstance(available_option_values[0], list) or isinstance(available_option_values[0], tuple):
                        for i in available_option_values:
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    else:
                        # convert to tuple, compatible with previous version
                        for i in self.available_appearance_options:
                            self.available_appearance_options[i] = (0, self.available_appearance_options[i])
                    self.available_appearance_options[DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE] = None

                    self.appearance_id = server.add_gui_slider(
                        "Appearance Direct",
                        min=0,
                        max=max_input_id,
                        step=1,
                        initial_value=0,
                        visible=max_input_id > 0
                    )

                    self.normalized_appearance_id = server.add_gui_slider(
                        "Normalized Appearance Direct",
                        min=0.,
                        max=1.,
                        step=0.01,
                        initial_value=0.,
                    )

                    appearance_options = list(self.available_appearance_options.keys())

                    self.appearance_group_dropdown = server.add_gui_dropdown(
                        "Appearance Group",
                        options=appearance_options,
                        initial_value=appearance_options[0],
                    )
                    self.appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.normalized_appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.appearance_group_dropdown.on_update(self._handle_option_updated)

        def setup_transform_callback(
                idx: int,
                scale,
                rx,
                ry,
                rz,
                tx,
                ty,
                tz,
        ):
            def do_transform(_):
                with server.atomic():
                    self.viewer_renderer.gaussian_model.transform(
                        idx,
                        scale.value,
                        math.radians(rx.value),
                        math.radians(ry.value),
                        math.radians(rz.value),
                        tx.value,
                        ty.value,
                        tz.value,
                    )
                    self._handle_option_updated(_)

            scale.on_update(do_transform)
            rx.on_update(do_transform)
            ry.on_update(do_transform)
            rz.on_update(do_transform)
            tx.on_update(do_transform)
            ty.on_update(do_transform)
            tz.on_update(do_transform)

        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                for i in range(self.loaded_model_count):
                    with server.add_gui_folder("Model {} Transform".format(i)):
                        scale_slider = server.add_gui_slider(
                            "scale",
                            min=0.,
                            max=2.,
                            step=0.01,
                            initial_value=1.,
                        )
                        rx_slider = server.add_gui_slider(
                            "rx",
                            min=-180,
                            max=180,
                            step=1,
                            initial_value=0,
                        )
                        ry_slider = server.add_gui_slider(
                            "ry",
                            min=-180,
                            max=180,
                            step=1,
                            initial_value=0,
                        )
                        rz_slider = server.add_gui_slider(
                            "rz",
                            min=-180,
                            max=180,
                            step=1,
                            initial_value=0,
                        )
                        tx_slider = server.add_gui_slider(
                            "tx",
                            min=-10.,
                            max=10.,
                            step=0.01,
                            initial_value=0,
                        )
                        ty_slider = server.add_gui_slider(
                            "ty",
                            min=-10.,
                            max=10.,
                            step=0.01,
                            initial_value=0,
                        )
                        tz_slider = server.add_gui_slider(
                            "tz",
                            min=-10.,
                            max=10.,
                            step=0.01,
                            initial_value=0,
                        )

                        setup_transform_callback(
                            i,
                            scale_slider,
                            rx_slider,
                            ry_slider,
                            rz_slider,
                            tx_slider,
                            ty_slider,
                            tz_slider,
                        )

        while True:
            time.sleep(999)

    def _handle_appearance_embedding_slider_updated(self, _):
        self.appearance_group_dropdown.value = DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE
        self._handle_option_updated(_)

    def _handle_activate_sh_degree_slider_updated(self, _):
        self.viewer_renderer.gaussian_model.active_sh_degree = self.active_sh_degree_slider.value
        self._handle_option_updated(_)

    def get_appearance_id_value(self):
        if self.available_appearance_options is None:
            return (0, 0.)
        name = self.appearance_group_dropdown.value
        if name == DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE or name not in self.available_appearance_options:
            return (self.appearance_id.value, self.normalized_appearance_id.value)
        return self.available_appearance_options[name]

    def _handle_option_updated(self, _):
        for i in self.clients:
            try:
                self.clients[i].state = "low"
                self.clients[i].render_trigger.set()
            except:
                pass

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    args = parser.parse_args()

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)

    # start viewer server
    viewer.start()

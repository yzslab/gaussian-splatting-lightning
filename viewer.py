import os
import glob
import time
import json
from typing import Tuple, Literal
from jsonargparse import CLI
import viser
import torch
from internal.renderers import VanillaRenderer
from internal.models.gaussian_model_simplified import GaussianModelSimplified
from internal.viewer import ClientThread, ViewerRenderer
from internal.utils.rotation import rotation_matrix

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"


class Viewer:
    def __init__(
            self,
            model_path: str,
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
    ):
        self.device = torch.device("cuda")

        self.host = host
        self.port = port
        self.image_format = image_format

        self.sh_degree = sh_degree

        load_from = model_path
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path):
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

        # load and create models
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

        # reorient the scene
        cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)

        # load appearance groups
        self.available_appearance_options = {}
        appearance_group_filename = os.path.join(training_output_base_dir, "appearance_group_ids.json")
        if os.path.exists(appearance_group_filename) is True:
            with open(appearance_group_filename, "r") as f:
                self.available_appearance_options = json.load(f)

        self.available_appearance_options[DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE] = None
        # self.available_appearance_options["@Disabled"] = None

        # create renderer
        self.viewer_renderer = ViewerRenderer(
            model,
            renderer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

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

    def _initialize_models_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        hparams = checkpoint["hyper_parameters"]

        # initialize gaussian and renderer model
        model = GaussianModelSimplified.construct_from_state_dict(checkpoint["state_dict"], self.sh_degree, self.device)
        # extract state dict of renderer
        renderer = hparams["renderer"]
        renderer_state_dict = {}
        for i in checkpoint["state_dict"]:
            if i.startswith("renderer."):
                renderer_state_dict[i[len("renderer."):]] = checkpoint["state_dict"][i]
        # load state dict of renderer
        renderer.load_state_dict(renderer_state_dict)
        renderer = renderer.to(self.device)

        return model, renderer, checkpoint

    def _initialize_models_from_point_cloud(self, point_cloud_path: str):
        model = GaussianModelSimplified.construct_from_ply(ply_path=point_cloud_path, active_sh_degree=self.sh_degree, device=self.device)
        renderer = VanillaRenderer()
        renderer.setup(stage="val")
        renderer = renderer.to(self.device)

        return model, renderer

    def start(self):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

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
            self.appearance_embedding = server.add_gui_slider(
                "Appearance Direct",
                min=0.,
                max=1.,
                step=0.01,
                initial_value=0.,
            )
            appearance_options = list(self.available_appearance_options.keys())
            self.appearance_embedding_dropdown = server.add_gui_dropdown(
                "Appearance Group",
                options=appearance_options,
                initial_value=appearance_options[0],
            )
            self.scaling_modifier.on_update(self._handle_option_updated)
            self.appearance_embedding.on_update(self._handle_appearance_embedding_slider_updated)
            self.appearance_embedding_dropdown.on_update(self._handle_option_updated)

        while True:
            time.sleep(999)

    def _handle_appearance_embedding_slider_updated(self, _):
        self.appearance_embedding_dropdown.value = DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE
        self._handle_option_updated(_)

    def get_appearance_embedding_value(self):
        name = self.appearance_embedding_dropdown.value
        if name == DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE or name not in self.available_appearance_options:
            return self.appearance_embedding.value
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
    CLI(Viewer, set_defaults={
        "subcommand": "start",
    })

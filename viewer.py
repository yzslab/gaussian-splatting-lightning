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
            cameras_json_path = os.path.join(os.path.dirname(os.path.dirname(load_from)), "cameras.json")
            dataset_type = checkpoint["datamodule_hyper_parameters"]["type"]
        elif load_from.endswith(".ply") is True:
            model, renderer = self._initialize_models_from_point_cloud(load_from)
            cameras_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(load_from))), "cameras.json")
        else:
            raise ValueError("unsupported file {}".format(load_from))

        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)

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
        if dataset_type == "blender" and mode == "auto":
            print("skip reorient for blender dataset")
            return transform

        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))

        def rotation_matrix(a, b):
            """Compute the rotation matrix that rotates vector a to vector b.

            Args:
                a: The vector to rotate.
                b: The vector to rotate to.
            Returns:
                The rotation matrix.
            """
            a = a / torch.linalg.norm(a)
            b = b / torch.linalg.norm(b)
            v = torch.cross(a, b)
            c = torch.dot(a, b)
            # If vectors are exactly opposite, we add a little noise to one of them
            if c < -1 + 1e-8:
                eps = (torch.rand(3) - 0.5) * 0.01
                return rotation_matrix(a + eps, b)
            s = torch.linalg.norm(v)
            skew_sym_mat = torch.Tensor(
                [
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0],
                ]
            )
            return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))

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
        self.scaling_modifier.on_update(self._handle_option_updated)

        while True:
            time.sleep(999)

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

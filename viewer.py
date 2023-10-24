import os
import glob
import traceback
import threading
import numpy as np
import time
from typing import Tuple, Literal
from jsonargparse import CLI
import viser
import viser.transforms as vtf
import torch
from internal.models.gaussian_model_simplified import GaussianModelSimplified
from internal.cameras.cameras import Cameras
from internal.utils.graphics_utils import fov2focal
import internal.renderers as renderers

VISER_SCALE_RATIO: float = 10.0


class Renderer:
    def __init__(
            self,
            gaussian_model: GaussianModelSimplified,
            renderer: renderers.Renderer,
            background_color,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

    def get_outputs(self, camera):
        return self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
        )["render"]


class Client(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client

        self.render_trigger = threading.Event()

        self.last_move_time = 0

        self.last_camera = None  # store camera information

        self.state = "low"  # low or high render resolution

        self.stop_client = False  # whether stop this thread

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    def render_and_send(self):
        cam = self.last_camera

        self.last_move_time = time.time()

        # get camera pose
        R = vtf.SO3(wxyz=self.client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(self.client.camera.position, dtype=torch.float64) / VISER_SCALE_RATIO
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # calculate resolution
        aspect_ratio = cam.aspect
        max_res, jpeg_quality = self.get_render_options()
        image_height = max_res
        image_width = int(image_height * aspect_ratio)
        if image_width > max_res:
            image_width = max_res
            image_height = int(image_width / aspect_ratio)

        # construct camera
        fx = torch.tensor([fov2focal(cam.fov, image_width)], dtype=torch.float)
        camera = Cameras(
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            fx=fx,
            fy=fx,
            cx=torch.tensor([(image_width // 2)], dtype=torch.int),
            cy=torch.tensor([(image_height // 2)], dtype=torch.int),
            width=torch.tensor([image_width], dtype=torch.int),
            height=torch.tensor([image_height], dtype=torch.int),
            appearance_embedding=torch.tensor([0]),
            distortion_params=None,
            camera_type=torch.tensor([0], dtype=torch.int),
        )[0].to_device(self.viewer.device)

        with torch.no_grad():
            image = self.renderer.get_outputs(camera)
            image = torch.clamp(image, max=1.)
            image = torch.permute(image, (1, 2, 0))
            self.client.set_background_image(
                image.cpu().numpy(),
                format=self.viewer.image_format,
                jpeg_quality=jpeg_quality,
            )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:  # TODO: avoid wasting CPU
                # if we haven't received a trigger in a while, switch to high resolution

                # skip if camera is none
                if self.last_camera is None:
                    continue

                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None


class Viewer:
    def __init__(
            self,
            model_path: str,
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
    ):
        self.device = torch.device("cuda")

        self.host = host
        self.port = port
        self.image_format = image_format

        # load checkpoint and create models
        ckpt_path = model_path
        if ckpt_path.endswith(".ckpt") is False:
            # find checkpoint with max iterations
            checkpoint_dir = os.path.join(ckpt_path, "checkpoints")

            previous_checkpoint_iteration = -1
            for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
                if checkpoint_iteration > previous_checkpoint_iteration:
                    previous_checkpoint_iteration = checkpoint_iteration
                    ckpt_path = i
            print("auto select checkpoint {}".format(ckpt_path))

        self.ckpt = torch.load(ckpt_path)
        self._initialize_models()

        # create renderer
        self.renderer = Renderer(
            self.model,
            self.renderer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

    def _initialize_models(self):
        self.hparams = self.ckpt["hyper_parameters"]

        # initialize gaussian and renderer model
        self.model = GaussianModelSimplified.construct_from_state_dict(self.ckpt["state_dict"], 3, self.device)
        # extract state dict of renderer
        self.renderer = self.hparams["renderer"]
        renderer_state_dict = {}
        for i in self.ckpt["state_dict"]:
            if i.startswith("renderer."):
                renderer_state_dict[i[len("renderer."):]] = self.ckpt["state_dict"][i]
        # load state dict of renderer
        self.renderer.load_state_dict(renderer_state_dict)
        self.renderer = self.renderer.to(self.device)

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
            self.jpeg_quality_when_static = server.add_gui_slider(
                "JPEG Quality",
                min=0,
                max=100,
                step=1,
                initial_value=100,
            )
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

        while True:
            time.sleep(999)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        # create client thread
        client_thread = Client(self, self.renderer, client)
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

import queue
import traceback

import numpy as np
import torch
import viser

import viewer
from queue import Queue


class MockGaussianModel:
    max_sh_degree: int = 3
    active_sh_degree: int = 3


class TrainingViewerRenderer:
    def __init__(self, camera_queue: Queue, renderer_output_queue: Queue):
        self.camera_queue = camera_queue
        self.renderer_output_queue = renderer_output_queue
        self.gaussian_model = MockGaussianModel

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        # TODO: support multiple client
        # if multiple clients connected, they may get images mismatch to their camera poses,
        # but I think fix it is unnecessary
        self.camera_queue.put((camera, scaling_modifier))
        return self.renderer_output_queue.get()


# TODO: refactoring the the viewer
class TrainingViewer(viewer.Viewer):
    def __init__(
            self,
            up_direction: np.ndarray = None,
            camera_center: np.ndarray = None,
            host: str = "0.0.0.0",
            port: int = 8080,
    ):
        self.host = host
        self.port = port
        self.image_format = "jpeg"
        self.enable_transform = False
        self.show_cameras = False
        self.show_edit_panel = False
        self.show_render_panel = False
        self.available_appearance_options = None
        self.default_camera_position = None
        self.default_camera_look_at = None
        self.camera_transform = torch.eye(4)
        self.device = torch.device("cpu")

        self.up_direction = up_direction
        self.camera_center = camera_center

        self.camera_queue = Queue()
        self.renderer_output_queue = Queue()
        self.viewer_renderer = TrainingViewerRenderer(self.camera_queue, self.renderer_output_queue)

        self.clients = {}

        self.is_training_paused = False

    def setup_training_panel(self, viewer, server: viser.ViserServer):
        self.pause_training_button = server.add_gui_button("Pause Training", icon=viser.Icon.PLAYER_PAUSE_FILLED)

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training_button.visible = False
            self.resume_training_button.visible = True
            self.is_training_paused = True

        self.resume_training_button = server.add_gui_button("Resume Training", icon=viser.Icon.PLAYER_PLAY_FILLED, visible=False)

        @self.resume_training_button.on_click
        def _(event):
            self.pause_training_button.visible = True
            self.resume_training_button.visible = False
            # mark training resumed
            self.is_training_paused = False
            # send camera to wake blocking thread
            self.camera_queue.put((event.client.camera, self.scaling_modifier.value))

        self.global_step_label = server.add_gui_markdown(content="Step: 0")

        self.render_frequency_slider = server.add_gui_slider("Render Freq", initial_value=10, min=1, max=100, step=1)

    def start(self):
        super().start(False, server_config_fun=self.setup_training_panel)

    def process_all_render_requests(self, gaussian_model, renderer, background_color):
        device = gaussian_model.get_xyz.device
        while True:
            try:
                if self.is_training_paused is True:
                    client_camera, scaling_modifier = self.camera_queue.get()
                else:
                    client_camera, scaling_modifier = self.camera_queue.get_nowait()
            except queue.Empty:
                break

            try:
                with torch.no_grad():
                    self.renderer_output_queue.put(renderer(
                        client_camera.to_device(device),
                        gaussian_model,
                        bg_color=background_color.to(device),
                        scaling_modifier=scaling_modifier,
                    )["render"])
            except:
                traceback.print_exc()

    def training_step(self, gaussian_model, renderer, background_color, step: int):
        self.global_step_label.content = f"Step: {step}"

        if self.is_training_paused is False:
            if self.camera_queue.empty() is True:
                if step % int(self.render_frequency_slider.value) == 0:
                    self.rerender_for_all_client()
                else:
                    return

        self.process_all_render_requests(gaussian_model, renderer, background_color)

    def validation_step(self, gaussian_model, renderer, background_color, step: int):
        if self.camera_queue.empty() is True:
            return
        self.process_all_render_requests(gaussian_model, renderer, background_color)

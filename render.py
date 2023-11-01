import os
import subprocess
import argparse
import json
import torch
import torchvision
from tqdm import tqdm
from internal.cameras.cameras import Cameras
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.models.simplified_gaussian_model_manager import SimplifiedGaussianModelManager
from internal.viewer.renderer import ViewerRenderer


def initializer_viewer_renderer(
        model_paths: list[str],
        enable_transform: bool,
        sh_degree: int,
        background_color,
        device,
) -> ViewerRenderer:
    model_list = []
    renderer = None

    load_device = torch.device("cuda") if len(model_paths) == 1 or enable_transform is False else torch.device("cpu")
    for model_path in model_paths:
        model, renderer = GaussianModelLoader.search_and_load(model_path, sh_degree, load_device)
        model_list.append(model)

    if len(model_paths) > 1:
        renderer = VanillaRenderer()

    model_manager = SimplifiedGaussianModelManager(model_list, enable_transform, device)

    return ViewerRenderer(model_manager, renderer, torch.tensor(background_color, dtype=torch.float, device=device))


def parse_camera_poses(camera_path: dict):
    orientation_transform = torch.tensor(camera_path["orientation_transform"], dtype=torch.float)

    c2w_flatten_list = []
    fov_list = []
    aspect_list = []

    for camera in camera_path["camera_path"]:
        c2w_flatten_list.append(camera["camera_to_world"])
        fov_list.append(camera["fov"])
        aspect_list.append(camera["aspect"])

    width = torch.tensor([int(camera_path["render_width"])], dtype=torch.int16).expand(len(c2w_flatten_list))
    height = torch.tensor([int(camera_path["render_height"])], dtype=torch.int16).expand(len(c2w_flatten_list))

    c2w = torch.reshape(torch.tensor(c2w_flatten_list, dtype=torch.float), (-1, 4, 4))
    c2w = torch.matmul(orientation_transform, c2w)
    c2w[..., :3, 1:3] *= -1
    w2c = torch.linalg.inv(c2w)
    fov = torch.tensor(fov_list, dtype=torch.float)
    fx = width / (2 * torch.atan(fov / 2))
    fy = fx

    return Cameras(
        R=w2c[..., :3, :3],
        T=w2c[..., :3, 3],
        fx=fx,
        fy=fy,
        cx=width / 2.,
        cy=height / 2.,
        width=width,
        height=height,
        appearance_id=torch.zeros_like(fx),
        normalized_appearance_id=torch.zeros_like(fx),
        distortion_params=None,
        camera_type=torch.zeros_like(fx),
    )


def render_frames(cameras: Cameras, viewer_renderer: ViewerRenderer, output_path: str, device):
    os.makedirs(output_path, exist_ok=True)
    for idx in tqdm(range(len(cameras)), desc="rendering frames"):
        camera = cameras[idx].to_device(device)
        image = viewer_renderer.get_outputs(camera).cpu()
        torchvision.utils.save_image(image, os.path.join(output_path, "{:06d}.png".format(idx)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("--camera-path-filename", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")

    with open(args.camera_path_filename, "r") as f:
        camera_path = json.load(f)

    renderer = initializer_viewer_renderer(
        args.model_paths,
        enable_transform=camera_path["enable_transform"],
        sh_degree=camera_path["sh_degree"],
        background_color=camera_path["background_color"],
        device=device,
    )

    cameras = parse_camera_poses(camera_path)

    frame_output_path = args.output_path + "_frames"
    with torch.no_grad():
        render_frames(cameras, viewer_renderer=renderer, output_path=frame_output_path, device=device)

    subprocess.call([
        "ffmpeg",
        "-y",
        "-framerate", str(camera_path["fps"]),
        "-i", os.path.join(frame_output_path, "%06d.png"),
        "-pix_fmt", "yuv420p",
        args.output_path,
    ])

    subprocess.call(["stty", "sane"])

import argparse
import time
import viser
import viser.transforms as vtf
import json
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", default="cameras.json", required=True, type=str)
parser.add_argument("--points", default=None, type=str)
parser.add_argument("--up", nargs="+", required=False, type=float, default=None)
args = parser.parse_args()

with open(args.cameras, "r") as f:
    camera_poses = json.load(f)

viser_server = viser.ViserServer()

camera_transform = torch.eye(4, dtype=torch.float)

camera_pose_transform = np.linalg.inv(camera_transform.cpu().numpy())
up = torch.zeros(3)
for camera in camera_poses:
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
        scale=0.02,
        aspect=float(cx / cy),
        wxyz=R.wxyz,
        position=c2w[:3, 3],
        color=(255, 0, 0),
    )

    up += torch.tensor(camera["rotation"])[:3, 1]

if args.up is not None:
    up = torch.tensor(args.up)
print("up vector = {}".format(up))
up = -up / torch.linalg.norm(up)

if args.points is not None:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(args.points)
    viser_server.add_point_cloud(
        "points",
        np.asarray(pcd.points),
        (np.asarray(pcd.colors) * 255).astype(np.uint8),
        point_size=0.005,
    )

reset_up_button = viser_server.add_gui_button(
    "Reset up direction",
    icon=viser.Icon.ARROW_AUTOFIT_UP,
    hint="Reset the orbit up direction.",
)


@reset_up_button.on_click
def _(event: viser.GuiEvent) -> None:
    assert event.client is not None
    event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])


viser_server.set_up_direction(up)


while True:
    time.sleep(1 << 15)

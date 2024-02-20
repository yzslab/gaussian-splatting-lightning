import os
import argparse
import math
import json
import random
import numpy as np
import cv2
import open3d as o3d

from tqdm import tqdm

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+",
                        help="list of the path of the transforms.json")
    parser.add_argument("--output", type=str, required=True,
                        help="path to the output ply file")
    parser.add_argument("--rgb", type=str, default="rgb",
                        help="rgb image dirname")
    parser.add_argument("--scale", type=float, default=1.,
                        help="scene scale factor")
    parser.add_argument("--depth-scale", type=float, default=0.01,
                        help="depth scale factor")
    parser.add_argument("--max-points", type=int, default=20_480_000)
    parser.add_argument("--max", type=int, default=-1,
                        help="max read frames")
    parser.add_argument("--max-depth", type=float, default=65_000,
                        help="max valid depth value")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return args


def read_depth(path: str, scale: float):
    return cv2.imread(
        path,
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
    )[..., 0] * scale


def read_rgb(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_point_cloud(frames: list[dict], max_point_per_image: int, max_depth: float, scene_scale: float, depth_scale: float, rgb_dirname: str):
    xyz_array_list = []
    rgb_array_list = []

    final_depth_scale = scene_scale * depth_scale
    final_max_depth = max_depth * final_depth_scale

    for frame in tqdm(frames):
        # build intrinsics matrix
        fov_x = frame["camera_angle_x"]
        rgb = read_rgb(os.path.join(frame["base_dir"], rgb_dirname, "{:04d}.png".format(frame["frame_index"])))
        image_shape = rgb.shape
        height, width = image_shape[0], image_shape[1]
        fx = float(.5 * width / np.tan(.5 * fov_x))
        fy = fx
        cx = width / 2
        cy = height / 2
        K = np.eye(3)
        K[0, 2] = cx
        K[1, 2] = cy
        K[0, 0] = fx
        K[1, 1] = fy

        # build pixel coordination
        image_pixel_count = height * width
        u_coord = np.tile(np.arange(width), (height, 1)).reshape(image_pixel_count)
        v_coord = np.tile(np.arange(height), (width, 1)).T.reshape(image_pixel_count)
        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)]).T
        homogenous_coordinate = np.matmul(p2d, np.linalg.inv(K).T)

        # read rgb and depth
        rgb = rgb.reshape((-1, 3))
        depth = read_depth(
            os.path.join(frame["base_dir"], "depth", "{:04d}.exr".format(frame["frame_index"])),
            final_depth_scale,
        ).reshape((-1,))

        # discard invalid depth
        valid_depth_indices = np.where(depth < final_max_depth)
        rgb = rgb[valid_depth_indices]
        depth = depth[valid_depth_indices]
        homogenous_coordinate = homogenous_coordinate[valid_depth_indices]

        # random sample
        valid_pixel_count = rgb.shape[0]
        if max_point_per_image < valid_pixel_count:
            sample_indices = np.random.choice(valid_pixel_count, max_point_per_image, replace=False)
            homogenous_coordinate = homogenous_coordinate[sample_indices]
            rgb = rgb[sample_indices]
            depth = depth[sample_indices]

        # build camera-to-world transform matrix
        rot_mat = np.asarray(frame["rot_mat"], dtype=np.float32)
        rot_mat[:3, :3] *= 100
        if scene_scale != 1.:
            rot_mat[:3, 3] *= scene_scale
        c2w = rot_mat

        # convert to world coordination
        points_3d_in_camera = homogenous_coordinate * depth[:, None]
        """
        convert to right-handed coordinates
        see:
            https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L137
            https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html
        """
        points_3d_in_camera[:, 1] *= -1
        points_3d_in_camera[:, 2] *= -1
        points_3d_in_world = np.matmul(points_3d_in_camera, c2w[:3, :3].T) + c2w[:3, 3]

        xyz_array_list.append(points_3d_in_world)
        rgb_array_list.append(rgb)

    return np.concatenate(xyz_array_list, axis=0), np.concatenate(rgb_array_list, axis=0)


def merge_frames(paths: list[str]) -> list:
    frame_list = []
    for i in paths:
        with open(i, "r") as f:
            transforms = json.load(f)
        base_dir = os.path.dirname(i)
        fov_x = transforms["camera_angle_x"]
        for frame in transforms["frames"]:
            frame["base_dir"] = base_dir
            frame["camera_angle_x"] = fov_x
            frame_list.append(frame)

    return frame_list


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    frames = merge_frames(args.input)

    # select frames
    frames = frames[::args.step]
    if args.max > 0:
        frames = frames[:args.max]

    # calculate max point per frame
    max_point_per_image = -1
    if args.max_points > 0:
        max_point_per_image = math.ceil(args.max_points / len(frames))

    xyz, rgb = build_point_cloud(
        frames,
        max_point_per_image=max_point_per_image,
        max_depth=args.max_depth,
        scene_scale=args.scale,
        depth_scale=args.depth_scale,
        rgb_dirname=args.rgb,
    )
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(xyz)
    final_pcd.colors = o3d.utility.Vector3dVector(rgb / 255.)
    o3d.io.write_point_cloud(args.output, final_pcd)


if __name__ == "__main__":
    main()

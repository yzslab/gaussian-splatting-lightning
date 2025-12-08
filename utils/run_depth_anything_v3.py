import add_pypath

import os
import sys
import numpy as np
import argparse

from tqdm import tqdm
from common import AsyncImageSaver
from internal.utils.colmap import read_images_binary, read_cameras_binary
from distibuted_tasks import configure_arg_parser, get_task_list_with_args, clear_slurm_env

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("--image_dir", type=str, default="images")
parser.add_argument("--input_size", "-s", type=int, default=504)
parser.add_argument("--name", "-n", default="da3")
parser.add_argument("--da3_path", type=str, default=os.path.join(os.path.dirname(__file__), "Depth-Anything-3"))
parser.add_argument("--batch_size", "-b", type=int, default=32)
configure_arg_parser(parser)
args = parser.parse_args()

# set output dir
base_output_dir_prefix = os.path.join(args.dataset_dir, args.name)
depth_output_dir = "{}_depths".format(base_output_dir_prefix)
inverse_depth_output_dir = "{}_inverse_depths".format(base_output_dir_prefix)
os.makedirs(depth_output_dir, exist_ok=True)
os.makedirs(inverse_depth_output_dir, exist_ok=True)

# load colmap sparse
print("loading colmap sparse model...")
colmap_cameras = read_cameras_binary(os.path.join(args.dataset_dir, "sparse", "cameras.bin"))
colmap_images = read_images_binary(os.path.join(args.dataset_dir, "sparse", "images.bin"))
image_id_list = sorted(list(colmap_images.keys()))
n_images = len(image_id_list)

# group image by camera for batching
image_id_list_group_by_camera_id = {}
for idx in image_id_list:
    camera_id = colmap_images[idx].camera_id
    image_id_list_group_by_camera_id.setdefault(camera_id, []).append(idx)

n_assigned_images = 0
image_id_list_slice_group_by_camera_id = {}
for camera_id, camera_image_id_list in image_id_list_group_by_camera_id.items():
    image_id_list_slice_group_by_camera_id[camera_id] = get_task_list_with_args(args, all_tasks=camera_image_id_list, clear_slurm_env=False)
    n_assigned_images += len(image_id_list_slice_group_by_camera_id[camera_id])
try:
    clear_slurm_env()
except:
    pass
del image_id_list_group_by_camera_id

print("{}/{} added to list".format(n_assigned_images, n_images))
is_single_task_mode = n_assigned_images == n_images


def get_image_data(image_id_list):
    image_names = []
    image_files = []
    extrinsics = []
    intrinsics = []

    for image_id in image_id_list:
        image_data = colmap_images[image_id]
        image_name = image_data.name
        image_path = os.path.join(os.path.join(args.dataset_dir, args.image_dir), image_name)

        image_names.append(image_name)
        image_files.append(image_path)

        # Get camera parameters
        camera = colmap_cameras[image_data.camera_id]

        # Convert quaternion to rotation matrix
        R = image_data.qvec2rotmat()
        t = image_data.tvec

        # Create extrinsic matrix (world to camera)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        extrinsics.append(extrinsic)

        # Create intrinsic matrix
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        else:
            # For other models, use basic pinhole approximation
            fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
            cx = camera.width / 2
            cy = camera.height / 2

        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    return image_names, image_files, extrinsics, intrinsics


image_data_group_by_camera_id = {}
for camera_id, camera_image_id_list in image_id_list_slice_group_by_camera_id.items():
    image_data_group_by_camera_id[camera_id] = get_image_data(camera_image_id_list)


# load Depth Anything
sys.path.insert(0, args.da3_path)
import torch
from depth_anything_3.api import DepthAnything3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# saver


def store_depth_and_set_scale(depth, image_name, output_dir, image_saver):
    max_depth = depth.max()

    depth_uint16 = np.clip((depth / max_depth) * 65535., a_min=0, a_max=65535).astype(np.uint16)

    scale_output_path = os.path.join(output_dir, "{}.scale.npy".format(image_name))
    os.makedirs(os.path.dirname(scale_output_path), exist_ok=True)
    np.save(scale_output_path, max_depth)
    image_saver.save(depth_uint16, os.path.join(output_dir, "{}.uint16.png".format(image_name)))


batch_size = args.batch_size

# start inferring
depth_saver = AsyncImageSaver()
t = tqdm(total=n_assigned_images)
try:
    for image_names, image_files, extrinsics, intrinsics in image_data_group_by_camera_id.values():
        for i in range(0, len(image_names), batch_size):
            slice_end = i + batch_size
            prediction = None

            image_file_slice = image_files[i:slice_end]
            try:
                prediction = model.inference(
                    image=image_file_slice,
                    extrinsics=extrinsics[i:slice_end],  # (N, 4, 4)
                    intrinsics=intrinsics[i:slice_end],   # (N, 3, 3)
                    align_to_input_ext_scale=True,
                    infer_gs=False,
                    process_res=args.input_size,
                )
            except:
                import traceback
                traceback.print_exc()
                print("batch skipped")
                continue
            finally:
                t.update(len(image_file_slice))

            for idx, image_name in zip(range(prediction.depth.shape[0]), image_names[i:slice_end]):
                depth = prediction.depth[idx]
                inverse_depth = 1. / depth
                inverse_depth = np.nan_to_num(inverse_depth, copy=False, nan=0., posinf=0., neginf=0.)

                store_depth_and_set_scale(
                    depth=depth,
                    image_name=image_name,
                    output_dir=depth_output_dir,
                    image_saver=depth_saver,
                )
                store_depth_and_set_scale(
                    depth=inverse_depth,
                    image_name=image_name,
                    output_dir=inverse_depth_output_dir,
                    image_saver=depth_saver,
                )
finally:
    t.close()
    depth_saver.stop()

if is_single_task_mode:
    import subprocess
    subprocess.call([
        "python",
        os.path.join(os.path.dirname(__file__), "get_da3_depth_scales.py"),
        args.dataset_dir,
        "--name={}".format(os.path.basename(args.name)),
    ])
else:
    print("To generate scales in JSON format:")
    print("python utils/get_da3_depth_scales.py {} --name={}".format(args.dataset_dir, args.name))

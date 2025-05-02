import add_pypath
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from internal.utils.visualizers import Visualizers
from common import find_files, AsyncNDArraySaver, AsyncImageSaver, AsyncImageReader
from distibuted_tasks import configure_arg_parser, get_task_list_with_args
from utils.distibuted_tasks import get_task_list_with_args

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--input_size", "-s", type=int, default=518)
parser.add_argument("--output", "-o", default=None)
parser.add_argument("--encoder", default="vitl")
parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
parser.add_argument("--preview", "-p", action="store_true", default=False)
parser.add_argument("--colormap", type=str, default="default")
parser.add_argument("--da2_path", type=str, default=os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
parser.add_argument("--image_list", type=str, default=None)
parser.add_argument("--uint16", action="store_true", default=False)
configure_arg_parser(parser)
args = parser.parse_args()

sys.path.insert(0, args.da2_path)
from depth_anything_v2.dpt import DepthAnythingV2

if args.output is None:
    args.output = os.path.join(os.path.dirname(args.image_dir), "estimated_depths")

found_image_list = find_files(args.image_dir, args.extensions, as_relative_path=False)
if args.image_list is not None:
    provided_image_list = {}
    with open(args.image_list, "r") as f:
        for row in f:
            provided_image_list[row.strip("\n")] = True

    valid_image_list = []
    for i in found_image_list:
        if i[len(args.image_dir):].lstrip("/") not in provided_image_list:
            continue
        valid_image_list.append(i)
    found_image_list = valid_image_list

images = get_task_list_with_args(args, found_image_list)
assert len(images) > 0, "not an image with extension name '{}' can be found in '{}'".format(args.extensions, args.image_dir)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

depth_anything = DepthAnythingV2(**model_configs[args.encoder])
depth_anything.load_state_dict(torch.load(f'{args.da2_path}/checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()


def apply_color_map(normalized_depth):
    colored_depth = Visualizers.float_colormap(torch.from_numpy(normalized_depth).unsqueeze(0), colormap=args.colormap)
    colored_depth = (colored_depth.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return colored_depth


if args.uint16:
    depth_saver = AsyncImageSaver()

    def save_depth(depth, image_name):
        depth = np.clip((depth * 65535) + 0.5, a_min=0., a_max=65535.).astype(np.uint16)
        output_filename = os.path.join(args.output, "{}.uint16.png".format(image_name))
        depth_saver.save(depth, output_filename)
else:
    depth_saver = AsyncNDArraySaver()

    def save_depth(depth, image_name):
        output_filename = os.path.join(args.output, "{}.npy".format(image_name))
        depth_saver.save(depth, output_filename)

image_reader = AsyncImageReader(image_list=images)
image_saver = AsyncImageSaver(is_rgb=True)
try:
    with torch.no_grad(), tqdm(range(len(images))) as t:
        for _ in t:
            image_path, raw_image = image_reader.get()
            image_name = image_path[len(args.image_dir):].lstrip(os.path.sep)

            depth = depth_anything.infer_image(raw_image, args.input_size)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

            save_depth(normalized_depth, image_name)

            if args.preview is True:
                image_saver.save(normalized_depth, os.path.join(args.output, "{}.png".format(image_name)), processor=apply_color_map)
finally:
    depth_saver.stop()
    image_reader.stop()
    image_saver.stop()

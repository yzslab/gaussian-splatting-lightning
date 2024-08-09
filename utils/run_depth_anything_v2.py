import os
import sys
import numpy as np
import argparse
import torch
import cv2

from common import find_files
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--input_size", "-s", type=int, default=518)
parser.add_argument("--output", "-o", default=None)
parser.add_argument("--encoder", default="vitl")
parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG"])
parser.add_argument("--preview", "-p", action="store_true", default=False)
parser.add_argument("--da2_path", type=str, default=os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
args = parser.parse_args()

sys.path.insert(0, args.da2_path)
from depth_anything_v2.dpt import DepthAnythingV2

if args.output is None:
    args.output = os.path.join(os.path.dirname(args.image_dir), "estimated_depths")

images = find_files(args.image_dir, args.extensions)

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

with torch.no_grad(), tqdm(images) as t:
    for image_name in t:
        raw_image = cv2.imread(os.path.join(args.image_dir, image_name))

        depth = depth_anything.infer_image(raw_image, args.input_size)
        normalized_depth = (depth - depth.min()) / (depth.max() - depth.min())

        output_filename = os.path.join(args.output, "{}.npy".format(image_name))
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.save(output_filename, normalized_depth)

        if args.preview is True:
            cv2.imwrite(os.path.join(args.output, "{}.png".format(image_name)), (normalized_depth * 255)[..., np.newaxis].astype(np.uint8))

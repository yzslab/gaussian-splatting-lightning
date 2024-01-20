"""
For colmap dataset
"""

import add_pypath
import os
import json
import argparse
from tqdm import tqdm
from internal.utils.colmap import read_images_binary

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--dirname", action="store_true", default=False,
                    help="Share same appearance group for every directory")
parser.add_argument("--camera", action="store_true", default=False,
                    help="Share same appearance group for every camera")
parser.add_argument("--image", action="store_true", default=False,
                    help="Every image has different appearance")
parser.add_argument("--name", type=str, default=None,
                    help="output filename without extension")
args = parser.parse_args()

images_bin_path = os.path.join(args.dir, "sparse", "images.bin")
if os.path.exists(images_bin_path) is False:
    images_bin_path = os.path.join(args.dir, "sparse", "0", "images.bin")

print("reading {}".format(images_bin_path))
images = read_images_binary(images_bin_path)
image_group = {}
for i in tqdm(images, desc="reading image information"):
    image = images[i]

    if args.dirname is True:
        key = os.path.dirname(image.name)
    elif args.camera is True:
        key = image.camera_id
    elif args.image is True:
        key = image.name
    else:
        raise ValueError("unsupported group type")

    if key not in image_group:
        image_group[key] = []
    image_group[key].append(image.name)

for i in image_group:
    image_group[i].sort()

save_path = os.path.join(args.dir, "appearance_groups.json" if args.name is None else "{}.json".format(args.name))
with open(save_path, "w") as f:
    json.dump(image_group, f, indent=4, ensure_ascii=False)
print(save_path)

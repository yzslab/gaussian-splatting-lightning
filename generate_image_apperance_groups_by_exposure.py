"""
For HDR-NeRF dataset: https://xhuangcv.github.io/hdr-nerf/
"""

import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--exposure", type=str, required=True,
                    help="path to exposure json file")
parser.add_argument("--name", type=str, default=None,
                    help="output filename without extension")
args = parser.parse_args()

with open(args.exposure, "r") as f:
    exposure_key_by_image_name = json.load(f)

image_group_by_exposure = {}
for image_name in exposure_key_by_image_name:
    exposure = exposure_key_by_image_name[image_name]
    if exposure not in image_group_by_exposure:
        image_group_by_exposure[exposure] = []
    image_group_by_exposure[exposure].append(image_name)

save_path = os.path.join(args.dir, "appearance_groups.json" if args.name is None else "{}.json".format(args.name))
with open(save_path, "w") as f:
    json.dump(image_group_by_exposure, f, indent=4, ensure_ascii=False)
print(save_path)

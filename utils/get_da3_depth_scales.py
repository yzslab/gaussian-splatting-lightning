import os
import numpy as np
import json
from glob import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("--name", "-n", type=str, default="da3")
args = parser.parse_args()

args.output = os.path.join(args.dataset_dir, args.name).rstrip(os.path.sep)
depth_output_dir = "{}_depths".format(args.output)
inverse_depth_output_dir = "{}_inverse_depths".format(args.output)

SUFFIX = ".scale.npy"
SUFFIX_LEN = len(SUFFIX)


def convert(dir: str):
    cwd = os.getcwd()

    scales = {}
    try:
        os.chdir(dir)
        for i in sorted(glob(os.path.join("**", "*{}".format(SUFFIX)), recursive=True)):
            image_name = i[:-SUFFIX_LEN]
            scales[image_name] = {
                "scale": np.load(i).item(),
                "offset": 0,
            }
    finally:
        os.chdir(cwd)

    json_path = "{}-scales.json".format(dir.rstrip(os.path.sep))
    with open(json_path, "w") as f:
        json.dump(scales, f, indent=4, ensure_ascii=False)
    print("{}-scales.json".format(dir.rstrip(os.path.sep)))


convert(depth_output_dir)
convert(inverse_depth_output_dir)

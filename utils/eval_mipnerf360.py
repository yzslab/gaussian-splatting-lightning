import os
import subprocess
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--config", "-c", default=None)
parser.add_argument("--project", "-p", default="MipNeRF360")
args = parser.parse_args()

# find scenes
scenes = []
for i in list(os.listdir(args.path)):
    if os.path.isdir(os.path.join(args.path, i, "sparse")) is False:
        continue
    scenes.append(i)
print(scenes)


def start(command: str, scene: str, extra_args: list = None):
    arg_list = [
        "python",
        "main.py",
        command,
        "--data.path", os.path.join(args.path, scene),
        "--data.parser", "Colmap",
        "--data.parser.down_sample_factor", "4",
        "--data.parser.split_mode", "experiment",
        "--data.parser.down_sample_rounding_model", "round_half_up",
        "--cache_all_images",
        "--logger", "wandb",
        "--output", os.path.join("outputs", args.project),
        "--project", args.project,
        "-n", scene,
    ]
    if args.config is not None:
        arg_list += ["--config", args.config]
    if extra_args is not None:
        arg_list += extra_args

    return subprocess.call(arg_list)


with tqdm(scenes) as t:
    for i in t:
        t.set_description(i)
        start("fit", i)
        start("validate", i, extra_args=["--save_val"])

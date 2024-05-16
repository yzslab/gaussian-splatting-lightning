import add_pypath
import os
import argparse
import lightning
import torch
from internal.utils.gaussian_utils import Gaussian

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output", "-o", required=False, default=None)
parser.add_argument("--colored", "-c", action="store_true", default=False)
args = parser.parse_args()

if args.output is None:
    args.output = args.input + ".ply"
assert os.path.exists(args.output) is False, "File exists at output path"
# assert args.input != args.output

ckpt = torch.load(args.input)
model = Gaussian.load_from_state_dict(ckpt["hyper_parameters"]["gaussian"].sh_degree, ckpt["state_dict"]).to_ply_format().save_to_ply(args.output, args.colored)
print(f"Saved to '{args.output}'")

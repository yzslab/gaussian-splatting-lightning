import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the model output directory")
args = parser.parse_args()

checkpoint_dir = os.path.join(args.path, "checkpoints")

# find max iteration
print("Searching checkpoint files...")
max_iteration = -1
checkpoint_files = []
for i in os.listdir(checkpoint_dir):
    if i.endswith(".ckpt") is False:
        continue
    try:
        step = int(i[i.index("step=") + 5:i.rindex("-rank=")])
        if step > max_iteration:
            max_iteration = step
            checkpoint_files = []
        if step == max_iteration:
            checkpoint_files.append(i)
    except:
        pass

assert len(checkpoint_files) > 0

print(checkpoint_files)

import add_pypath
import torch

param_list_key_by_name = {}
extra_param_list_key_by_name = {}
optimizer_state_exp_avg_list_key_by_index = {}
optimizer_state_exp_avg_sq_list_key_by_index = {}
for i in tqdm(checkpoint_files, desc="Loading checkpoints"):
    ckpt = torch.load(os.path.join(checkpoint_dir, i), map_location="cpu")
    for i in ckpt["state_dict"]:
        if i.startswith("gaussian_model."):
            param_list_key_by_name.setdefault(i, []).append(ckpt["state_dict"][i])
    for i in ckpt["gaussian_model_extra_state_dict"]:
        if isinstance(ckpt["gaussian_model_extra_state_dict"][i], torch.Tensor) is False:
            continue
        if ckpt["gaussian_model_extra_state_dict"][i].dim() == 0:
            continue
        extra_param_list_key_by_name.setdefault(i, []).append(ckpt["gaussian_model_extra_state_dict"][i])

    assert len(ckpt["optimizer_states"]) == 1
    for i in ckpt["optimizer_states"][0]["state"]:
        optimizer_state_exp_avg_list_key_by_index.setdefault(i, []).append(ckpt["optimizer_states"][0]["state"][i]["exp_avg"])
        optimizer_state_exp_avg_sq_list_key_by_index.setdefault(i, []).append(ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"])

print("Merging Gaussians...")
for i in param_list_key_by_name:
    ckpt["state_dict"][i] = torch.concat(param_list_key_by_name[i], dim=0)
for i in extra_param_list_key_by_name:
    ckpt["gaussian_model_extra_state_dict"][i] = torch.concat(extra_param_list_key_by_name[i], dim=0)
print("Merging optimizers...")
for i in optimizer_state_exp_avg_list_key_by_index.keys():
    ckpt["optimizer_states"][0]["state"][i]["exp_avg"] = torch.concat(optimizer_state_exp_avg_list_key_by_index[i], dim=0)
    ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"] = torch.concat(optimizer_state_exp_avg_sq_list_key_by_index[i], dim=0)

import internal.renderers.gsplat_renderer

ckpt["hyper_parameters"]["renderer"] = internal.renderers.gsplat_renderer.GSPlatRenderer()

output_path = os.path.join(checkpoint_dir, checkpoint_files[0][:checkpoint_files[0].rfind("-")] + ".ckpt")
print("Saving...")
torch.save(ckpt, output_path)
print(f"Saved to '{output_path}'")

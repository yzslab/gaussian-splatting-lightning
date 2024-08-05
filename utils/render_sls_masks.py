import add_pypath
import os
import argparse
import torch
from tqdm import tqdm
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.gaussian_splatting import GaussianSplatting
from internal.dataset import Dataset, CacheDataLoader
from common import AsyncImageSaver

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
args = parser.parse_args()

load_file = GaussianModelLoader.search_load_file(args.model_path)
# setup model
ckpt = torch.load(load_file, map_location="cpu")
model = GaussianSplatting(**ckpt["hyper_parameters"])
model.setup("validation")
model.on_load_checkpoint(ckpt)
model.load_state_dict(ckpt["state_dict"])
model.cuda()
model.eval()
# setup dataset
dataparser_outputs = ckpt["datamodule_hyper_parameters"]["parser"].instantiate(path=ckpt["datamodule_hyper_parameters"]["path"], output_path=os.getcwd(), global_rank=0).get_outputs()
dataset = Dataset(dataparser_outputs.train_set, undistort_image=False)
dataloader = CacheDataLoader(dataset, max_cache_num=-1, shuffle=False)

image_output_path = os.path.join(os.path.dirname(os.path.dirname(load_file)), "{}-sls_predicts".format(os.path.basename(load_file)))
os.makedirs(image_output_path, exist_ok=True)

image_saver = AsyncImageSaver(is_rgb=True)
try:
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = model.transfer_batch_to_device(batch, model.device, 0)

            outputs = model(batch[0])
            metrics, _ = model.metric.get_train_metrics(model, model.gaussian_model, model.restored_global_step, batch, outputs)

            mask = outputs["raw_pred_mask"].squeeze(-1).repeat(3, 1, 1)
            bool_mask = mask > 0.5

            fused_rgb_mask = batch[1][1].clone()
            fused_rgb_mask[1:3, ~bool_mask[0]] *= 0.25
            fused_rgb_mask[1:3, ~bool_mask[0]] += 0.75

            output_image = (torch.concat([batch[1][1], mask, fused_rgb_mask, outputs["render"].clamp_max(1.)], dim=2) * 255).permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()

            image_saver.save(output_image, os.path.join(image_output_path, batch[1][0] + ".png"))
finally:
    image_saver.stop()

print(f"Saved to `{image_output_path}`")

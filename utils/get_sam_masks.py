import add_pypath
import os
import gc
import argparse
from tqdm import tqdm
from glob import glob
import cv2
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from common import AsyncTensorSaving, AsyncImageReading, AsyncImageSaving

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, default=None)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--sam_ckpt", "-c", type=str, default="sam_vit_h_4b8939.pth")
parser.add_argument("--sam_arch", type=str, default="vit_h")
args = parser.parse_args()

MODEL_DEVICE = "cuda"

# build paths
image_path = args.image_path
output_path = args.output
if output_path is None:
    output_path = os.path.join(os.path.dirname(image_path))
print(f"output_path={output_path}")

# build output dirs
mask_dir = os.path.join(output_path, "semantic", "masks")
os.makedirs(mask_dir, exist_ok=True)

# initialize SAM
print("Initializing SAM...")
model_type = args.sam_arch
sam = sam_model_registry[model_type](checkpoint=args.sam_ckpt).to('cuda')
# predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    box_nms_thresh=0.7,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)

image_list = list(glob(os.path.join(image_path, "**/*.jpg"), recursive=True))
image_list.sort()
image_reader = AsyncImageReading(image_list)
tensor_saver = AsyncTensorSaving()
try:
    with tqdm(range(len(image_list))) as t:
        for _ in t:
            # get image information
            image_full_path, img = image_reader.queue.get()
            image_name = image_full_path[len(image_path):].lstrip("/")

            t.set_description(f"Generating masks: {image_name}")
            semantic_file_name = f"{image_name}.pt"

            # read image
            # img = cv2.imread(image_path)
            # img = image_queue.get()

            # extract masks
            masks = mask_generator.generate(img)
            mask_list = []
            for m in masks:
                # TODO: resize
                m_score = torch.from_numpy(m['segmentation']).float().to('cuda')
                if len(m_score.unique()) < 2:
                    continue
                mask_list.append(m_score.bool())
            tensor_saver.save_tensor(torch.stack(mask_list, dim=0), os.path.join(mask_dir, semantic_file_name))
finally:
    tensor_saver.stop()

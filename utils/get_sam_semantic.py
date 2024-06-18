import add_pypath
import os
import gc
import argparse
from tqdm import tqdm
from glob import glob
import cv2
import torch
import queue
import threading
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
# from internal.renderers.vanilla_depth_renderer import VanillaDepthRenderer
from internal.renderers.gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--sam_ckpt", "-c", type=str, default="sam_vit_h_4b8939.pth")
parser.add_argument("--sam_arch", type=str, default="vit_h")
args = parser.parse_args()

MODEL_DEVICE = "cuda"

# TODO: support vanilla renderer
# search checkpoint and load
load_file = GaussianModelLoader.search_load_file(args.model_path)
assert load_file.endswith(".ckpt")
model, _, ckpt = GaussianModelLoader.initialize_simplified_model_from_checkpoint(load_file, device=MODEL_DEVICE)
# renderer = VanillaDepthRenderer()
renderer = GSplatContrastiveFeatureRenderer()

# load dataset
dataset_path = ckpt["datamodule_hyper_parameters"]["path"] if args.data_path is None else args.data_path
dataparser_outputs = ColmapDataParser(
    path=dataset_path,
    output_path=os.getcwd(),
    global_rank=0,
    params=ckpt["datamodule_hyper_parameters"]["params"].colmap,
).get_outputs()

del ckpt
torch.cuda.empty_cache()
gc.collect()

# build output dirs
mask_dir = os.path.join(dataset_path, "semantic", "masks")
scales_dir = os.path.join(dataset_path, "semantic", "scales")
depths_dir = os.path.join(dataset_path, "semantic", "depths")
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(scales_dir, exist_ok=True)
os.makedirs(depths_dir, exist_ok=True)

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

image_queue = queue.Queue(maxsize=8)


def read_images(image_list: list):
    for i in image_list:
        image_queue.put(cv2.imread(i))
    image_queue.put(None)


# save tensor

tensor_queue = queue.Queue(maxsize=8)


def save_tensor_to_file(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path + ".tmp")
    os.rename(path + ".tmp", path)


def save_tensor_from_queue():
    while True:
        t = tensor_queue.get()
        if t is None:
            break
        tensor, path = t
        save_tensor_to_file(tensor, path)


def save_tensor(tensor, path):
    tensor_queue.put((tensor.cpu(), path))


# save image

image_saving_queue = queue.Queue(maxsize=16)


def save_image_from_queue():
    while True:
        i = image_saving_queue.get()
        if i is None:
            break
        image, path = i
        save_image(image, path)


def save_image(image, path):
    cv2.imwrite(path, image)


def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


# threading.Thread(target=read_images, args=(dataparser_outputs.train_set.image_paths,)).start()
threading.Thread(target=save_image_from_queue).start()
threading.Thread(target=save_tensor_from_queue).start()
try:
    bg_color = torch.zeros((3,), dtype=torch.float, device=MODEL_DEVICE)
    with tqdm(range(len(dataparser_outputs.train_set.image_paths))) as t:
        for i in t:
            # get image information
            image_path = dataparser_outputs.train_set.image_paths[i]
            image_name = dataparser_outputs.train_set.image_names[i]

            t.set_description(f"Generating masks: {image_name}")
            semantic_file_name = f"{image_name}.pt"

            # read image
            img = cv2.imread(image_path)
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
            masks = torch.stack(mask_list, dim=0)
            save_tensor(masks, os.path.join(mask_dir, semantic_file_name))

            # get scale
            ## render depth map
            camera = dataparser_outputs.train_set.cameras[i]
            camera.to_device(MODEL_DEVICE)
            with torch.no_grad():
                depth = renderer.depth_forward(
                    camera,
                    model,
                )  # [C, H, W]
            camera.to_device("cpu")

            image_saving_queue.put((
                depth.permute(1, 2, 0).cpu().numpy(),
                os.path.join(depths_dir, f"{image_name}.tiff")
            ))

            depth = depth[0].cpu()  # [H, W]
            grid_index = generate_grid_index(depth)
            points_in_3D = torch.zeros((depth.shape[0], depth.shape[1], 3), device=MODEL_DEVICE)
            points_in_3D[:, :, -1] = depth

            cx = camera.cx.item()
            cy = camera.cy.item()
            fx = camera.fx.item()
            fy = camera.fy.item()

            points_in_3D[:, :, 0] = (grid_index[:, :, 0] - cx) * depth / fx
            points_in_3D[:, :, 1] = (grid_index[:, :, 1] - cy) * depth / fy

            # TODO: resize mask if not match to depth map
            upsampled_mask = masks.unsqueeze(1)

            eroded_masks = torch.conv2d(
                upsampled_mask.float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3).to(device=upsampled_mask.device),
                padding=1,
            )
            eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)

            scale = torch.zeros(len(masks))
            for mask_id in range(len(masks)):
                point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]

                scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

            save_tensor(scale, os.path.join(scales_dir, semantic_file_name))
finally:
    image_saving_queue.put(None)
    tensor_queue.put(None)

import add_pypath
import argparse
import os
import yaml
import torch
import csv
from glob import glob
from internal.dataparsers.colmap_dataparser import Colmap
from internal.renderers.partition_lod_renderer import PartitionLoDRenderer
from internal.dataset import Dataset, CacheDataLoader
from tqdm import tqdm
from utils.common import AsyncImageSaver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    parser.add_argument("--val-side", "--side", "--val-on", type=str, default="auto")
    parser.add_argument("--level", "-l", type=int, default=None)
    return parser.parse_args()


def get_metric_calculator(device, val_side: str = None):
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr = PeakSignalNoiseRatio(data_range=1.).to(device=device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.).to(device=device)
    vgg_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device=device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device=device)

    def get_metrics(predicts, batch):
        predicted_image = torch.clamp_max(predicts["render"], max=1.)
        gt_image = batch[1][1]

        # mask
        if batch[1][-1] is not None:
            predicted_image = predicted_image * batch[1][-1]
            gt_image = gt_image * batch[1][-1]

        if val_side is not None:
            image_width = gt_image.shape[-1]
            half_width = image_width // 2

            if val_side == "left":
                width_from = 0
                width_to = half_width
            elif val_side == "right":
                width_from = half_width
                width_to = image_width
            else:
                raise RuntimeError()

            predicted_image = predicted_image[..., width_from:width_to]
            gt_image = gt_image[..., width_from:width_to]

        predicted_image = predicted_image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)

        return {
            "psnr": psnr(predicted_image, gt_image),
            "ssim": ssim(predicted_image, gt_image),
            "vgg_lpips": vgg_lpips(predicted_image, gt_image),
            "alex_lpips": alex_lpips(predicted_image, gt_image),
        }, (predicted_image.squeeze(0), gt_image.squeeze(0))

    return get_metrics


@torch.no_grad()
def validate(dataloader, renderer, metric_calculator, image_saver):
    all_metrics = {}

    bg_color = torch.zeros((3,), dtype=torch.float, device="cuda")
    with tqdm(dataloader) as t:
        for camera, (name, image, mask), extra_data in t:
            outputs = renderer(
                camera,
                None,
                bg_color,
            )

            all_metrics[name], (predicted, gt) = metric_calculator(outputs, (camera, (name, image, mask), extra_data))

            image_saver(outputs, (camera, (name, image, mask), extra_data))

    return all_metrics


@torch.no_grad()
def main():
    args = parse_args()

    with open(args.yaml, "r") as f:
        config = yaml.safe_load(f)

    level_name = "{}_levels".format(len(config["names"]))
    if args.level is not None:
        config["names"] = config["names"][args.level:args.level + 1]
        level_name = "level_{}".format(args.level)

    if args.val_side == "auto":
        ckpt_name = config.get("ckpt_name", "")
        if ckpt_name.endswith("retain_appearance-left_optimized"):
            args.val_side = "right"
        elif ckpt_name.endswith("retain_appearance-right_optimized"):
            args.val_side = "left"
        else:
            args.val_side = None
    elif args.val_side.lower() in ["none", "null", "all", "full"]:
        args.val_side = None
    elif args.val_side == "left":
        config["ckpt_name"] = "preprocessed-retain_appearance-right_optimized"
        print("Updated `ckpt_name` to {}".format(config["ckpt_name"]))
    elif args.val_side == "right":
        config["ckpt_name"] = "preprocessed-retain_appearance-left_optimized"
        print("Updated `ckpt_name` to {}".format(config["ckpt_name"]))
    else:
        raise ValueError(args.val_side)

    if args.val_side is None:
        print("[WARNING]Full image validation")
    else:
        print("[WARNING]Validate on half of the image: {}".format(args.val_side))

    # setup renderer
    renderer = PartitionLoDRenderer(**config).instantiate()
    renderer.setup("validation")
    renderer.synchronized = True

    if len(config["names"]) > 1:
        renderer.config.visibility_filter = True
        renderer.gsplat_renderer.runtime_options.radius_clip = 1.5
        renderer.gsplat_renderer.runtime_options.radius_clip_from = 1.5 * renderer.default_partition_size
        print("Automatically enable visibility filter and radius clipping")
        print(renderer.gsplat_renderer.runtime_options)

    # load validation set
    # load a config to get the test set list
    project_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", config["names"][0])
    trained_file_list = list(glob(os.path.join(project_dir, "*-trained")))
    with open(os.path.join(project_dir, trained_file_list[0][:-8], "config.yaml"), "r") as f:
        training_config = yaml.safe_load(f)["data"]["parser"]["init_args"]
    dataparser_outputs = Colmap(
        image_dir=training_config["image_dir"],
        mask_dir=training_config["mask_dir"],
        split_mode=training_config["split_mode"],
        eval_list=training_config["eval_list"],
        scene_scale=training_config["scene_scale"],
        reorient=training_config["reorient"],
        appearance_groups=training_config["appearance_groups"],
        down_sample_factor=training_config["down_sample_factor"],
        down_sample_rounding_mode=training_config["down_sample_rounding_mode"],

        # CHANGED
        eval_image_select_mode="list",
        points_from="random",
    ).instantiate(
        path=os.path.dirname(config["data"]),
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()

    # setup DataLoader
    cuda_device = torch.device("cuda")
    dataloader = CacheDataLoader(
        Dataset(
            dataparser_outputs.test_set,
            undistort_image=False,
            camera_device=cuda_device,
            image_device=cuda_device,
            allow_mask_interpolation=True,
        ),
        max_cache_num=-1,
        shuffle=False,
        num_workers=2,
    )

    output_dir = os.path.join(config["data"], "val-{}-{}{}".format(os.path.basename(args.yaml), level_name, "-{}".format(args.val_side) if args.val_side is not None else ""))

    # image saver
    async_image_saver = AsyncImageSaver(is_rgb=True)

    def image_saver(predicts, batch):
        async_image_saver.save((torch.concat([
            torch.clamp_max(predicts["render"], max=1.),
            batch[1][1],
        ], dim=-1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy(), os.path.join(output_dir, "{}.png".format(batch[1][0])))

    try:
        # start
        metrics = validate(dataloader, renderer, get_metric_calculator(
            cuda_device,
            val_side=args.val_side,
        ), image_saver)
    finally:
        async_image_saver.stop()

    print("Repeat rendering for evaluating FPS...")
    cameras = [camera for camera, _, _ in dataloader]
    bg_color = torch.zeros((3,), dtype=torch.float, device=cameras[0].device)
    n_gaussian_list = []
    time_list = []
    render_time_list = []
    render_time_with_lod_preprocess_list = []
    n_rendered_frames = 0
    for _ in range(8):
        for camera in cameras:
            predicts = renderer(
                camera,
                None,
                bg_color,
            )
            n_gaussian_list.append(predicts["n_gaussians"])
            time_list.append(predicts["time"])
            render_time_list.append(predicts["render_time"])
            render_time_with_lod_preprocess_list.append(predicts["render_time_with_lod_preprocess"])
            n_rendered_frames += 1

    metric_list_key_by_name = {}
    available_metric_keys = ["psnr", "ssim", "vgg_lpips", "alex_lpips"]
    with open(os.path.join(output_dir, "metrics-{}.csv".format(os.path.basename(output_dir))), "w") as f:
        metrics_writer = csv.writer(f)
        metrics_writer.writerow(["name"] + available_metric_keys)
        for image_name, image_metrics in metrics.items():
            metric_row = [image_name]
            for k in available_metric_keys:
                v = image_metrics[k]
                metric_list_key_by_name.setdefault(k, []).append(v)
                metric_row.append(v.item())

            metrics_writer.writerow(metric_row)

        metrics_writer.writerow([""] * len(available_metric_keys))

        mean_row = ["MEAN"]
        for k in available_metric_keys:
            mean_row.append(torch.mean(torch.stack(metric_list_key_by_name[k]).float()).item())
        metrics_writer.writerow(mean_row)

        average_n_gaussians = torch.mean(torch.tensor(n_gaussian_list, dtype=torch.float)).item()
        fps = n_rendered_frames / torch.sum(torch.tensor(time_list, dtype=torch.float))
        render_fps = n_rendered_frames / torch.sum(torch.tensor(render_time_list, dtype=torch.float))
        render_fps_with_lod_preprocess = n_rendered_frames / torch.sum(torch.tensor(render_time_with_lod_preprocess_list, dtype=torch.float))
        metrics_writer.writerow(["FPS", "{}".format(fps)])
        metrics_writer.writerow(["RenderFPS", "{}".format(render_fps)])
        metrics_writer.writerow(["RenderFPSwithLOD", "{}".format(render_fps_with_lod_preprocess)])
        metrics_writer.writerow(["AverageNGaussians", "{}".format(average_n_gaussians)])

        print(mean_row)
        print("FPS={}, RenderFPS={}, RenderFPSwithLOD={}, AverageNGaussians={}".format(fps, render_fps, render_fps_with_lod_preprocess, average_n_gaussians))


main()

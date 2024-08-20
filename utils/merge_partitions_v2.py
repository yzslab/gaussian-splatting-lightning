import add_pypath
import os
import sys
import gc
import json
import argparse
import torch
from tqdm.auto import tqdm
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.vanilla_gaussian import VanillaGaussian, VanillaGaussianModel
from internal.models.appearance_feature_gaussian import AppearanceFeatureGaussianModel
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.renderers.gsplat_renderer import GSPlatRenderer
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.partitioning_utils import MinMaxBoundingBox
from train_partitions import PartitionTraining


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("partition_dir")
    parser.add_argument("--project_dir", "-p", type=str, required=True, help="Directory storing trained partition models")
    parser.add_argument("--output_path", "-o", type=str, required=True)
    parser.add_argument("--min-images", type=int, default=32)
    return parser.parse_args()


def prune_gaussian_model(gaussian_model, mask):
    gaussian_model.properties = {k: v[mask] for k, v in gaussian_model.properties.items()}


def get_partition_gaussian_mask(
        means: torch.Tensor,
        partition_bounding_box: MinMaxBoundingBox,
        orientation_transform: torch.Tensor = None,
):
    if orientation_transform is not None:
        means = means @ orientation_transform[:3, :3].T

    # include min bound, exclude max bound
    is_ge_min = torch.prod(torch.ge(means[..., :2], partition_bounding_box.min), dim=-1)
    is_lt_max = torch.prod(torch.lt(means[..., :2], partition_bounding_box.max), dim=-1)
    is_in_bounding_box = torch.logical_and(is_ge_min, is_lt_max)

    return is_in_bounding_box


def split_partition_gaussians(ckpt: dict, partition_bounding_box: MinMaxBoundingBox, orientation_transform: torch.Tensor = None) -> tuple[
    VanillaGaussianModel,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")
    is_in_partition = get_partition_gaussian_mask(model.get_means(), partition_bounding_box, orientation_transform=orientation_transform)

    inside_part = {}
    outside_part = {}
    for k, v in model.properties.items():
        inside_part[k] = v[is_in_partition]
        outside_part[k] = v[~is_in_partition]

    model.properties = inside_part

    return model, outside_part, is_in_partition


def fuse_appearance_features(ckpt: dict, gaussian_model, cameras_json: list[dict], image_name_to_camera: dict[str, Camera]):
    cuda_device = torch.device("cuda")
    gaussian_device = gaussian_model.get_means().device
    gaussian_model.to(device=cuda_device)
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, "validation", cuda_device)

    cameras = [image_name_to_camera[i["img_name"]] for i in cameras_json]

    from fuse_appearance_embeddings_into_shs_dc import prune_and_get_weights, average_embedding_fusing
    n_average_cameras = 32
    _, visibility_score_pruned_sorted_indices, visibility_score_pruned_top_k_pdf = prune_and_get_weights(
        gaussian_model=gaussian_model,
        cameras=cameras,
        n_average_cameras=n_average_cameras,
        weight_device=cuda_device,
    )
    rgb_offset = average_embedding_fusing(
        gaussian_model,
        renderer,
        n_average_cameras=n_average_cameras,
        cameras=cameras,
        visibility_score_pruned_sorted_indices=visibility_score_pruned_sorted_indices,
        visibility_score_pruned_top_k_pdf=visibility_score_pruned_top_k_pdf,
        view_dir_average_mode="view_direction",
    )

    from internal.utils.sh_utils import RGB2SH
    sh_offset = RGB2SH(rgb_offset)
    gaussian_model.shs_dc = gaussian_model.shs_dc + sh_offset.unsqueeze(1).to(device=gaussian_model.shs_dc.device)
    gaussian_model.to(device=gaussian_device)


def main():
    """
    Overall pipeline:
        * Load the partition data
        * Get trainable partitions and their checkpoint filenames
        * For each partition
          * Load the checkpoint
          * Extract Gaussians falling into the partition bounding box
          * Fuse appearance features into SHs
        * Merge all extracted Gaussians
        * Update the checkpoint
          * Replace GaussianModel with the vanilla one
          * Replace `AppearanceEmbeddingRenderer` with the `GSPlatRenderer`
          * Clear optimizers' states
          * Re-initialize density controller's states
          * Replace with merged Gaussians
        * Saving
    """


    MERGABLE_PROPERTY_NAMES = ["means", "shs_dc", "shs_rest", "scales", "rotations", "opacities"]

    args = parse_args()
    partition_training = PartitionTraining(args.partition_dir)
    trainable_partition_idx_list = partition_training.get_trainable_partition_idx_list(
        min_images=args.min_images,
        n_processes=1,
        process_id=1,
    )

    torch.autograd.set_grad_enabled(False)

    # ensure that all required partitions exist
    mergable_partitions = []
    for partition_idx in tqdm(trainable_partition_idx_list, desc="Searching checkpoints"):
        partition_id_str = partition_training.get_partition_id_str(partition_idx)
        model_dir = os.path.join(args.project_dir, partition_id_str)
        ckpt_file = GaussianModelLoader.search_load_file(model_dir)
        assert ckpt_file.endswith(".ckpt"), "checkpoint not found for partition #{} ({})".format(partition_idx, partition_id_str)
        mergable_partitions.append((
            partition_idx,
            partition_id_str,
            ckpt_file,
        ))

    # load partition info
    partition_bounding_boxes = partition_training.partition_coordinates.get_bounding_boxes(partition_training.scene["scene_config"]["partition_size"])
    orientation_transformation = partition_training.scene["extra_data"]["rotation_transform"] if partition_training.scene["extra_data"] is not None else None

    image_name_to_camera = None

    gaussians_to_merge = {}

    with tqdm(mergable_partitions, desc="Merging") as t:
        for partition_idx, partition_id_str, ckpt_file in t:
            t.set_postfix_str("Loading '{}'...".format(ckpt_file))
            ckpt = torch.load(ckpt_file, map_location="cpu")

            t.set_postfix_str("Splitting...")
            gaussian_model, _, _ = split_partition_gaussians(
                ckpt,
                partition_bounding_boxes[partition_idx],
                orientation_transformation,
            )

            if isinstance(gaussian_model, AppearanceFeatureGaussianModel):
                with open(os.path.join(
                        os.path.dirname(os.path.dirname(ckpt_file)),
                        "cameras.json"
                ), "r") as f:
                    cameras_json = json.load(f)

                if image_name_to_camera is None and isinstance(ckpt["datamodule_hyper_parameters"]["parser"], Colmap):
                    t.set_postfix_str("Loading colmap model...")

                    dataparser_config = Colmap(
                        split_mode="reconstruction",
                        eval_step=64,
                        points_from="random",
                    )
                    for i in ["image_dir", "mask_dir", "scene_scale", "reorient", "appearance_groups", "down_sample_factor", "down_sample_rounding_mode"]:
                        setattr(dataparser_config, i, getattr(ckpt["datamodule_hyper_parameters"]["parser"], i))
                    dataparser_outputs = dataparser_config.instantiate(
                        path=partition_training.dataset_path,
                        output_path=os.getcwd(),
                        global_rank=0,
                    ).get_outputs()

                    image_name_to_camera = {}
                    for idx in range(len(dataparser_outputs.train_set)):
                        image_name = dataparser_outputs.train_set.image_names[idx]
                        camera = dataparser_outputs.train_set.cameras[idx]
                        image_name_to_camera[image_name] = camera

                t.set_postfix_str("Fusing...")
                fuse_appearance_features(
                    ckpt,
                    gaussian_model,
                    cameras_json,
                    image_name_to_camera=image_name_to_camera,
                )

            for i in MERGABLE_PROPERTY_NAMES:
                gaussians_to_merge.setdefault(i, []).append(gaussian_model.get_property(i))

    # merge
    print("Merging...")
    merged_gaussians = {}
    for k, v in gaussians_to_merge.items():
        merged_gaussians[k] = torch.concat(v, dim=0)
        # release merged one to avoid OOM
        v.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # replace `AppearanceFeatureGaussian` with `VanillaGaussian`
    ckpt["hyper_parameters"]["gaussian"] = VanillaGaussian(sh_degree=gaussian_model.max_sh_degree)

    # remove `GSplatAppearanceEmbeddingRenderer`'s states from ckpt
    state_dict_key_to_delete = []
    for i in ckpt["state_dict"]:
        if i.startswith("renderer."):
            state_dict_key_to_delete.append(i)
    for i in state_dict_key_to_delete:
        del ckpt["state_dict"][i]

    # replace `GSplatAppearanceEmbeddingRenderer` with `GSPlatRenderer`
    anti_aliased = True
    if isinstance(ckpt["hyper_parameters"]["renderer"], VanillaRenderer):
        anti_aliased = False
    ckpt["hyper_parameters"]["renderer"] = GSPlatRenderer(anti_aliased=anti_aliased)

    # remove existing Gaussians from ckpt
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians."):
            del ckpt["state_dict"][i]

    # remove optimizer states
    ckpt["optimizer_states"] = []

    # reinitialize density controller states
    if isinstance(ckpt["hyper_parameters"]["density"], VanillaDensityController):
        for k in list(ckpt["state_dict"].keys()):
            if k.startswith("density_controller."):
                ckpt["state_dict"][k] = torch.zeros((merged_gaussians["means"].shape[0], *ckpt["state_dict"][k].shape[1:]), dtype=ckpt["state_dict"][k].dtype)

    # add merged gaussians to ckpt
    for k, v in merged_gaussians.items():
        ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = v

    # save
    print("Saving...")
    torch.save(ckpt, args.output_path)
    print("Saved to '{}'".format(args.output_path))


def test_main():
    sys.argv = [
        __file__,
        os.path.expanduser("~/dataset/JNUCar_undistorted/colmap/drone/dense_max_2048/0/partitions-size_3.0-enlarge_0.1-visibility_0.9_0.1"),
        "-p", os.path.join("..", "outputs", "JNUAerial-0820"),
        "-o", os.path.join("..", "outputs", "JNUAerial-0820", "merged.ckpt"),
    ]
    main()


if __name__ == "__main__":
    main()

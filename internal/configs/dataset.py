from typing import Optional
from dataclasses import dataclass


@dataclass
class ColmapParams:
    """
        Args:
            mask_dir:
                the path to the directory store mask files;
                the mask file of the image `a/image_name.jpg` is `a/image_name.jpg.png`;
                single channel, 0 is the masked pixel;

            eval_step: -1: use all images as training set; > 1: pick an image for every eval_step

            reorient: whether reorient the scene
    """

    image_dir: str = None

    mask_dir: str = None

    eval_step: int = -1

    scene_scale: float = 1.  # TODO

    reorient: bool = False  # TODO


@dataclass
class BlenderParams:
    white_background: bool = False


@dataclass
class DatasetParams:
    """
        Args:
            train_max_num_images_to_cache: limit the max num images to be load at the same time

            val_max_num_images_to_cache: limit the max num images to be load at the same time
    """

    colmap: ColmapParams

    blender: BlenderParams

    image_scale_factor: float = 1.  # TODO

    train_max_num_images_to_cache: int = -1

    val_max_num_images_to_cache: int = 0

    test_max_num_images_to_cache: int = 0

    num_workers: int = 8

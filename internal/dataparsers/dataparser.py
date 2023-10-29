from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np

from internal.cameras.cameras import Cameras


@dataclass
class ImageSet:
    image_names: list

    image_paths: list
    """ Full path to the image file """

    mask_paths: Optional[list]
    """ Full path to the mask file """

    cameras: Cameras

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        return self.image_names[index], self.image_paths[index], self.mask_paths[index], self.cameras[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = [None for _ in range(len(self.image_paths))]


@dataclass
class PointCloud:
    xyz: np.ndarray

    rgb: np.ndarray


@dataclass
class DataParserOutputs:
    train_set: ImageSet

    val_set: ImageSet

    test_set: ImageSet

    point_cloud: PointCloud

    # ply_path: str

    camera_extent: float

    appearance_group_ids: Optional[dict]


class DataParser:
    def get_outputs(self) -> DataParserOutputs:
        """
        :return: [training set, validation set, point cloud]
        """

        pass

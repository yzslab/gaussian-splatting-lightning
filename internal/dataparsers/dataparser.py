from typing import Tuple
from dataclasses import dataclass
from internal.cameras.cameras import Cameras
from internal.utils.graphics_utils import BasicPointCloud


@dataclass
class ImageSet:
    image_names: list

    image_paths: list
    """ Full path to the image file """

    mask_paths: list
    """ Full path to the mask file """

    cameras: Cameras

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        return self.image_names[index], self.image_paths[index], self.mask_paths[index], self.cameras[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class DataParserOutputs:
    train_set: ImageSet

    val_set: ImageSet

    test_set: ImageSet

    point_cloud: BasicPointCloud

    ply_path: str

    camera_extent: float


class DataParser:
    def get_outputs(self) -> DataParserOutputs:
        """
        :return: [training set, validation set, point cloud]
        """

        pass

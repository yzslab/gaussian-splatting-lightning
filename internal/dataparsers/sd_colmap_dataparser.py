import os
import numpy as np
import torch

from dataclasses import dataclass
from typing import Literal, Tuple
from . import DataParserOutputs
from .colmap_dataparser import Colmap, ColmapDataParser


@dataclass
class SDColmap(Colmap):
    sd_feature_dir: str = "SD"

    train_keyword: str = "clutter"

    test_keyword: str = "extra"

    split_mode: Literal["reconstruction", "experiment", "keyword"] = "keyword"

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "SDColmapDataParser":
        return SDColmapDataParser(path, output_path, global_rank, self)


class SDColmapDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: SDColmap) -> None:
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        # val_set and test_set are same object
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                image_name_without_ext = image_name[:image_name.rfind(".")]
                semantic_file_name = f"{image_name_without_ext}.npy"
                image_set.extra_data[idx] = os.path.join(self.path, self.params.sd_feature_dir, semantic_file_name)
            image_set.extra_data_processor = SDColmapDataParser.read_sd_feature

        return dataparser_outputs

    def build_split_indices(self, image_name_list) -> Tuple[list, list]:
        if self.params.split_mode != "keyword":
            return super().build_split_indices(image_name_list)

        train_indices = []
        test_indices = []

        for idx, image_name in enumerate(image_name_list):
            if image_name.find(self.params.train_keyword) != -1:
                train_indices.append(idx)
            elif image_name.find(self.params.test_keyword) != -1:
                test_indices.append(idx)

        return train_indices, test_indices

    @staticmethod
    def read_sd_feature(path):
        sd_feature = np.load(path)
        return torch.tensor(sd_feature, dtype=torch.float)

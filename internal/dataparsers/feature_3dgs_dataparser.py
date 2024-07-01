import os
import torch

from . import DataParserOutputs
from .colmap_dataparser import ColmapDataParser
from ..configs.dataset import Feature3DGSColmapParams


class Feature3DGSColmapDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: Feature3DGSColmapParams) -> None:
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        # val_set and test_set are same object
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                semantic_file_name = f"{image_name}.pt"
                image_set.extra_data[idx] = os.path.join(self.path, self.params.feature_dir, semantic_file_name)
            image_set.extra_data_processor = Feature3DGSColmapDataParser.read_semantic_data

        # remove image paths to avoid caching
        for i in [dataparser_outputs.train_set, dataparser_outputs.val_set, dataparser_outputs.test_set]:
            for j in range(len(i.image_paths)):
                i.image_paths[j] = None

        return dataparser_outputs

    @staticmethod
    def read_semantic_data(path):
        return torch.load(path, map_location="cpu")

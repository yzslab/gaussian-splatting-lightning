import os
import torch

from . import DataParserOutputs
from .colmap_dataparser import ColmapDataParser
from ..configs.dataset import SemanticColmapParams


class SemanticColmapDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: SemanticColmapParams) -> None:
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        # val_set and test_set are same object
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                semantic_file_name = f"{image_name}.pt"
                image_set.extra_data[idx] = (
                    os.path.join(self.path, self.params.semantic_mask_dir, semantic_file_name),
                    os.path.join(self.path, self.params.semantic_scale_dir, semantic_file_name),
                )
            image_set.extra_data_processor = SemanticColmapDataParser.read_semantic_data

        return dataparser_outputs

    @staticmethod
    def read_semantic_data(paths):
        mask_path, scale_path = paths
        return torch.load(mask_path, map_location="cpu"), torch.load(scale_path, map_location="cpu")

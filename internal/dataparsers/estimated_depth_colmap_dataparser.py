from typing import Union, Optional
import os
import json
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from .colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers import DataParserOutputs
from internal.cameras import Camera


@dataclass
class EstimatedDepthColmap(Colmap):
    depth_dir: str = "estimated_depths"

    depth_sparse_dir: str = None

    depth_rescaling: bool = True

    depth_scale_name: str = "estimated_depth_scales"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.

    allow_depth_interpolation: bool = False

    median_normalization: bool = False

    depth_in_uint16: bool = False

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "EstimatedDepthColmapDataParser":
        return EstimatedDepthColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


@dataclass
class DepthMap:
    depth: Union[torch.Tensor, np.ndarray]

    scale: Optional[float] = None

    offset: Optional[float] = None

    median: Optional[float] = None

    mask: Optional[torch.Tensor] = None

    camera: Optional[Camera] = None

    @classmethod
    def create(cls, depth_map, scale, offset, in_uint16: bool = False):
        """
        in_uint16: depth_map in uint16, scale, offset
        not in_uint16: transform and scale depth_map to float32 tensor
        """

        if in_uint16:
            assert depth_map.dtype == np.uint16

            return cls(
                depth=depth_map,
                scale=scale,
                offset=offset,
            )

        if depth_map.dtype == np.uint16:
            depth_map = torch.from_numpy(depth_map.astype(np.float32)) / 65535.
        depth_map = depth_map * scale + offset
        depth_map = torch.clamp_min(depth_map, min=0.)
        return cls(
            depth=depth_map,
        )

    def media_normalize(self):
        """
        in_uint16: only calculate the median
        not in_uint16: calculate the median and normalize the depth
        """
        if self.depth.dtype == np.uint16:
            depth = self.get_scaled()
            self.median = torch.median(depth[depth > 0])
        else:
            self.median = torch.median(self.depth[self.depth > 0])
            self.depth = self.depth / self.median

    def get_scaled(self, device="cpu"):
        """
        scale uint16 depth to float32 one
        """
        depth = torch.from_numpy(self.depth.astype(np.float32)).to(device=device) / 65535.
        return (depth * self.scale + self.offset).clamp_min_(min=0.)

    def get(self, device="cpu"):
        if self.depth.dtype == np.uint16:
            depth = self.get_scaled(device=device)
            if self.median is not None:
                depth = depth / self.median
            return depth
        return self.depth.to(device=device)


class EstimatedDepthColmapDataParser(ColmapDataParser):
    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        # allow depth maps have a different sparse model
        depth_set = dataparser_outputs.train_set
        if self.params.depth_sparse_dir is not None:
            depth_set = Colmap(
                # image_dir=self.params.depth_dir,
                sparse_dir=self.params.depth_sparse_dir,
                eval_step=99999999,
                scene_scale=self.params.scene_scale,
                reorient=self.params.reorient,
                points_from="random",
                n_random_points=0,
            ).instantiate(self.path, self.output_path, self.global_rank).get_outputs().train_set
        image_name_to_depth_camera = {}
        for name, camera in zip(depth_set.image_names, depth_set.cameras):
            image_name_to_depth_camera[name] = camera

        if self.params.depth_rescaling is True:
            with open(os.path.join(self.path, self.params.depth_scale_name + ".json"), "r") as f:
                depth_scales = json.load(f)

            image_name_set = {image_name: True for image_name in dataparser_outputs.train_set.image_names + dataparser_outputs.val_set.image_names}
            depth_scale_list = []
            for image_name, image_depth_scale in depth_scales.items():
                if image_name not in image_name_set:
                    continue
                depth_scale_list.append(image_depth_scale["scale"])

            median_scale = np.median(np.asarray(depth_scale_list))

        loaded_depth_count = 0
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{image_name}.npy")
                if os.path.exists(depth_file_path) is False:
                    depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{image_name}.uint16.png")
                    if os.path.exists(depth_file_path) is False:
                        print("[WARNING] {} does not have a depth file".format(image_name))
                        continue

                depth_scale = {
                    "scale": 1.,
                    "offset": 0.,
                }
                if self.params.depth_rescaling is True:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        print("[WARNING {} does not have a depth scale]".format(image_name))
                        continue
                    if depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale:
                        print("[WARNING depth scale '{}' of '{}' out of bound '({}, {})']".format(
                            depth_scale["scale"],
                            image_name,
                            self.params.depth_scale_lower_bound * median_scale,
                            self.params.depth_scale_upper_bound * median_scale,
                        ))
                        continue

                image_set.extra_data[idx] = (depth_file_path, depth_scale, image_name_to_depth_camera[image_name])
                loaded_depth_count += 1
            image_set.extra_data_processor = self.get_depth_loader(
                self.params.allow_depth_interpolation,
                self.params.median_normalization,
                self.params.depth_in_uint16,
            )

        assert loaded_depth_count > 0
        print("found {} depth maps".format(loaded_depth_count))

        return dataparser_outputs

    @staticmethod
    def load_npy_depth(file_path):
        return np.load(file_path)

    @staticmethod
    def load_uint16_png_depth(file_path):
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        assert image.ndim == 2 and image.dtype == np.uint16

        return image

        return image.astype(np.float32) / 65535.

    @classmethod
    def load_depth_file(cls, file_path):
        if file_path.endswith(".npy"):
            return cls.load_npy_depth(file_path)
        return cls.load_uint16_png_depth(file_path)

    @classmethod
    def get_depth_loader(cls, allow_depth_interpolation: bool, median_normalization: bool, depth_in_uint16: bool):
        def load_depth(depth_info):
            if depth_info is None:
                return None

            depth_file_path, depth_scale, camera = depth_info

            depth = DepthMap.create(
                cls.load_depth_file(depth_file_path),
                scale=depth_scale["scale"],
                offset=depth_scale["offset"],
                in_uint16=depth_in_uint16,
            )

            depth_hw = depth.depth.shape[:2]
            if not (depth_hw[0] == camera.height.item() and depth_hw[1] == camera.width.item()):
                camera = camera.rescale(
                    width=depth_hw[1],
                    height=depth_hw[0],
                )
            depth.camera = camera

            # depth = cls.load_depth_file(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]
            # depth = torch.tensor(depth, dtype=torch.float)
            # depth = torch.clamp_min(depth, min=0.)

            # if depth.shape != image_shape:
            #     assert allow_depth_interpolation, "the shape '{}' of depth map '{}' and '{}' of image not match, add the '--data.parser.allow_depth_interpolation=true' if you are sure this is expected".format(depth.shape, depth_file_path, image_shape)
            #     depth = torch.nn.functional.interpolate(
            #         depth[None, None, ...],
            #         image_shape,
            #         mode="bilinear",
            #         align_corners=True,
            #     )[0, 0]

            if median_normalization:
                depth.media_normalize()
                assert not torch.any(torch.isnan(depth.get())), depth_file_path
                # median = torch.median(depth[depth > 0])
                # depth = depth / median
                # assert not torch.any(torch.isnan(depth)), depth_file_path
                # return depth, median
            return depth

        return load_depth

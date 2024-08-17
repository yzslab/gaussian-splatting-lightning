import os.path
from typing import Tuple, Callable
import dataclasses
from dataclasses import dataclass
from tqdm.auto import tqdm
import torch


@dataclass
class SceneConfig:
    origin: torch.Tensor

    partition_size: float

    location_based_enlarge: float = None
    """ enlarge bounding box by `partition_size * location_based_enlarge`, used for location based camera assignment """

    visibility_based_distance: float = None
    """ enlarge bounding box by `partition_size * visibility_based_distance`, used for visibility based camera assignment """

    visibility_threshold: float = None


@dataclass
class MinMaxBoundingBox:
    min: torch.Tensor  # [2]
    max: torch.Tensor  # [2]


@dataclass
class MinMaxBoundingBoxes:
    min: torch.Tensor  # [N, 2]
    max: torch.Tensor  # [N, 2]

    def to(self, *args, **kwargs):
        self.min = self.min.to(*args, **kwargs)
        self.max = self.max.to(*args, **kwargs)
        return self


@dataclass
class SceneBoundingBox:
    bounding_box: MinMaxBoundingBox
    n_partitions: torch.Tensor
    origin_partition_offset: torch.Tensor


@dataclass
class PartitionCoordinates:
    id: torch.Tensor  # [N_partitions, 2]
    xy: torch.Tensor  # [N_partitions, 2]

    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, item):
        return self.id[item], self.xy[item]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.id[idx], self.xy[idx]

    def get_bounding_boxes(self, size: float, enlarge: float = 0.) -> MinMaxBoundingBoxes:
        xy_min = self.xy - (enlarge * size)  # [N_partitions, 2]
        xy_max = self.xy + size + (enlarge * size)  # [N_partitions, 2]
        return MinMaxBoundingBoxes(
            min=xy_min,
            max=xy_max,
        )


class Partitioning:
    @staticmethod
    def get_bounding_box_by_camera_centers(camera_centers: torch.Tensor, enlarge: float = 0.) -> MinMaxBoundingBox:
        xyz_min = torch.min(camera_centers, dim=0)[0]
        xyz_min -= xyz_min * enlarge

        xyz_max = torch.max(camera_centers, dim=0)[0]
        xyz_max += xyz_max * enlarge

        return MinMaxBoundingBox(
            min=xyz_min,
            max=xyz_max,
        )

    @staticmethod
    def align_xyz(xyz, origin: torch.Tensor, size: float):
        xyz_radius_factor = (xyz - origin) / size
        n_partitions = torch.ceil(torch.abs(xyz_radius_factor)).to(torch.long)
        xyz_ceil_radius_factor = n_partitions * torch.sign(xyz_radius_factor)
        new_xyz = origin + xyz_ceil_radius_factor * size
        return new_xyz, n_partitions

    @classmethod
    def align_bounding_box(cls, bounding_box: MinMaxBoundingBox, origin: torch.Tensor, size: float):
        # TODO: if center out of bounding box, incorrect result will be produced
        assert torch.all(origin >= bounding_box.min), "center {} out-of-min-bound {}".format(origin, bounding_box.min)
        assert torch.all(origin <= bounding_box.max), "center {} out-of-max-bound {}".format(origin, bounding_box.max)

        new_min, n1 = cls.align_xyz(bounding_box.min, origin, size)
        new_max, n2 = cls.align_xyz(bounding_box.max, origin, size)

        aligned = MinMaxBoundingBox(
            min=new_min,
            max=new_max,
        )

        return SceneBoundingBox(
            bounding_box=aligned,
            n_partitions=n1 + n2,
            origin_partition_offset=-n1,
        )

    @staticmethod
    def build_partition_coordinates(scene_bounding_box: SceneBoundingBox, size: float):
        partition_count = scene_bounding_box.n_partitions
        origin_partition_offset = scene_bounding_box.origin_partition_offset
        grid_x, grid_y = torch.meshgrid(
            torch.arange(partition_count[0], dtype=torch.int) + origin_partition_offset[0],
            torch.arange(partition_count[1], dtype=torch.int) + origin_partition_offset[1],
            indexing="xy",
        )

        partition_id = torch.dstack([grid_x, grid_y])
        partition_xy = partition_id * size

        return PartitionCoordinates(
            id=partition_id.reshape(torch.prod(partition_count), 2),
            xy=partition_xy.reshape(torch.prod(partition_count), 2),
        )

    @staticmethod
    def is_in_bounding_boxes(
            bounding_boxes: MinMaxBoundingBoxes,
            coordinates: torch.Tensor,  # [N, 2]
    ):
        xy_min = bounding_boxes.min  # [N_partitions, 2]
        xy_max = bounding_boxes.max  # [N_partitions, 2]

        coordinates = coordinates.unsqueeze(0)  # [1, N_coordinates, 2]
        xy_min = xy_min.unsqueeze(1)  # [N_partitions, 1, 2]
        xy_max = xy_max.unsqueeze(1)  # [N_partitions, 1, 2]

        is_gt_min = torch.prod(torch.ge(coordinates, xy_min), dim=-1)  # [N_partitions, N_coordinates]
        is_le_max = torch.prod(torch.le(coordinates, xy_max), dim=-1)  # [N_partitions, N_coordinates]
        is_in_partition = is_gt_min * is_le_max != 0  # [N_partitions, N_coordinates]

        return is_in_partition

    @classmethod
    def camera_center_based_partition_assignment(
            cls,
            partition_coordinates: PartitionCoordinates,
            camera_centers: torch.Tensor,  # [N_images, 2]
            size: float,
            enlarge: float = 0.1,
    ) -> torch.Tensor:
        assert enlarge >= 0.

        bounding_boxes = partition_coordinates.get_bounding_boxes(
            size=size,
            enlarge=enlarge,
        )
        return cls.is_in_bounding_boxes(
            bounding_boxes=bounding_boxes,
            coordinates=camera_centers,
        )

    @classmethod
    def cameras_point_based_visibilities_calculation(
            cls,
            partition_coordinates: PartitionCoordinates,
            size: float,
            n_cameras,
            point_getter: Callable[[int], torch.Tensor],
            device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(size, enlarge=0.).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_3d = point_getter(camera_idx)
            visibilities, _ = Partitioning.calculate_point_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points=points_3d[..., :2],
            )  # [N_partitions]
            all_visibilities[camera_idx].copy_(visibilities.to(device=all_visibilities.device))

        from concurrent.futures.thread import ThreadPoolExecutor
        with ThreadPoolExecutor() as tpe:
            for _ in tqdm(
                    tpe.map(calculate_visibilities, range(n_cameras)),
                    total=n_cameras,
            ):
                pass

        assert torch.all(all_visibilities >= 0.)

        return all_visibilities.T  # [N_partitions, N_cameras]

    @classmethod
    def visibility_based_partition_assignment(
            cls,
            partition_coordinates: PartitionCoordinates,
            camera_centers: torch.Tensor,
            size: float,
            max_distance: float,
            assigned_mask: torch.Tensor,  # the Tensor produced by `camera_center_based_partition_assignment` above
            visibilities: torch.Tensor,  # [N_partitions, N_cameras]
            visibility_threshold: float,
    ):
        is_in_range = cls.camera_center_based_partition_assignment(
            partition_coordinates,
            camera_centers,
            size,
            max_distance,
        )  # [N_partitions, N_cameras]
        # exclude assigned
        is_not_assigned = torch.logical_and(is_in_range, torch.logical_not(assigned_mask))

        is_visible = torch.ge(visibilities, visibility_threshold)  # [N_partitions, N_cameras]

        return torch.logical_and(is_visible, is_not_assigned)

    @classmethod
    def calculate_point_based_visibilities(
            cls,
            partition_bounding_boxes: MinMaxBoundingBoxes,
            points: torch.Tensor,  # [N_points, 2 or 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.shape[0] == 0:
            n_points_in_partitions = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.long)
            visibilities = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.float)
        else:
            is_in_bounding_boxes = cls.is_in_bounding_boxes(
                bounding_boxes=partition_bounding_boxes,
                coordinates=points,
            )  # [N_partitions, N_points]
            n_points_in_partitions = is_in_bounding_boxes.sum(dim=-1)  # [N_partitions]
            visibilities = n_points_in_partitions / points.shape[0]  # [N_partitions]

        return visibilities, n_points_in_partitions

    @classmethod
    def save_partitions(
            cls,
            dir: str,
            scene_config: SceneConfig,
            scene_bounding_box: SceneBoundingBox,
            partition_coordinates: PartitionCoordinates,
            visibilities: torch.Tensor,  # [N_partitions, N_cameras]
            location_based_assignments: torch.Tensor,  # [N_partitions, N_cameras]
            visibility_based_assignments: torch.Tensor,  # [N_partitions, N_cameras]
    ):
        save_to = os.path.join(dir, "partitions.pt")
        torch.save(
            {
                "scene_config": dataclasses.asdict(scene_config),
                "scene_bounding_box": dataclasses.asdict(scene_bounding_box),
                "partition_coordinates": dataclasses.asdict(partition_coordinates),
                "visibilities": visibilities,
                "location_based_assignments": location_based_assignments,
                "visibility_based_assignments": visibility_based_assignments,
            },
            save_to,
        )
        return save_to

    @classmethod
    def partition_id_to_str(cls, id):
        x, y = id[0], id[1]
        if isinstance(id, torch.Tensor):
            x = x.item()
            y = y.item()

        return "{:03d}_{:03d}".format(x, y)

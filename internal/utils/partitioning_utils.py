import os
from typing import Tuple, Callable, Union
import dataclasses
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
from matplotlib.pyplot import cm


@dataclass
class SceneConfig:
    origin: torch.Tensor

    partition_size: float

    location_based_enlarge: float = None
    """ enlarge bounding box by `partition_size * location_based_enlarge`, used for location based camera assignment """

    visibility_based_distance: float = None
    """ enlarge bounding box by `partition_size * visibility_based_distance`, used for visibility based camera assignment """

    visibility_based_partition_enlarge: float = None
    """ enlarge bounding box by `partition_size * location_based_enlarge`, the points in this bounding box will be treated as inside partition """

    visibility_threshold: float = None

    visibility_based_distance_enlarge_on_no_location_based: float = 4.
    """ distance = distance *  visibility_based_distance_enlarge_on_no_location_based """

    visibility_threshold_reduce_on_no_location_based: float = 4.
    """ visibility_threshold = visibility_threshold / visibility_threshold_reduce_on_no_location_based """

    convex_hull_based_visibility: bool = False


@dataclass
class MinMaxBoundingBox:
    min: torch.Tensor  # [2]
    max: torch.Tensor  # [2]


@dataclass
class MinMaxBoundingBoxes:
    min: torch.Tensor  # [N, 2]
    max: torch.Tensor  # [N, 2]

    def __getitem__(self, item):
        return MinMaxBoundingBox(
            min=self.min[item],
            max=self.max[item],
        )

    def to(self, *args, **kwargs):
        self.min = self.min.to(*args, **kwargs)
        self.max = self.max.to(*args, **kwargs)
        return self


@dataclass
class SceneBoundingBox:
    bounding_box: MinMaxBoundingBox
    n_partitions: torch.Tensor  # = [N_x, N_y]
    origin_partition_offset: torch.Tensor


@dataclass
class PartitionCoordinates:
    id: torch.Tensor  # [N_partitions, 2]
    xy: torch.Tensor  # [N_partitions, 2]
    size: torch.Tensor  # [N_partitions, 2]

    def __len__(self):
        return self.id.shape[0]

    def __getitem__(self, item):
        return self.id[item], self.xy[item]

    def __iter__(self):
        for idx in range(len(self)):
            yield self.id[idx], self.xy[idx], self.size[idx]

    def get_bounding_boxes(self, enlarge: Union[float, torch.Tensor] = 0.) -> MinMaxBoundingBoxes:
        xy_min = self.xy - (enlarge * self.size)  # [N_partitions, 2]
        xy_max = self.xy + self.size + (enlarge * self.size)  # [N_partitions, 2]
        return MinMaxBoundingBoxes(
            min=xy_min,
            max=xy_max,
        )

    def get_str_id(self, idx: int) -> str:
        return Partitioning.partition_id_to_str(self.id[idx])


@dataclass
class PartitionableScene:
    scene_config: SceneConfig = None

    camera_centers: torch.Tensor = None

    camera_center_based_bounding_box: MinMaxBoundingBox = None

    point_based_bounding_box: MinMaxBoundingBox = None

    scene_bounding_box: SceneBoundingBox = None

    partition_coordinates: PartitionCoordinates = None

    is_camera_in_partition: torch.Tensor = None  # [N_partitions, N_cameras]

    camera_visibilities: torch.Tensor = None  # [N_partitions, N_cameras]

    is_partitions_visible_to_cameras: torch.Tensor = None  # [N_partitions, N_cameras]

    unmerged = None

    def get_bounding_box_by_camera_centers(self, enlarge: float = 0.):
        self.camera_center_based_bounding_box = Partitioning.get_bounding_box_by_camera_centers(self.camera_centers, enlarge=enlarge)
        return self.camera_center_based_bounding_box

    def get_bounding_box_by_points(self, points: torch.Tensor, enlarge: float = 0., outlier_threshold: float = 0.001):
        self.point_based_bounding_box = Partitioning.get_bounding_box_by_points(
            points,
            enlarge=enlarge,
            outlier_threshold=outlier_threshold,
        )

        return self.point_based_bounding_box

    def get_scene_bounding_box(self):
        self.scene_bounding_box = Partitioning.align_bounding_box(
            self.point_based_bounding_box,
            origin=self.scene_config.origin,
            size=self.scene_config.partition_size,
        )
        return self.scene_bounding_box

    def build_partition_coordinates(self):
        self.partition_coordinates = Partitioning.build_partition_coordinates(
            self.scene_bounding_box,
            self.scene_config.origin,
            self.scene_config.partition_size,
        )
        return self.partition_coordinates

    def camera_center_based_partition_assignment(self):
        self.is_camera_in_partition = Partitioning.camera_center_based_partition_assignment(
            partition_coordinates=self.partition_coordinates,
            camera_centers=self.camera_centers,
            enlarge=self.scene_config.location_based_enlarge,
        )
        return self.is_camera_in_partition

    def calculate_camera_visibilities(self, *args, **kwargs):
        if self.scene_config.convex_hull_based_visibility:
            return self.calculate_convex_hull_based_camera_visibilities(*args, **kwargs)
        return self.calculate_point_based_camera_visibilities(*args, **kwargs)

    def calculate_point_based_camera_visibilities(self, point_getter: Callable, device):
        self.camera_visibilities = Partitioning.cameras_point_based_visibilities_calculation(
            partition_coordinates=self.partition_coordinates,
            n_cameras=self.camera_centers.shape[0],
            point_getter=point_getter,
            device=device,
        )
        return self.camera_visibilities

    def calculate_convex_hull_based_camera_visibilities(self, point_getter: Callable, device):
        self.camera_visibilities = Partitioning.cameras_convex_hull_based_visibilities_calculation(
            partition_coordinates=self.partition_coordinates,
            n_cameras=self.camera_centers.shape[0],
            point_getter=point_getter,
            enlarge=self.scene_config.visibility_based_partition_enlarge,
            device=device,
        )
        return self.camera_visibilities

    def visibility_based_partition_assignment(self):
        self.is_partitions_visible_to_cameras = Partitioning.visibility_based_partition_assignment(
            partition_coordinates=self.partition_coordinates,
            camera_centers=self.camera_centers,
            max_distance=self.scene_config.visibility_based_distance,
            assigned_mask=self.is_camera_in_partition,
            visibilities=self.camera_visibilities,
            visibility_threshold=self.scene_config.visibility_threshold,
            no_camera_enlarge_distance=self.scene_config.visibility_based_distance_enlarge_on_no_location_based,
            no_camera_reduce_threshold=self.scene_config.visibility_threshold_reduce_on_no_location_based,
        )
        return self.is_partitions_visible_to_cameras

    def merge_no_location_based_partitions(self, min_location_based: int = 32):
        self.pre_merge()

        shape = self.scene_bounding_box.n_partitions[1], self.scene_bounding_box.n_partitions[0]
        n_location_based_assignments = self.is_camera_in_partition.sum(dim=-1).reshape(shape)  # [H, W]
        has_merged = torch.zeros_like(n_location_based_assignments, dtype=torch.bool)

        partition_ids = []
        partition_xys = []
        partition_sizes = []
        is_camera_in_partition = []
        is_partitions_visible_to_cameras = []

        def get_partition_idx(rowIdx, colIdx):
            return shape[1] * rowIdx + colIdx

        for rowIdx in range(shape[0]):
            for colIdx in range(shape[1]):
                partition_idx = get_partition_idx(rowIdx, colIdx)

                # skip merged
                if has_merged[rowIdx, colIdx]:
                    continue

                def unmergeable():
                    partition_ids.append(self.partition_coordinates.id[partition_idx])
                    partition_xys.append(self.partition_coordinates.xy[partition_idx])
                    partition_sizes.append(self.partition_coordinates.size[partition_idx])
                    is_camera_in_partition.append(self.is_camera_in_partition[partition_idx])
                    is_partitions_visible_to_cameras.append(self.is_partitions_visible_to_cameras[partition_idx])

                # find unmerged neighbors
                # left_mergeable = False
                # left_n_location_based_cameras = 0
                right_mergeable = False
                right_n_location_based_cameras = 0
                up_mergeable = False
                up_n_location_based_cameras = 0
                up_right_n_location_based_cameras = 0
                up_right_mergeable = False
                # down_mergeable = False
                # down_n_location_based_cameras = 0
                # # left
                # if colIdx - 1 >= 0:
                #     left_mergeable = not has_merged[rowIdx, colIdx - 1]
                #     left_n_location_based_cameras = n_location_based_assignments[rowIdx, colIdx - 1]
                # right
                if colIdx + 1 < shape[1]:
                    right_mergeable = not has_merged[rowIdx, colIdx + 1]
                    right_n_location_based_cameras = n_location_based_assignments[rowIdx, colIdx + 1]
                # up
                if rowIdx + 1 < shape[0]:
                    up_mergeable = not has_merged[rowIdx + 1, colIdx]
                    up_n_location_based_cameras = n_location_based_assignments[rowIdx + 1, colIdx]
                # up right
                if rowIdx + 1 < shape[0] and colIdx + 1 < shape[1]:
                    up_right_mergeable = not has_merged[rowIdx + 1, colIdx + 1]
                    up_right_n_location_based_cameras = n_location_based_assignments[rowIdx + 1, colIdx + 1]
                # # down
                # if rowIdx - 1 >= 0:
                #     down_mergeable = not has_merged[rowIdx - 1, colIdx]
                #     down_n_location_based_cameras = n_location_based_assignments[rowIdx - 1, colIdx]

                n_partitions_have_location_based = int(right_n_location_based_cameras >= min_location_based) + int(up_n_location_based_cameras >= min_location_based) + int(up_right_n_location_based_cameras >= min_location_based)
                n_mergeable = int(right_mergeable) + int(up_mergeable) + int(up_right_mergeable)

                def merge_three_neighbors():
                    right_idx = get_partition_idx(rowIdx, colIdx + 1)
                    up_idx = get_partition_idx(rowIdx + 1, colIdx)
                    up_right_idx = get_partition_idx(rowIdx + 1, colIdx + 1)

                    # make sure all have same size
                    all_same_size = True
                    for i in [right_idx, up_idx, up_right_idx]:
                        if not torch.allclose(self.partition_coordinates.size[partition_idx], self.partition_coordinates.size[i]):
                            all_same_size = False
                    if not all_same_size:
                        unmergeable()
                        return False

                    merged_id = self.partition_coordinates.id[partition_idx]
                    merged_xy = self.partition_coordinates.xy[partition_idx]
                    merged_size = self.partition_coordinates.size[partition_idx] * 2.
                    merged_location_based_assignments = self.is_camera_in_partition[partition_idx]
                    merged_visibility_based_assignments = self.is_partitions_visible_to_cameras[partition_idx]

                    for i in [right_idx, up_idx, up_right_idx]:
                        merged_location_based_assignments = merged_location_based_assignments | self.is_camera_in_partition[i]
                        merged_visibility_based_assignments = merged_visibility_based_assignments | self.is_partitions_visible_to_cameras[i]

                    merged_visibility_based_assignments = merged_visibility_based_assignments & ~merged_location_based_assignments

                    partition_ids.append(merged_id)
                    partition_xys.append(merged_xy)
                    partition_sizes.append(merged_size)
                    is_camera_in_partition.append(merged_location_based_assignments)
                    is_partitions_visible_to_cameras.append(merged_visibility_based_assignments)

                    # mark as merged
                    has_merged[rowIdx, colIdx] = True
                    has_merged[rowIdx, colIdx + 1] = True
                    has_merged[rowIdx + 1, colIdx] = True
                    has_merged[rowIdx + 1, colIdx + 1] = True

                    return True

                def merge_single(merge_with):
                    merge_with_partition_idx = get_partition_idx(*merge_with)
                    if torch.any(self.partition_coordinates.size[partition_idx] != self.partition_coordinates.size[merge_with_partition_idx]):
                        unmergeable()
                        return False

                    merged_id = torch.minimum(self.partition_coordinates.id[partition_idx], self.partition_coordinates.id[merge_with_partition_idx])
                    merged_xy = torch.minimum(self.partition_coordinates.xy[partition_idx], self.partition_coordinates.xy[merge_with_partition_idx])
                    merged_size = self.partition_coordinates.size[partition_idx] * (1. + ((self.partition_coordinates.id[partition_idx] - self.partition_coordinates.id[merge_with_partition_idx]) != 0))

                    merged_location_based_assignments = self.is_camera_in_partition[partition_idx] | self.is_camera_in_partition[merge_with_partition_idx]
                    merged_visibility_based_assignments = self.is_partitions_visible_to_cameras[partition_idx] | self.is_partitions_visible_to_cameras[merge_with_partition_idx]

                    merged_visibility_based_assignments = merged_visibility_based_assignments & ~merged_location_based_assignments

                    partition_ids.append(merged_id)
                    partition_xys.append(merged_xy)
                    partition_sizes.append(merged_size)
                    is_camera_in_partition.append(merged_location_based_assignments)
                    is_partitions_visible_to_cameras.append(merged_visibility_based_assignments)

                    has_merged[merge_with[0], merge_with[1]] = True

                    return True

                # current location has location based, simply add to merged list
                if n_location_based_assignments[rowIdx, colIdx] >= min_location_based:
                    if n_partitions_have_location_based == 0 and n_mergeable == 3:
                        merge_three_neighbors()
                    else:
                        if right_mergeable and right_n_location_based_cameras < min_location_based:
                            merge_single((rowIdx, colIdx + 1))
                        elif up_mergeable and up_n_location_based_cameras < min_location_based:
                            merge_single((rowIdx + 1, colIdx))
                        else:
                            unmergeable()
                    continue

                """
                When is a mergeable partition: unmerged && exists

                Cases:
                    if all of the three neighbors are mergeable
                        if n_has_location_based_partitions <= 1, then merge them
                        else
                            if both on the right, then merge with up
                            else if both on up, then merge with left
                            else merge into the least number of cameras one
                    if right is unmergeable, then only merge into the up is possible
                        if up is mergeable, merge them
                        else unmergeable()
                    if up is unmergeable, then only merge into the right is possible
                        if right is mergeable, merge them
                        else unmergeable()
                """

                # Merge 3 neighbors
                if right_mergeable and up_mergeable and up_right_mergeable and n_partitions_have_location_based <= 1:
                    merge_three_neighbors()
                    continue
                elif right_mergeable or up_mergeable:
                    merge_with = None
                    if n_partitions_have_location_based - int(up_right_n_location_based_cameras >= min_location_based) > 0:
                        # neighbor has location based
                        if up_n_location_based_cameras > right_n_location_based_cameras:
                            merge_with = (rowIdx + 1, colIdx)
                        else:
                            merge_with = (rowIdx, colIdx + 1)
                    else:
                        # neighbor does not have location based
                        if up_mergeable:
                            merge_with = (rowIdx + 1, colIdx)
                        elif right_mergeable:
                            merge_with = (rowIdx, colIdx + 1)

                    if merge_with is None:
                        unmergeable()
                        continue

                    merge_single(merge_with)

                else:
                    unmergeable()

        self.partition_coordinates = PartitionCoordinates(
            torch.stack(partition_ids),
            torch.stack(partition_xys),
            torch.stack(partition_sizes)
        )
        self.is_camera_in_partition = torch.stack(is_camera_in_partition)
        self.is_partitions_visible_to_cameras = torch.stack(is_partitions_visible_to_cameras)

        return self.partition_coordinates, self.is_camera_in_partition, self.is_partitions_visible_to_cameras

    def manual_merge(self, merge_list: list):
        self.pre_merge()

        merged_ids_list = []
        merged_xys_list = []
        merged_sizes_list = []
        merged_location_based_assignments_list = []
        merged_visibility_based_assignments_list = []

        area_before_merging = torch.sum(torch.prod(self.partition_coordinates.size, dim=-1))

        merged_partition_idx_set = {}
        for partition_idx_list in merge_list:
            ids_list = []
            xys_list = []
            size_list = []
            merged_location_based_assignments = torch.zeros_like(self.is_camera_in_partition[0], dtype=torch.bool)
            merged_visibility_based_assignments = torch.zeros_like(self.is_partitions_visible_to_cameras[0], dtype=torch.bool)

            for partition_idx in partition_idx_list:
                assert partition_idx not in merged_partition_idx_set, partition_idx
                merged_partition_idx_set[partition_idx] = True
                ids_list.append(self.partition_coordinates.id[partition_idx])
                xys_list.append(self.partition_coordinates.xy[partition_idx])
                size_list.append(self.partition_coordinates.size[partition_idx])
                merged_location_based_assignments |= self.is_camera_in_partition[partition_idx]
                merged_visibility_based_assignments |= self.is_partitions_visible_to_cameras[partition_idx]

            ids = torch.stack(ids_list)
            xys = torch.stack(xys_list)
            sizes = torch.stack(size_list)

            area_of_pending_merging_partitions = torch.sum(torch.prod(sizes, dim=-1))

            merged_id = torch.min(ids, dim=0).values
            merged_xy = torch.min(xys, dim=0).values
            merged_size = ((torch.max(ids, dim=0).values - merged_id) + 1) * self.scene_config.partition_size

            area_of_merged_partition = torch.prod(merged_size)

            assert torch.abs(area_of_pending_merging_partitions - area_of_merged_partition) < 1e-2, (area_of_pending_merging_partitions, area_of_merged_partition, partition_idx_list, merged_id, torch.max(ids, dim=0).values, sizes)

            merged_visibility_based_assignments &= ~merged_location_based_assignments

            merged_ids_list.append(merged_id)
            merged_xys_list.append(merged_xy)
            merged_sizes_list.append(merged_size)
            merged_location_based_assignments_list.append(merged_location_based_assignments)
            merged_visibility_based_assignments_list.append(merged_visibility_based_assignments)

        for partition_idx in range(len(self.partition_coordinates)):
            if partition_idx in merged_partition_idx_set:
                continue
            merged_ids_list.append(self.partition_coordinates.id[partition_idx])
            merged_xys_list.append(self.partition_coordinates.xy[partition_idx])
            merged_sizes_list.append(self.partition_coordinates.size[partition_idx])
            merged_location_based_assignments_list.append(self.is_camera_in_partition[partition_idx])
            merged_visibility_based_assignments_list.append(self.is_partitions_visible_to_cameras[partition_idx])

        new_partition_coordinates = PartitionCoordinates(
            id=torch.stack(merged_ids_list),
            xy=torch.stack(merged_xys_list),
            size=torch.stack(merged_sizes_list),
        )
        assert torch.abs(torch.sum(torch.prod(new_partition_coordinates.size, dim=-1)) - area_before_merging) < 1e-4

        self.partition_coordinates = new_partition_coordinates
        self.is_camera_in_partition = torch.stack(merged_location_based_assignments_list)
        self.is_partitions_visible_to_cameras = torch.stack(merged_visibility_based_assignments_list)

    def pre_merge(self):
        self.unmerge()
        self.save_unmerged()

    def save_unmerged(self):
        assert self.unmerged is None, "Please call `unmerge()` first"
        self.unmerged = (
            self.partition_coordinates,
            self.is_camera_in_partition,
            self.is_partitions_visible_to_cameras,
        )

    def unmerge(self):
        if self.unmerged is None:
            return

        # restore
        self.partition_coordinates, self.is_camera_in_partition, self.is_partitions_visible_to_cameras = self.unmerged
        self.unmerged = None

    def build_output_dirname(self):
        return "partitions-size_{}-enlarge_{}-{}_visibility_{}_{}".format(
            self.scene_config.partition_size,
            self.scene_config.location_based_enlarge,
            "convex_hull" if self.scene_config.convex_hull_based_visibility else "point",
            self.scene_config.visibility_based_distance,
            round(self.scene_config.visibility_threshold, 2),
        )

    def save(self, output_dir: str, extra_data=None):
        return Partitioning.save_partitions(
            output_dir,
            scene_config=self.scene_config,
            scene_bounding_box=self.scene_bounding_box,
            partition_coordinates=self.partition_coordinates,
            visibilities=self.camera_visibilities,
            location_based_assignments=self.is_camera_in_partition,
            visibility_based_assignments=self.is_partitions_visible_to_cameras,
            unmerged=self.unmerged,
            extra_data=extra_data,
        )

    def set_plot_ax_limit(self, ax, plot_enlarge: float = 0.25):
        ax.set_xlim([
            self.scene_bounding_box.bounding_box.min[0] - plot_enlarge * self.scene_config.partition_size,
            self.scene_bounding_box.bounding_box.max[0] + plot_enlarge * self.scene_config.partition_size,
        ])
        ax.set_ylim([
            self.scene_bounding_box.bounding_box.min[1] - plot_enlarge * self.scene_config.partition_size,
            self.scene_bounding_box.bounding_box.max[1] + plot_enlarge * self.scene_config.partition_size,
        ])

    def plot_scene_bounding_box(self, ax):
        ax.set_aspect('equal', adjustable='box')

        ax.scatter(self.camera_centers[:, 0], self.camera_centers[:, 1], s=0.2)
        ax.add_artist(mpatches.Rectangle(
            self.scene_bounding_box.bounding_box.min.tolist(),
            self.scene_bounding_box.bounding_box.max[0] - self.scene_bounding_box.bounding_box.min[0],
            self.scene_bounding_box.bounding_box.max[1] - self.scene_bounding_box.bounding_box.min[1],
            fill=False,
            color="green",
        ))
        self.set_plot_ax_limit(ax)

    def plot_partitions(self, ax=None, annotate_font_size: int = 5, annotate_position_x: float = 0.125, annotate_position_y: float = 0.25, plot_cameras: bool = True):
        self.set_plot_ax_limit(ax)
        ax.set_aspect('equal', adjustable='box')

        colors = list(iter(cm.rainbow(np.linspace(0, 1, len(self.partition_coordinates)))))
        # random.shuffle(colors)
        if plot_cameras:
            ax.scatter(self.camera_centers[:, 0], self.camera_centers[:, 1], s=0.2)

        color_iter = iter(colors)
        idx = 0
        for partition_id, partition_xy, partition_size in self.partition_coordinates:
            ax.add_artist(mpatches.Rectangle(
                (partition_xy[0], partition_xy[1]),
                partition_size[0],
                partition_size[1],
                fill=False,
                color=next(color_iter),
            ))
            ax.annotate(
                "#{}\n({}, {})".format(idx, partition_id[0], partition_id[1]),
                xy=(
                    partition_xy[0] + annotate_position_x * partition_size[0].item(),
                    partition_xy[1] + annotate_position_y * partition_size[1].item(),
                ),
                fontsize=annotate_font_size,
            )
            idx += 1

    def plot_partition_assigned_cameras(self, ax, partition_idx: int, point_xyzs: torch.Tensor = None, point_rgbs: torch.Tensor = None, point_size: float = 0.1, point_sparsify: int = 8):
        location_base_assignment_bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            enlarge=self.scene_config.location_based_enlarge,
        )
        location_base_assignment_bounding_box_sizes = location_base_assignment_bounding_boxes.max - location_base_assignment_bounding_boxes.min

        visibility_base_assignment_bounding_boxes = self.partition_coordinates.get_bounding_boxes(
            enlarge=self.scene_config.visibility_based_distance,
        )
        visibility_base_assignment_bounding_box_size = visibility_base_assignment_bounding_boxes.max - visibility_base_assignment_bounding_boxes.min

        location_based_assignment = self.is_camera_in_partition[partition_idx]
        visibility_based_assignment = self.is_partitions_visible_to_cameras[partition_idx]
        n_assigned_cameras = (location_based_assignment | visibility_based_assignment).sum().item()

        ax.set_aspect('equal', adjustable='box')
        if point_xyzs is not None:
            ax.scatter(point_xyzs[::point_sparsify, 0], point_xyzs[::point_sparsify, 1], c=point_rgbs[::point_sparsify] / 255., s=point_size)

        # plot original partition bounding box
        ax.add_artist(mpatches.Rectangle(
            self.partition_coordinates.xy[partition_idx],
            self.partition_coordinates.size[partition_idx][0],
            self.partition_coordinates.size[partition_idx][1],
            fill=False,
            color="green",
        ))

        # plot location based assignment bounding box
        ax.add_artist(mpatches.Rectangle(
            location_base_assignment_bounding_boxes.min[partition_idx],
            location_base_assignment_bounding_box_sizes[partition_idx, 0].item(),
            location_base_assignment_bounding_box_sizes[partition_idx, 1].item(),
            fill=False,
            color="purple",
        ))

        # plot visibility based assignment bounding box
        ax.add_artist(mpatches.Rectangle(
            visibility_base_assignment_bounding_boxes.min[partition_idx],
            visibility_base_assignment_bounding_box_size[partition_idx, 0].item(),
            visibility_base_assignment_bounding_box_size[partition_idx, 1].item(),
            fill=False,
            color="yellow",
        ))

        # plot assigned cameras
        ax.scatter(self.camera_centers[location_based_assignment.numpy(), 0],
                   self.camera_centers[location_based_assignment.numpy(), 1], s=0.2, c="blue")
        ax.scatter(self.camera_centers[visibility_based_assignment.numpy(), 0],
                   self.camera_centers[visibility_based_assignment.numpy(), 1], s=0.2, c="red")

        ax.annotate(
            text="#{} ({}, {}) - {}".format(
                partition_idx,
                self.partition_coordinates.id[partition_idx, 0].item(),
                self.partition_coordinates.id[partition_idx, 1].item(),
                n_assigned_cameras,
            ),
            xy=self.partition_coordinates.xy[partition_idx] + 0.05 * self.scene_config.partition_size,
            color="orange",
        )

        self.set_plot_ax_limit(ax)

    def plot(self, func: Callable, *args, **kwargs):
        plt.close()
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.show(fig)

    def save_plot(self, func: Callable, path: str, *args, **kwargs):
        plt.close()
        fig, ax = plt.subplots()
        func(ax, *args, **kwargs)
        plt.savefig(path, dpi=600)
        plt.show(fig)


class Partitioning:
    @staticmethod
    def get_bounding_box_by_camera_centers(camera_centers: torch.Tensor, enlarge: float = 0.) -> MinMaxBoundingBox:
        xyz_min = torch.min(camera_centers, dim=0)[0]
        xyz_max = torch.max(camera_centers, dim=0)[0]

        size = xyz_max - xyz_min

        xyz_min -= size * enlarge
        xyz_max += size * enlarge

        return MinMaxBoundingBox(
            min=xyz_min,
            max=xyz_max,
        )

    @staticmethod
    def get_bounding_box_by_points(points: torch.Tensor, enlarge: float = 0., outlier_threshold: float = 0.001) -> MinMaxBoundingBox:
        xyz_min = torch.quantile(points, outlier_threshold, dim=0)
        xyz_max = torch.quantile(points, 1. - outlier_threshold, dim=0)

        if enlarge > 0.:
            size = xyz_max - xyz_min
            enlarge_size = size * enlarge
            xyz_min -= enlarge_size
            xyz_max += enlarge_size

        return MinMaxBoundingBox(
            min=xyz_min[..., :2],
            max=xyz_max[..., :2],
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
    def build_partition_coordinates(scene_bounding_box: SceneBoundingBox, origin: torch.Tensor, size: float):
        partition_count = scene_bounding_box.n_partitions
        origin_partition_offset = scene_bounding_box.origin_partition_offset
        grid_x, grid_y = torch.meshgrid(
            torch.arange(partition_count[0], dtype=torch.int) + origin_partition_offset[0],
            torch.arange(partition_count[1], dtype=torch.int) + origin_partition_offset[1],
            indexing="xy",
        )

        partition_id = torch.dstack([grid_x, grid_y])
        partition_xy = partition_id * size + origin

        return PartitionCoordinates(
            id=partition_id.reshape(torch.prod(partition_count), 2),
            xy=partition_xy.reshape(torch.prod(partition_count), 2),
            size=torch.tensor([[size, size]], dtype=torch.float).repeat(torch.prod(partition_count), 1)
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
            enlarge: Union[float, torch.Tensor] = 0.1,
    ) -> torch.Tensor:
        if isinstance(enlarge, torch.Tensor):
            assert torch.all(enlarge >= 0.)
        else:
            assert enlarge >= 0.

        bounding_boxes = partition_coordinates.get_bounding_boxes(
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
            n_cameras,
            point_getter: Callable[[int], torch.Tensor],
            device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(enlarge=0.).to(device=device)
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
    def cameras_convex_hull_based_visibilities_calculation(
            cls,
            partition_coordinates: PartitionCoordinates,
            n_cameras,
            point_getter: Callable[[int], Tuple[torch.Tensor, torch.Tensor, int]],
            enlarge: float,
            device,
    ):
        partition_bounding_boxes = partition_coordinates.get_bounding_boxes(enlarge=enlarge).to(device=device)
        all_visibilities = torch.ones((n_cameras, len(partition_coordinates)), device="cpu") * -255.  # [N_cameras, N_partitions]

        def calculate_visibilities(camera_idx: int):
            points_2d, points_3d, projected_points = point_getter(camera_idx)
            visibilities, _, _ = Partitioning.calculate_convex_hull_based_visibilities(
                partition_bounding_boxes=partition_bounding_boxes,
                points_2d=points_2d,
                points_3d=points_3d[..., :2],
                projected_points=projected_points,
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
            max_distance: float,
            assigned_mask: torch.Tensor,  # the Tensor produced by `camera_center_based_partition_assignment` above
            visibilities: torch.Tensor,  # [N_partitions, N_cameras]
            visibility_threshold: float,
            no_camera_enlarge_distance: float = 2.,
            no_camera_reduce_threshold: float = 4.,
    ):
        # increase the distance, and lower the threshold, if a partition does not have location based camera assignments
        assert no_camera_enlarge_distance >= 1.
        assert no_camera_reduce_threshold >= 1.

        have_assigned_cameras = torch.sum(assigned_mask, dim=-1, keepdim=True) > 0
        no_assigned_cameras = ~have_assigned_cameras
        have_assigned_cameras = have_assigned_cameras.to(dtype=visibilities.dtype)
        no_assigned_cameras = no_assigned_cameras.to(dtype=visibilities.dtype)

        max_distance_adjustments = have_assigned_cameras + no_camera_enlarge_distance * no_assigned_cameras
        visibility_adjustments = have_assigned_cameras + ((1. / no_camera_reduce_threshold) * no_assigned_cameras)

        max_distances = torch.tensor([[max_distance]], dtype=camera_centers.dtype) * max_distance_adjustments
        is_in_range = cls.camera_center_based_partition_assignment(
            partition_coordinates,
            camera_centers,
            max_distances,
        )  # [N_partitions, N_cameras]

        visibility_threshold = torch.tensor(
            [[visibility_threshold]],
            dtype=visibilities.dtype,
            device=visibilities.device,
        ).repeat(visibilities.shape[0], 1)  # [N_partitions, 1]
        visibility_threshold *= visibility_adjustments

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
    def calculate_convex_hull_based_visibilities(
            cls,
            partition_bounding_boxes: MinMaxBoundingBoxes,
            points_2d: torch.Tensor,  # [N_points, 2]
            points_3d: torch.Tensor,  # [N_points, 2 or 3]
            projected_points: torch.Tensor,  # [N_points, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert projected_points.shape[-1] == 2

        visibilities = torch.zeros((partition_bounding_boxes.min.shape[0],), dtype=torch.float)
        scene_convex_hull = None
        partition_convex_hull_list = []
        is_in_bounding_boxes = None
        if points_3d.shape[0] > 2:
            try:
                scene_convex_hull = ConvexHull(projected_points)
            except:
                return visibilities, scene_convex_hull, partition_convex_hull_list
            scene_area = scene_convex_hull.volume

            is_in_bounding_boxes = cls.is_in_bounding_boxes(
                bounding_boxes=partition_bounding_boxes,
                coordinates=points_3d,
            )  # [N_partitions, N_points]

            # TODO: batchify
            for partition_idx in range(is_in_bounding_boxes.shape[0]):
                if is_in_bounding_boxes[partition_idx].sum() < 3:
                    partition_convex_hull_list.append(None)
                    continue
                points_2d_in_partition = points_2d[is_in_bounding_boxes[partition_idx]]
                try:
                    partition_convex_hull = ConvexHull(points_2d_in_partition)
                except Exception as e:
                    partition_convex_hull_list.append(None)
                    continue
                partition_convex_hull_list.append(partition_convex_hull)
                partition_area = partition_convex_hull.volume

                visibilities[partition_idx] = partition_area / scene_area

        return visibilities, scene_convex_hull, (partition_convex_hull_list, is_in_bounding_boxes)

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
            unmerged: torch.Tensor,
            extra_data=None,
    ):
        os.makedirs(dir, exist_ok=True)
        save_to = os.path.join(dir, "partitions.pt")
        torch.save(
            {
                "scene_config": dataclasses.asdict(scene_config),
                "scene_bounding_box": dataclasses.asdict(scene_bounding_box),
                "partition_coordinates": dataclasses.asdict(partition_coordinates),
                "visibilities": visibilities,
                "location_based_assignments": location_based_assignments,
                "visibility_based_assignments": visibility_based_assignments,
                "unmerged": unmerged,
                "extra_data": extra_data,
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

from dataclasses import dataclass
import os
import torch
from .vanilla_store import VanillaStore, VanillaStoreModule


@dataclass
class PartitionStore(VanillaStore):
    partition: str = None
    """Partition data directory"""

    partition_idx: int = None

    def instantiate(self, *args, **kwargs):
        return PartitionStoreModule(self)


class PartitionStoreModule(VanillaStoreModule, torch.nn.Module):
    def setup(self, stage, pl_module):
        # load partition data
        partition_data = torch.load(os.path.join(
            self.config.partition,
            "partitions.pt",
        ))

        default_partition_size = partition_data["scene_config"]["partition_size"]
        self.register_buffer(
            "default_partition_size",
            torch.tensor(default_partition_size, dtype=torch.float),
            persistent=False,
        )
        partition_size = partition_data["partition_coordinates"]["size"][self.config.partition_idx]
        self.partition_size = partition_size

        # get transform matrix
        try:
            rotation_transform = partition_data["extra_data"]["rotation_transform"]
        except:
            print("FFDensityController: No orientation transform")
            rotation_transform = torch.eye(4, dtype=torch.float, device=pl_module.device)
        self.register_buffer(
            "rotation_transform",
            rotation_transform,
            persistent=False,
        )

        # get bounding box
        partition_bbox_min = partition_data["partition_coordinates"]["xy"][self.config.partition_idx]
        partition_bbox_max = partition_bbox_min + partition_size

        self.register_buffer("partition_bbox_min", partition_bbox_min, persistent=False)
        self.register_buffer("partition_bbox_max", partition_bbox_max, persistent=False)

        print("partition_idx=#{}, id={}, transform={}, default_size={}, partition_size={}, bbox=(\n  {}, \n  {}\n)".format(
            self.config.partition_idx,
            partition_data["partition_coordinates"]["id"][self.config.partition_idx],
            self.rotation_transform.tolist(),
            default_partition_size,
            partition_size.tolist(),
            self.partition_bbox_min.tolist(),
            self.partition_bbox_max.tolist(),
        ))

        # initialization
        self.calculate_distances(
            None,
            None,
            gaussian_model=pl_module.gaussian_model,
            global_step=None,
            pl_module=None,
        )

        # setup hook
        pl_module.on_train_batch_end_hooks.append(self.calculate_distances)

    @torch.no_grad()
    def _get_normalized_distance_to_bounding_box(self, gaussian_model):
        # transform 3D means
        transformed_means = gaussian_model.get_means() @ self.rotation_transform[:2, :3].T + self.rotation_transform[:2, 3]  # [N, 2]

        dist_min2p = self.partition_bbox_min - transformed_means
        dist_p2max = transformed_means - self.partition_bbox_max
        dxy = torch.maximum(dist_min2p, dist_p2max)
        distances = torch.sqrt(torch.pow(dxy.clamp(min=0.), 2).sum(dim=-1))
        return distances / self.default_partition_size, transformed_means, distances  # [N]

    @torch.no_grad()
    def calculate_distances(self, outputs, batch, gaussian_model, global_step, pl_module):
        self.distance_factors, self.transformed_means, self.distances = self._get_normalized_distance_to_bounding_box(gaussian_model)

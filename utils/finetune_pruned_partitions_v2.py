import add_pypath
import os
import argparse
from trained_partition_utils import PartitionTraining


class PartitionFinetuning(PartitionTraining):
    def __init__(self, path: str, trained_project: str, prune_percent: float, name: str = "partitions.pt"):
        super().__init__(path, name)
        self.trained_project = trained_project
        self.prune_percent = prune_percent

    def train_a_partition(
            self,
            partition_image_number: int,
            extra_epoches: int,
            scalable_params: dict[str, int],
            extra_epoch_scalable_params: list[str],
            project_output_dir: str,
            partition_idx: int,
            project_name: str,
            partition_id_str: str,
            config_file: str,
            extra_training_args: tuple,
            dry_run: bool,
    ):
        assert config_file is None, "--config must not be specified for finetuning"
        trained_project_name = self.trained_project

        config_file = os.path.join("outputs", trained_project_name, partition_id_str, "config.yaml")

        extra_training_args = tuple(list(extra_training_args) + ["--ckpt_path={}".format(os.path.join(
            "outputs",
            trained_project_name,
            partition_id_str,
            "pruned_checkpoints",
            "latest-opacity_pruned-{}.ckpt".format(self.prune_percent)
        ))])

        super().train_a_partition(
            partition_image_number,
            extra_epoches,
            scalable_params,
            extra_epoch_scalable_params,
            project_output_dir,
            partition_idx,
            project_name,
            partition_id_str,
            config_file,
            extra_training_args,
            dry_run,
        )

    @classmethod
    def instantiate_with_args(cls, args):
        return cls(
            path=args.partition_dir,
            trained_project=args.trained_project,
            prune_percent=args.prune_percent,
        )


def main():
    parser = argparse.ArgumentParser()
    PartitionTraining.configure_argparser(parser, extra_epoches=30)
    parser.add_argument("--prune-percent", type=float, default=0.6)
    parser.add_argument("--trained-project", "-t", type=str, required=True)

    PartitionFinetuning.start_with_configured_argparser(parser)


main()

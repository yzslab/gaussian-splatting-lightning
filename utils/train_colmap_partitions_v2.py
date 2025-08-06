import os
from dataclasses import dataclass
from train_partitions import PartitionTrainingConfig, PartitionTraining
import argparse
import json


@dataclass
class ColmapPartitionTrainingConfig(PartitionTrainingConfig):
    eval: bool = False

    extra_points_dir: str = None

    ref_cap_max: dict = None

    ref_cap_max_factor: int = 1

    @classmethod
    def get_extra_init_kwargs(cls, args):
        return {
            "eval": args.eval,
            "extra_points_dir": args.extra_points_dir,
            "ref_cap_max": args.ref_cap_max,
            "ref_cap_max_factor": args.ref_cap_max_factor,
        }

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        PartitionTrainingConfig.configure_argparser(parser, extra_epoches)
        parser.add_argument("--eval", action="store_true", default=False)
        parser.add_argument("--extra-points-dir", type=str, default=None)
        parser.add_argument("--ref-cap-max", type=str, default=None)
        parser.add_argument("--ref-cap-max-factor", type=int, default=1)


class ColmapPartitionTraining(PartitionTraining):
    def get_default_dataparser_name(self) -> str:
        return "Colmap"

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        extra_args = []
        if self.config.extra_points_dir is not None:
            extra_point_file_path = os.path.join(self.config.extra_points_dir, "{}.ply".format(self.get_partition_id_str(partition_idx)))
            if os.path.exists(extra_point_file_path):
                extra_args.append("--data.extra_points={}".format(extra_point_file_path))
            else:
                from tqdm.auto import tqdm
                tqdm.write("[WARNING]{} not found".format(extra_point_file_path))
        
        if self.config.ref_cap_max is not None:
            with open(self.config.ref_cap_max, "r") as f:
                ref_cap_max = json.load(f)
            partition_id_str = self.get_partition_id_str(partition_idx)
            cap_max = ref_cap_max[partition_id_str] // self.config.ref_cap_max_factor
            extra_args += [
                "--model.density.cap_max={}".format(cap_max),
                "--data.parser.n_max_points={}".format(cap_max),
            ]

        return [
            "--data.parser.image_list={}".format(os.path.join(
                self.path,
                "{}.txt".format(self.get_partition_id_str(partition_idx)),
            )),
            "--data.parser.split_mode={}".format("experiment" if self.config.eval else "reconstruction"),
            "--data.parser.eval_step=64",
        ] + extra_args


def main():
    parser = argparse.ArgumentParser()
    ColmapPartitionTrainingConfig.configure_argparser(parser)
    ColmapPartitionTraining.start_with_configured_argparser(parser, config_cls=ColmapPartitionTrainingConfig)


main()

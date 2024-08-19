import os
from train_partitions import PartitionTraining
import argparse


class ColmapPartitionTraining(PartitionTraining):
    def get_default_dataparser_name(self) -> str:
        return "Colmap"

    def get_dataset_specified_args(self, partition_idx: int) -> list[str]:
        return [
            "--data.parser.image_list={}".format(os.path.join(
                self.path,
                "{}.txt".format(self.get_partition_id_str(partition_idx)),
            )),
            "--data.parser.split_mode=reconstruction",
            "--data.parser.eval_step=64",
        ]


def main():
    parser = argparse.ArgumentParser()
    ColmapPartitionTraining.configure_argparser(parser)
    ColmapPartitionTraining.start_with_configured_argparser(parser)


main()

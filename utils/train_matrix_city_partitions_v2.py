import os
from train_partitions import PartitionTraining
import argparse


class MatrixCityPartitionTraining(PartitionTraining):
    def get_default_dataparser_name(self) -> str:
        return "MatrixCity"

    def get_dataset_specified_args(self, partition_idx: int) -> list[str]:
        return [
            "--data.parser.train={}".format([os.path.join(
                self.path,
                "partition-{}.json".format(self.get_partition_id_str(partition_idx)),
            )]),
            "--data.parser.test={}".format([os.path.join(
                self.path,
                "partition-{}-test.json".format(self.get_partition_id_str(partition_idx)),
            )]),
        ]


def main():
    parser = argparse.ArgumentParser()
    MatrixCityPartitionTraining.configure_argparser(parser)
    MatrixCityPartitionTraining.start_with_configured_argparser(parser)


main()

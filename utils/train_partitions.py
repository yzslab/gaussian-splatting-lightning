import add_pypath
import os
import torch
import subprocess
from internal.utils.partitioning_utils import PartitionCoordinates
from distibuted_tasks import get_task_list
from auto_hyper_parameter import auto_hyper_parameter, to_command_args, SCALABEL_PARAMS, EXTRA_EPOCH_SCALABLE_STEP_PARAMS
from argparser_utils import parser_stoppable_args
from distibuted_tasks import configure_arg_parser_v2


class PartitionTraining:
    def __init__(self, path: str, name: str = "partitions.pt"):
        self.path = path
        self.scene = torch.load(os.path.join(path, name), map_location="cpu")
        self.scene["partition_coordinates"] = PartitionCoordinates(**self.scene["partition_coordinates"])
        self.dataset_path = os.path.dirname(path.rstrip("/"))

    @property
    def partition_coordinates(self) -> PartitionCoordinates:
        return self.scene["partition_coordinates"]

    def get_default_dataparser_name(self) -> str:
        raise NotImplementedError()

    def get_dataset_specified_args(self, partition_idx: int) -> list[str]:
        raise NotImplementedError()

    def get_location_based_assignment_numbers(self) -> torch.Tensor:
        return self.scene["location_based_assignments"].sum(-1)

    def get_partition_id_str(self, idx: int) -> str:
        return self.partition_coordinates.get_str_id(idx)

    def get_image_numbers(self) -> torch.Tensor:
        return torch.logical_or(
            self.scene["location_based_assignments"],
            self.scene["visibility_based_assignments"],
        ).sum(-1)

    def get_trainable_partition_idx_list(
            self,
            min_images: int,
            n_processes: int,
            process_id: int,
    ) -> list[int]:
        location_based_numbers = self.get_location_based_assignment_numbers()
        all_trainable_partition_indices = torch.ge(location_based_numbers, min_images).nonzero().squeeze(-1).tolist()
        return get_task_list(n_processors=n_processes, current_processor_id=process_id, all_tasks=all_trainable_partition_indices)

    def get_partition_trained_step_filename(self, partition_idx: int):
        return "{}_trained".format(self.get_partition_id_str(partition_idx))

    def train_partitions(
            self,
            project_name: str,
            min_images: int = 32,
            n_processes: int = 1,
            process_id: int = 1,
            dry_run: bool = False,
            extra_epoches: int = 0,
            scalable_params: dict[str, int] = {},
            extra_epoch_scalable_params: list[str] = [],
            partition_id_strs: list[str] = None,
            config_file: str = None,
            extra_training_args: list[str] = [],
    ):
        project_output_dir = os.path.join("outputs", project_name)

        partition_image_numbers = self.get_image_numbers()

        for partition_idx in self.get_trainable_partition_idx_list(
                min_images=min_images,
                n_processes=n_processes,
                process_id=process_id,
        ):
            partition_id_str = self.get_partition_id_str(partition_idx)
            if partition_id_strs is not None and partition_id_str not in partition_id_strs:
                continue

            # scale hyper parameters
            max_steps, scaled_params, scale_up = auto_hyper_parameter(
                partition_image_numbers[partition_idx].item(),
                extra_epoch=extra_epoches,
                scalable_params=scalable_params,
                extra_epoch_scalable_params=extra_epoch_scalable_params,
            )

            # whether a trained partition
            partition_trained_step_file_path = os.path.join(
                project_output_dir,
                self.get_partition_trained_step_filename(partition_idx)
            )

            try:
                with open(partition_trained_step_file_path, "r") as f:
                    trained_steps = int(f.read())
                    if trained_steps >= max_steps:
                        print("Skip trained partition '{}'".format(self.partition_coordinates.id[partition_idx].tolist()))
                        continue
            except:
                pass

            # build args
            args = [
                "python",
                "main.py", "fit",
                "--output", project_output_dir,
                "--project", project_name,
                "--logger", "wandb",
                "-n={}".format(partition_id_str),
                "--data.path", self.dataset_path,
                "--data.parser", self.get_default_dataparser_name(),  # can be overridden by config file or later args
            ]
            if config_file is not None:
                args.append("--config={}".format(config_file))
            args += extra_training_args
            args += self.get_dataset_specified_args(partition_idx)
            args += to_command_args(max_steps, scaled_params)

            if dry_run:
                print(args)
            else:
                ret_code = subprocess.call(args)
                if ret_code == 0:
                    with open(partition_trained_step_file_path, "w") as f:
                        f.write("{}".format(max_steps))

    @staticmethod
    def configure_argparser(parser):
        parser.add_argument("partition_dir")
        parser.add_argument("--project", "-p", type=str, required=True,
                            help="Project name")
        parser.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        parser.add_argument("--config", "-c", type=str, default=None)
        parser.add_argument("--parts", default=None, nargs="*", action="extend")
        parser.add_argument("--extra-epoches", "-e", type=int, default=0)
        parser.add_argument("--scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--extra-epoch-scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--no-default-scalable", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        configure_arg_parser_v2(parser)

    @classmethod
    def start_with_configured_argparser(cls, parser):
        args, training_args = parser_stoppable_args(parser)

        partition_training = cls(
            args.partition_dir,
        )

        # parse scalable params
        scalable_params = SCALABEL_PARAMS
        extra_epoch_scalable_params = EXTRA_EPOCH_SCALABLE_STEP_PARAMS
        if args.no_default_scalable:
            scalable_params = {}
            extra_epoch_scalable_params = []
        for i in args.scalable_params:
            name, value = i.split("=", 1)
            value = int(value)
            scalable_params[name] = value
        extra_epoch_scalable_params += args.extra_epoch_scalable_params

        # start training
        partition_training.train_partitions(
            project_name=args.project,
            min_images=args.min_images,
            n_processes=args.n_processes,
            process_id=args.process_id,
            dry_run=args.dry_run,
            extra_epoches=args.extra_epoches,
            scalable_params=scalable_params,
            extra_epoch_scalable_params=extra_epoch_scalable_params,
            partition_id_strs=args.parts,
            config_file=args.config,
            extra_training_args=training_args,
        )

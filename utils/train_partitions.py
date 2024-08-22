import add_pypath
import os
import traceback
import yaml
import torch
import subprocess
from tqdm.auto import tqdm
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
        return "{}-trained".format(self.get_partition_id_str(partition_idx))

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
        # scale hyper parameters
        max_steps, scaled_params, scale_up = auto_hyper_parameter(
            partition_image_number,
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
                    return
        except:
            pass

        # build args
        # basic
        args = [
            "python",
            "main.py", "fit",
        ]
        # dataparser
        try:
            args += ["--data.parser", self.get_default_dataparser_name()]  # can be overridden by config file or the args later
        except NotImplementedError:
            pass

        # config file
        if config_file is not None:
            args.append("--config={}".format(config_file))

        # extra
        args += extra_training_args

        # dataset specified
        try:
            args += self.get_dataset_specified_args(partition_idx)
        except NotImplementedError:
            pass

        # scalable
        args += to_command_args(max_steps, scaled_params)

        args += [
            "-n={}".format(partition_id_str),
            "--data.path", self.dataset_path,
            "--project", project_name,
            "--output", project_output_dir,
            "--logger", "wandb",
        ]

        if dry_run:
            print(" ".join(args))
        else:
            try:
                print(args)
                ret_code = subprocess.call(args)
                if ret_code == 0:
                    with open(partition_trained_step_file_path, "w") as f:
                        f.write("{}".format(max_steps))
            except KeyboardInterrupt as e:
                raise e
            except:
                traceback.print_exc()

    def train_partitions(
            self,
            project_name: str,
            min_images: int = 32,
            n_processes: int = 1,
            process_id: int = 1,
            dry_run: bool = False,
            extra_epoches: int = 0,
            scalable_params=None,
            extra_epoch_scalable_params=None,
            partition_id_strs: list[str] = None,
            config_file: str = None,
            extra_training_args=None,
    ):
        if scalable_params is None:
            scalable_params = {}
        if extra_epoch_scalable_params is None:
            extra_epoch_scalable_params = []
        if extra_training_args is None:
            extra_training_args = []
        extra_training_args = tuple(extra_training_args)

        project_output_dir = os.path.join("outputs", project_name)

        partition_image_numbers = self.get_image_numbers()

        trainable_partition_idx_list = self.get_trainable_partition_idx_list(
            min_images=min_images,
            n_processes=n_processes,
            process_id=process_id,
        )
        print([self.get_partition_id_str(i) for i in trainable_partition_idx_list])

        with tqdm(trainable_partition_idx_list) as t:
            for partition_idx in t:
                partition_id_str = self.get_partition_id_str(partition_idx)
                t.set_description(partition_id_str)
                if partition_id_strs is not None and partition_id_str not in partition_id_strs:
                    continue

                self.train_a_partition(
                    partition_image_number=partition_image_numbers[partition_idx].item(),
                    extra_epoches=extra_epoches,
                    scalable_params=scalable_params,
                    extra_epoch_scalable_params=extra_epoch_scalable_params,
                    project_output_dir=project_output_dir,
                    partition_idx=partition_idx,
                    project_name=project_name,
                    partition_id_str=partition_id_str,
                    config_file=config_file,
                    extra_training_args=extra_training_args,
                    dry_run=dry_run,
                )

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        parser.add_argument("partition_dir")
        parser.add_argument("--project", "-p", type=str, required=True,
                            help="Project name")
        parser.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        parser.add_argument("--config", "-c", type=str, default=None)
        parser.add_argument("--parts", default=None, nargs="*", action="extend")
        parser.add_argument("--extra-epoches", "-e", type=int, default=extra_epoches)
        parser.add_argument("--scalable-config", type=str, default=None,
                            help="Load scalable params from a yaml file")
        parser.add_argument("--scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--extra-epoch-scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--no-default-scalable", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        configure_arg_parser_v2(parser)

    @classmethod
    def parse_scalable_params(cls, args):
        # parse scalable params
        scalable_params = SCALABEL_PARAMS
        extra_epoch_scalable_params = EXTRA_EPOCH_SCALABLE_STEP_PARAMS
        if args.no_default_scalable:
            scalable_params = {}
            extra_epoch_scalable_params = []

        if args.scalable_config is not None:
            with open(args.scalable_config, "r") as f:
                scalable_config = yaml.safe_load(f)

            for i in scalable_config.keys():
                if i not in ["scalable", "extra_epoch_scalable", "no_default"]:
                    raise ValueError("Found an unexpected key '{}' in '{}'".format(i, args.scalable_yaml))

            if scalable_config.get("no_default", False):
                scalable_params = {}
                extra_epoch_scalable_params = []

            scalable_params.update(scalable_config.get("scalable", {}))
            extra_epoch_scalable_params += scalable_config.get("extra_epoch_scalable", [])

        for i in args.scalable_params:
            name, value = i.split("=", 1)
            value = int(value)
            scalable_params[name] = value
        extra_epoch_scalable_params += args.extra_epoch_scalable_params

        return scalable_params, extra_epoch_scalable_params

    @classmethod
    def instantiate_with_args(cls, args):
        return cls(
            args.partition_dir,
        )

    @classmethod
    def start_with_configured_argparser(cls, parser):
        args, training_args = parser_stoppable_args(parser)

        partition_training = cls.instantiate_with_args(args)

        # parse scalable params
        scalable_params, extra_epoch_scalable_params = cls.parse_scalable_params(args)

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

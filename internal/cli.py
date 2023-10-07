import os.path
from jsonargparse import Namespace
from typing import Optional, Union, List, Literal
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import lightning.pytorch.loggers


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--iterations", "--iteration", "--steps", "--step", "--max_steps", type=Optional[int],
                            default=30_000)
        parser.add_argument("--epochs", "--epoch", "--max_epochs", type=Optional[int], default=None)
        parser.add_argument("--name", "-n", type=Optional[str], default=None,
                            help="the training result output path will be 'output/name'")
        parser.add_argument("--version", "-v", type=Optional[str], default=None,
                            help="the training result output path will be 'output/name/version'")
        # TODO: add max_steps to save_iterations, but need to compatible with --max_steps < 0 & --max_epochs > 0
        parser.add_argument("--save_iterations", type=List[int], default=[7_000, 30_000])
        parser.add_argument("--logger", type=str, default="tensorboard")
        parser.add_argument("--project", type=str, default="Gaussian-Splatting", help="WanDB project name")
        parser.add_argument("--output", type=str, default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
        ), help="the base directory of the output")

        parser.link_arguments("iterations", "trainer.max_steps")
        parser.link_arguments("epochs", "trainer.max_epochs")
        # parser.link_arguments("name", "logger.init_args.name")
        # parser.link_arguments("version", "logger.init_args.version")
        # parser.link_arguments("output", "logger.init_args.save_dir")
        # parser.link_arguments("output", "model.output_dir")
        # parser.link_arguments("logger", "trainer.logger", apply_on="instantiate")
        parser.link_arguments("save_iterations", "model.save_iterations")

    def before_instantiate_classes(self) -> None:
        config = getattr(self.config, self.config.subcommand)
        if config.name is None:
            # auto set experiment name base on --data.path
            config.name = "_".join(config.data.path.strip("/").split("/")[-3:])
            print("auto determine experiment name: {}".format(config.name))

        # build output path
        output_path = os.path.join(config.output, config.name)
        if config.version is not None:
            output_path = os.path.join(output_path, config.version)
        os.makedirs(output_path, exist_ok=True)
        print("output path: {}".format(output_path))
        config.model.output_path = output_path
        if self.config.subcommand == "fit":
            assert os.path.exists(
                os.path.join(output_path, "point_cloud")
            ) is False, "point cloud output already exists in {}".format(output_path)

        # build logger
        logger_config = Namespace(
            class_path=None,
            init_args=Namespace(
                save_dir=output_path,
            ),
        )

        if config.logger == "tensorboard":
            logger_config.class_path = "lightning.pytorch.loggers.TensorBoardLogger"
        elif config.logger == "wandb":
            logger_config.class_path = "lightning.pytorch.loggers.WandbLogger"
            wandb_name = config.name
            if config.version is not None:
                wandb_name = "{}_{}".format(wandb_name, config.version)
            setattr(logger_config.init_args, "name", wandb_name)
            setattr(logger_config.init_args, "project", config.project)
        else:
            logger_config.class_path = config.logger

        config.trainer.logger = logger_config

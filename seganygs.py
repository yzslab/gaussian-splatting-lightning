import lightning
from internal.cli import CLI
from jsonargparse import lazy_instance

from internal.segany_splatting import SegAnySplatting
from internal.dataset import DataModule
from internal.callbacks import SaveCheckpoint, ProgressBar
import lightning.pytorch.loggers


def cli_main():
    cli = CLI(
        SegAnySplatting,
        DataModule,
        seed_everything_default=42,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "auto",
            "devices": 1,
            # "logger": "TensorBoardLogger",
            "num_sanity_val_steps": 1,
            # "max_epochs": 100,
            "max_steps": 30_000,
            "use_distributed_sampler": False,  # use custom ddp sampler
            "enable_checkpointing": False,
            "callbacks": [
                lazy_instance(SaveCheckpoint),
                lazy_instance(ProgressBar),
            ],
        },
        save_config_kwargs={"overwrite": True},
    )
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block

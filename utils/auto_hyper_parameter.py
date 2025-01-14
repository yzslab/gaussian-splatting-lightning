import math
from typing import Literal
import argparse

SCALABEL_PARAMS = {
    "model.gaussian.optimization.means_lr_scheduler.init_args.max_steps": 30_000,
    "model.density.densification_interval": 100,
    "model.density.opacity_reset_interval": 3_000,
    "model.density.densify_from_iter": 500,
    "model.density.densify_until_iter": 15_000,
}

EXTRA_EPOCH_SCALABLE_STEP_PARAMS = [
    "model.gaussian.optimization.means_lr_scheduler.init_args.max_steps",
]


def get_default_scalable_params(max_steps: int = 30_000):
    return {
        "model.gaussian.optimization.means_lr_scheduler.init_args.max_steps": max_steps,
        "model.density.densification_interval": 100,
        "model.density.opacity_reset_interval": 3_000,
        "model.density.densify_from_iter": 500,
        "model.density.densify_until_iter": 15_000,
    }, [
        "model.gaussian.optimization.means_lr_scheduler.init_args.max_steps",
    ]


def auto_hyper_parameter(
        n: int,
        base: int = 300,
        extra_epoch: int = 0,
        scalable_params: dict[str, int] = None,
        extra_epoch_scalable_params: list[str] = None,
        scale_mode: Literal["linear", "sqrt", "none"] = "linear",
        max_steps: int = 30_000,
):
    if scale_mode == "linear":
        scale_up = max(n / base, 1)
    elif scale_mode == "sqrt":
        scale_up = max(math.sqrt(n / base), 1)
    elif scale_mode == "none":
        scale_up = 1
    else:
        raise ValueError("Unknown scale mode '{}'".format(scale_mode))

    # make results more beautiful
    scale_up = math.ceil(scale_up * 100) / 100.

    extra_steps = 0
    if extra_epoch > 0:
        extra_steps = extra_epoch * max(n, base)

    default_scalable_params = get_default_scalable_params(max_steps)
    if scalable_params is None:
        scalable_params = default_scalable_params[0]
    if extra_epoch_scalable_params is None:
        extra_epoch_scalable_params = default_scalable_params[1]

    def scale_params(scalable_params, extra_epoch_scalable_step_params):
        scaled_params = {}
        for name, value in scalable_params.items():
            value = round(value * scale_up)
            if name in extra_epoch_scalable_step_params:
                value += extra_steps

            scaled_params[name] = value
        return scaled_params

    return round(max_steps * scale_up) + extra_steps, scale_params(scalable_params, extra_epoch_scalable_params), scale_up


def to_command_args(max_steps: int, scale_params):
    args = [
        "--max_steps={}".format(max_steps)
    ]

    for i, v in scale_params.items():
        args.append("--{}={}".format(i, v))

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--base", "-b", type=int, default=300)
    parser.add_argument("--extra-epoch", "-e", type=int, default=0)
    parser.add_argument("--mode", type=str, default="linear")
    parser.add_argument("--max-steps", type=int, default=30_000)
    args = parser.parse_args()

    max_steps, scaled_params, _ = auto_hyper_parameter(
        args.n,
        args.base,
        args.extra_epoch,
        scale_mode=args.mode,
        max_steps=args.max_steps,
    )
    print(" ".join(to_command_args(max_steps, scaled_params)))

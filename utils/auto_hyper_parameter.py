import add_pypath
import argparse

from internal.configs.optimization import OptimizationParams

SCALABEL_PARAMS = [
    "position_lr_max_steps",
    "feature_extra_lr_max_steps",
    "densification_interval",
    "opacity_reset_interval",
    "densify_from_iter",
    "densify_until_iter",
]

EXTRA_EPOCH_SCALABLE_STEP_PARAMS = [
    "position_lr_max_steps",
    "feature_extra_lr_max_steps",
]


def auto_hyper_parameter(n: int, base: int = 300, extra_epoch: int = 0):
    optimization_params = OptimizationParams()

    scale_up = max(n / base, 1)

    extra_steps = 0
    if extra_epoch > 0:
        extra_steps = extra_epoch * max(n, base)

    for attr in SCALABEL_PARAMS:
        value = round(getattr(optimization_params, attr) * scale_up)
        if attr in EXTRA_EPOCH_SCALABLE_STEP_PARAMS:
            value += extra_steps
        setattr(
            optimization_params,
            attr,
            value,
        )

    return round(30_000 * scale_up) + extra_steps, optimization_params, scale_up


def to_command_args(max_steps: int, optimization_params: OptimizationParams):
    args = [
        "--max_steps",
        str(max_steps),
    ]

    for i in SCALABEL_PARAMS:
        args.append(f"--model.gaussian.optimization.{i}")
        args.append(str(getattr(optimization_params, i)))

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--base", "-b", type=int, default=300)
    parser.add_argument("--extra-epoch", "-e", type=int, default=0)
    args = parser.parse_args()

    max_steps, params, _ = auto_hyper_parameter(args.n, args.base, args.extra_epoch)
    print(" ".join(to_command_args(max_steps, params)))

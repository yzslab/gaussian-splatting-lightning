import add_pypath
import argparse

from internal.configs.optimization import OptimizationParams
from internal.density_controllers.vanilla_density_controller import VanillaDensityController

SCALABEL_PARAMS = [
    "position_lr_max_steps",
    "feature_extra_lr_max_steps",
]

DENSITY_CONTROLLER_SCALABLE_PARAMS = [
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
    density_controller_params = VanillaDensityController()

    scale_up = max(n / base, 1)

    extra_steps = 0
    if extra_epoch > 0:
        extra_steps = extra_epoch * max(n, base)

    def scale_params(param_dataclass, scalable_params, extra_epoch_scalable_step_params):
        for attr in scalable_params:
            value = round(getattr(param_dataclass, attr) * scale_up)
            if attr in extra_epoch_scalable_step_params:
                value += extra_steps
            setattr(
                param_dataclass,
                attr,
                value,
            )

    scale_params(optimization_params, SCALABEL_PARAMS, EXTRA_EPOCH_SCALABLE_STEP_PARAMS)
    scale_params(density_controller_params, DENSITY_CONTROLLER_SCALABLE_PARAMS, [])

    return round(30_000 * scale_up) + extra_steps, optimization_params, density_controller_params, scale_up


def to_command_args(max_steps: int, optimization_params: OptimizationParams, density_controller_params):
    args = [
        "--max_steps",
        str(max_steps),
    ]

    for i in SCALABEL_PARAMS:
        args.append(f"--model.gaussian.optimization.{i}")
        args.append(str(getattr(optimization_params, i)))
    for i in DENSITY_CONTROLLER_SCALABLE_PARAMS:
        args.append(f"--model.density.{i}")
        args.append(str(getattr(density_controller_params, i)))

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--base", "-b", type=int, default=300)
    parser.add_argument("--extra-epoch", "-e", type=int, default=0)
    args = parser.parse_args()

    max_steps, params, density_params, _ = auto_hyper_parameter(args.n, args.base, args.extra_epoch)
    print(" ".join(to_command_args(max_steps, params, density_params)))

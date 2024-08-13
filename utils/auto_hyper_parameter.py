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


def auto_hyper_parameter(
        n: int,
        base: int = 300,
        extra_epoch: int = 0,
        scalable_params: dict = SCALABEL_PARAMS,
        extra_epoch_scalable_params: dict = EXTRA_EPOCH_SCALABLE_STEP_PARAMS,
):
    scale_up = max(n / base, 1)

    extra_steps = 0
    if extra_epoch > 0:
        extra_steps = extra_epoch * max(n, base)

    def scale_params(scalable_params, extra_epoch_scalable_step_params):
        scaled_params = {}
        for name, value in scalable_params.items():
            value = round(value * scale_up)
            if name in extra_epoch_scalable_step_params:
                value += extra_steps

            scaled_params[name] = value
        return scaled_params

    return round(30_000 * scale_up) + extra_steps, scale_params(scalable_params, extra_epoch_scalable_params), scale_up


def to_command_args(max_steps: int, scale_params):
    args = [
        "--max_steps",
        str(max_steps),
    ]

    for i, v in scale_params.items():
        args += ["--{}".format(i), "{}".format(v)]

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--base", "-b", type=int, default=300)
    parser.add_argument("--extra-epoch", "-e", type=int, default=0)
    args = parser.parse_args()

    max_steps, scaled_params, _ = auto_hyper_parameter(args.n, args.base, args.extra_epoch)
    print(" ".join(to_command_args(max_steps, scaled_params)))

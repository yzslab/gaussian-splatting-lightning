from . import _C


def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale

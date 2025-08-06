import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.utils import _log_api_usage_once, make_grid


@torch.no_grad()
def _save_tensor_image(
    tensor,
    fp,
    format=None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format, subsampling=0, quality=100)


def read_image_as_tensor(path: str) -> torch.Tensor:
    return torchvision.io.read_image(path)  # [C, H, W]


def save_tensor_image(path: str, image: torch.Tensor):
    _save_tensor_image(image, path)


def read_image(path: str) -> np.ndarray:
    pil_image = Image.open(path)
    return np.array(pil_image)  # [H, W, C]


def save_image(path: str, image: np.ndarray):
    pil_image = Image.fromarray(image)
    pil_image.save(path, subsampling=0, quality=100)


def rgba2rgb(rgba: np.ndarray, background: np.ndarray = None) -> np.ndarray:
    if background is None:
        background = np.array([0, 0, 0], dtype=np.float64)
    normalized_rgba = rgba / 255.
    rgb = normalized_rgba[:, :, :3] * normalized_rgba[:, :, 3:4] + background * (1 - normalized_rgba[:, :, 3:4])
    return np.asarray(rgb * 255, dtype=np.uint8)

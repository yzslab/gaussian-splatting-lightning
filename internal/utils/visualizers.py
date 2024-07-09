from typing import Literal
import torch
import matplotlib


class Visualizers:
    @staticmethod
    def no_processing(i, *args, **kwargs):
        return i

    @staticmethod
    def float_colormap(image, colormap: Literal["default", "turbo", "viridis", "magma", "inferno", "cividis", "gray"]):
        """Copied from NeRFStudio: https://github.com/nerfstudio-project/nerfstudio/blob/f97eb2e5f0c754e1ab0873374c8dcea5d18e169c/nerfstudio/utils/colormaps.py#L93-L114. Please follow their license.

        Convert single channel to a color image.

        Args:
            image: Single channel image.
            colormap: Colormap for image.

        Returns:
            Tensor: Colored image with colors in [0, 1]
        """
        if colormap == "default":
            colormap = "turbo"

        image = torch.nan_to_num(image, 0)
        if colormap == "gray":
            return image.repeat(3, 1, 1)
        image_long = (image * 255).long()
        image_long_min = torch.min(image_long)
        image_long_max = torch.max(image_long)
        assert image_long_min >= 0, f"the min value is {image_long_min}"
        assert image_long_max <= 255, f"the max value is {image_long_max}"
        return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[0, ...]].permute(2, 0, 1)

    @staticmethod
    def pca_colormap(image, ignore_zeros: bool = True):
        """Copied from NeRFStudio: https://github.com/nerfstudio-project/nerfstudio/blob/f97eb2e5f0c754e1ab0873374c8dcea5d18e169c/nerfstudio/utils/colormaps.py#L174-L221. Please follow their license.

        Convert feature image to 3-channel RGB via PCA. The first three principle
        components are used for the color channels, with outlier rejection per-channel

        Args:
            image: image of arbitrary vectors, in [C, H, W]
            ignore_zeros: whether to ignore zero values in the input image (they won't affect the PCA computation)

        Returns:
            Tensor: Colored image, in [C, H, W]
        """

        image = image.permute(1, 2, 0)  # [H, W, C]
        original_shape = image.shape
        image = image.view(-1, image.shape[-1])
        if ignore_zeros:
            valids = (image.abs().amax(dim=-1)) > 0
        else:
            valids = torch.ones(image.shape[0], dtype=torch.bool)

        _, _, pca_mat = torch.pca_lowrank(image[valids, :], q=3, niter=20)

        image = torch.matmul(image, pca_mat[..., :3])
        d = torch.abs(image[valids, :] - torch.median(image[valids, :], dim=0).values)
        mdev = torch.median(d, dim=0).values
        s = d / mdev
        m = 2.0  # this is a hyperparam controlling how many std dev outside for outliers
        rins = image[valids, :][s[:, 0] < m, 0]
        gins = image[valids, :][s[:, 1] < m, 1]
        bins = image[valids, :][s[:, 2] < m, 2]

        image[valids, 0] -= rins.min()
        image[valids, 1] -= gins.min()
        image[valids, 2] -= bins.min()

        image[valids, 0] /= rins.max() - rins.min()
        image[valids, 1] /= gins.max() - gins.min()
        image[valids, 2] /= bins.max() - bins.min()

        image = torch.clamp(image, 0, 1)

        return image.view(*original_shape[:-1], 3).permute(2, 0, 1)

    @staticmethod
    def normal_map_colormap(normal_map):
        return torch.nn.functional.normalize(normal_map, dim=0) * 0.5 + 0.5

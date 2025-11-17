from dataclasses import dataclass
import torch
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl


@dataclass
class BackgroundRemoval(VanillaDensityController):
    background_removal_from: int = 7_000

    foreground_radius_scaling: float = 1.

    def instantiate(self, *args, **kwargs):
        return BackgroundRemovalModule(self)


class BackgroundRemovalModule(VanillaDensityControllerImpl):
    def setup(self, stage, pl_module):
        super().setup(stage, pl_module)

        if pl_module is not None:
            scene_center = torch.mean(
                pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center,
                dim=0,
            )
            foreground_radius = torch.linalg.norm(pl_module.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center - scene_center, dim=-1).max() * self.config.foreground_radius_scaling

            print("scene_center={}, foreground_radius={}".format(
                scene_center,
                foreground_radius,
            ))

            self.register_buffer("scene_center", scene_center, persistent=False)
            self.register_buffer("foreground_radius", foreground_radius, persistent=False)

    def after_backward(self, outputs, batch, gaussian_model, optimizers, global_step, pl_module):
        if global_step > self.config.background_removal_from and global_step < self.config.densify_until_iter and global_step % self.config.densification_interval == 0:
            with torch.no_grad():
                prune_mask = torch.linalg.norm(gaussian_model.get_means() - self.scene_center, dim=-1) > self.foreground_radius
                gaussian_model.properties["opacities"][prune_mask] = gaussian_model.opacity_inverse_activation(torch.tensor(0., device=pl_module.device))

        return super().after_backward(outputs, batch, gaussian_model, optimizers, global_step, pl_module)

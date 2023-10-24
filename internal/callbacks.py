import os.path

from lightning.pytorch.callbacks import Callback


class SaveGaussian(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        # TODO: should save before densification
        pl_module.save_gaussian_to_ply()

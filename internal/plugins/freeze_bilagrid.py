from dataclasses import dataclass, field
import torch
from .plugin import Plugin, PluginModule


@dataclass
class FreezeBilagrid(Plugin):
    def instantiate(self):
        return FreezeBilagridModule(self)


class FreezeBilagridModule(PluginModule):
    def setup(self, pl_module):
        pl_module.on_after_backward_hooks.append(self.remove_grad)

    def remove_grad(self, outputs, batch, gaussian_model, global_step, pl_module):
        pl_module.output_processor.bgrid.grids.grad = None

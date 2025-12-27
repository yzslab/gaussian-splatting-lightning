from dataclasses import dataclass
import torch


class Plugin:
    def instantiate(self):
        raise NotImplementedError()


class PluginModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, pl_module):
        return

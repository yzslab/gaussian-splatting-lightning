from typing import Tuple, Union, List, Dict, Optional, Type
import torch
from torch import nn
from lightning import LightningModule
from internal.configs.instantiate_config import InstantiatableConfig


class DensityControllerImpl(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def before_backward(self, outputs: dict, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        pass

    def after_backward(self, outputs: dict, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        pass

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        pass

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def after_density_changed(self, gaussian_model, optimizers: List, pl_module: LightningModule) -> None:
        """
        This interface will be invoked when the density is changed elsewhere
        """
        pass


class DensityController(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        pass


class Utils:
    # TODO: update non-optimizable properties

    @staticmethod
    def cat_tensors_to_optimizers(new_properties: Dict[str, torch.Tensor], optimizers: List[torch.optim.Optimizer]) -> Dict[str, torch.Tensor]:
        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                extension_tensor = new_properties[group["name"]]

                # get current sates
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    # append states for new properties
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )
                    # delete old state key by old params from optimizer
                    del opt.state[group['params'][0]]
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))
                    # update optimizer states
                    opt.state[group['params'][0]] = stored_state
                else:
                    # append new parameters to optimizer
                    group["params"][0] = nn.Parameter(torch.cat(
                        (group["params"][0], extension_tensor),
                        dim=0,
                    ).requires_grad_(True))

                # add new `nn.Parameter` from optimizers to the dict returned later
                new_parameters[group["name"]] = group["params"][0]

        return new_parameters

    @staticmethod
    def prune_optimizers(mask, optimizers):
        """

        :param mask: The `False` indicating the ones to be pruned
        :param optimizers:
        :return: a new dict
        """

        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))

                new_parameters[group["name"]] = group["params"][0]

        return new_parameters

    @staticmethod
    def replace_tensors_to_optimizers(tensors: Dict[str, torch.Tensor], optimizers, selector=None):
        """
        Args:
            tensors: to be the new parameters of optimizers
            optimizers: optimizer list
            selector: indicating which ones have been updated, and their optimizer state will be reset. if is None, all states will be reset.
        Return:
            a new dict containing all the new parameters of optimizers
        """

        new_parameters = {}
        for opt in optimizers:
            for group in opt.param_groups:
                tensor = tensors.get(group["name"], None)
                if tensor is None:
                    continue

                assert len(group["params"]) == 1
                assert group["name"] not in new_parameters, "parameter `{}` appears in multiple optimizers".format(group["name"])

                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    if selector is not None:
                        stored_state["exp_avg"][selector] = 0
                        stored_state["exp_avg_sq"][selector] = 0
                    else:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))

                new_parameters[group["name"]] = group["params"][0]

        return new_parameters

from typing import Literal
import torch
from torch import nn


class NetworkFactory:
    def __init__(self, tcnn: bool = True, seed: int = 1337):
        self.tcnn = tcnn
        self.seed = seed

    def _get_seed(self):
        try:
            return self.seed
        finally:
            self.seed += 1

    def get_linear(self, in_features: int, out_features: int):
        return self.get_network(
            n_input_dims=in_features,
            n_output_dims=out_features,
            n_layers=1,
            n_neurons=out_features,
            activation="ReLU",
            output_activation="None",
        )

    def get_network(
            self,
            n_input_dims: int,
            n_output_dims: int,
            n_layers: int,
            n_neurons: int,
            activation: Literal["ReLU", "None"],
            output_activation: Literal["ReLU", "None"],
    ):
        assert n_layers > 0 and n_neurons > 0

        if self.tcnn is True:
            import tinycudann as tcnn
            otype = "FullyFusedMLP"
            if n_neurons > 128:
                otype = "CutlassMLP"
            return tcnn.Network(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                network_config={
                    "otype": otype,
                    "activation": activation,
                    "output_activation": output_activation,
                    "n_neurons": n_neurons,
                    "n_hidden_layers": n_layers - 1,
                },
                seed=self._get_seed(),
            )

        # PyTorch
        model_list = []
        # hidden layers
        in_features = n_input_dims
        for i in range(n_layers - 1):
            model_list += self._get_torch_layer(in_features, n_neurons, activation)
            in_features = n_neurons  # next layer's in_features
        # output layer
        model_list += self._get_torch_layer(in_features, n_output_dims, output_activation)

        return nn.Sequential(*model_list)

    def _get_torch_activation(self, name: str):
        if name == "None":
            return None
        if name == "ReLU":
            return nn.ReLU()
        raise ValueError("unsupported activation type {}".format(name))

    def _get_torch_layer(self, in_features: int, out_features: int, activation_name: str) -> list:
        model_list = []
        layer = nn.Linear(in_features, out_features)
        activation = self._get_torch_activation(activation_name)
        model_list.append(layer)
        if activation is not None:
            model_list.append(activation)

        return model_list

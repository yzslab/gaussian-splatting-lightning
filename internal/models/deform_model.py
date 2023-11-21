"""
Copied from https://github.com/ingra14m/Deformable-3D-Gaussians/blob/main/utils/time_utils.py
"""

import torch
import torch.nn as nn
from internal.encodings.positional_encoding import PositionalEncoding
from internal.utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    pe = PositionalEncoding(input_channels=i, n_frequencies=multires, log_sampling=True)
    return pe, pe.get_output_n_channels()


def get_time_embedder(multires, i=1, n_layers: int = 0, n_neurons: int = 0, output_ch: int = 30):
    if (n_layers > 0 and n_neurons > 0) is False:
        return get_embedder(multires, i)
    return TimeNetwork(D=n_layers, W=n_neurons, output_ch=output_ch, multires=multires), output_ch


class TimeNetwork(nn.Module):
    def __init__(self, D=2, W=256, input_ch=1, output_ch=30, multires=10):
        super().__init__()
        self.t_multires = multires
        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, input_ch)

        import tinycudann as tcnn

        self.timenet = tcnn.Network(
            n_input_dims=time_input_ch,
            n_output_dims=output_ch,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": D - 1
            }
        )

    def forward(self, t):
        return self.timenet(self.embed_time_fn(t))


class DeformModel(nn.Module):
    def __init__(
            self,
            D=8,
            W=256,
            input_ch=3,
            # output_ch=59,
            multires=10,
            t_D=0,
            t_W=0,
            t_multires=6,
            t_output_ch=30,
            is_6dof=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        # self.output_ch = output_ch
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_time_embedder(self.t_multires, 1, n_layers=t_D, n_neurons=t_W, output_ch=t_output_ch)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        import tinycudann as tcnn

        # seed generator, used to initialize tiny-cuda-nn
        seed = 1336

        def get_seed():
            nonlocal seed
            seed += 1
            return seed

        # build deformable field
        skip_layer_list = []
        initialized_layers = 0
        n_input_dims = self.input_ch
        for i in self.skips:
            n_hidden_layers = D - 1 - i
            skip_layer_list.append(tcnn.Network(
                n_input_dims=n_input_dims,
                n_output_dims=W,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "ReLU",
                    "n_neurons": W,
                    "n_hidden_layers": n_hidden_layers,
                },
                seed=get_seed(),
            ))
            n_input_dims = W + self.input_ch
            initialized_layers += n_hidden_layers + 1
        self.skip_layers = nn.ModuleList(skip_layer_list)
        self.output_linear = tcnn.Network(
            n_input_dims=n_input_dims,
            n_output_dims=W,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": W,
                "n_hidden_layers": D - 1 - initialized_layers,
            },
            seed=get_seed(),
        )

        self.is_6dof = is_6dof

        if is_6dof:
            # self.branch_w = nn.Linear(W, 3)
            # self.branch_v = nn.Linear(W, 3)
            self.branch_w = tcnn.Network(
                n_input_dims=W,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": W,
                    "n_hidden_layers": 0,
                },
                seed=get_seed(),
            )
            self.branch_v = tcnn.Network(
                n_input_dims=W,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": W,
                    "n_hidden_layers": 0,
                },
                seed=get_seed(),
            )
        else:
            # self.gaussian_warp = nn.Linear(W, 3)
            self.gaussian_warp = tcnn.Network(
                n_input_dims=W,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": W,
                    "n_hidden_layers": 0,
                },
                seed=get_seed(),
            )
        # self.gaussian_rotation = nn.Linear(W, 4)
        # self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_rotation = tcnn.Network(
            n_input_dims=W,
            n_output_dims=4,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": 0,
            },
            seed=get_seed(),
        )
        self.gaussian_scaling = tcnn.Network(
            n_input_dims=W,
            n_output_dims=3,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": W,
                "n_hidden_layers": 0,
            },
            seed=get_seed(),
        )

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)

        # query deformable field
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.skip_layers):
            h = self.skip_layers[i](h)
            h = torch.cat([x_emb, t_emb, h], -1)
        h = self.output_linear(h)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

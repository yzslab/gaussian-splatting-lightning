from .vanilla_renderer import *


class RGBMLPRenderer(VanillaRenderer):
    def __init__(
            self,
            compute_cov3D_python: bool = False,
            n_neurons: int = 128,
            n_hidden_layers: int = 3,
            lr: float = 1e-4,
            gamma: float = 0.1,
            max_steps: int = 30_000,
    ):
        super().__init__(compute_cov3D_python, convert_SHs_python=False)

        self.n_neurons = n_neurons
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.gamma = gamma
        self.max_steps = max_steps

    def setup(self, stage: str, **kwargs):
        super().setup(stage, **kwargs)
        import tinycudann as tcnn

        self.rgb_network = tcnn.NetworkWithInputEncoding(
            n_input_dims=1 + 3 + 3 * ((3 + 1) ** 2),  # 1: appearance embedding, 3: view direction, others: SHs
            n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    # encoding appearance embedding
                    {
                        "n_dims_to_encode": 1,
                        "otype": "Frequency",
                        "degree": 6
                    },
                    {
                        "otype": "Identity"
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": self.n_neurons,
                "n_hidden_layers": self.n_hidden_layers,
            },
        )

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        # view directions
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        # spherical harmonics
        shs = pc.get_features
        shs = shs.transpose(1, 2).reshape((shs.shape[0], -1))

        override_color = self.rgb_network(torch.concatenate([
            viewpoint_camera.appearance_embedding.repeat(shs.shape[0]).unsqueeze(-1),
            dir_pp_normalized,
            shs,
        ], dim=-1)).to(torch.float)
        return super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)

    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            params=[
                {"params": list(self.rgb_network.parameters()), "name": "mlp_renderer"},
            ],
            lr=self.lr,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.gamma ** min(iter / self.max_steps, 1),
            verbose=False,
        )

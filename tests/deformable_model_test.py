import unittest
import torch
from internal.utils.network_factory import NetworkFactory

from internal.encodings.positional_encoding import PositionalEncoding
from internal.models.deform_model import DeformModel


class DeformableModelTestCase(unittest.TestCase):
    def test_deformable_model(self):
        # make sure that both tcnn and pytorch implementation work
        network_factory = NetworkFactory(tcnn=True)
        dn = DeformModel(network_factory, t_D=2, t_W=256, t_multires=10).to("cuda")
        print(dn(torch.zeros((1, 3), device="cuda"), torch.zeros((1, 1), device="cuda")))

        network_factory = NetworkFactory(tcnn=False)
        dn = DeformModel(network_factory, t_D=2, t_W=256, t_multires=10).to("cuda")
        print(dn(torch.zeros((1, 3), device="cuda"), torch.zeros((1, 1), device="cuda")))

        # make sure that the network architecture is correct
        network_factory = NetworkFactory(tcnn=False)
        dn = DeformModel(network_factory, t_D=2, t_W=256, t_multires=6).to("cuda")
        # time network, [nn.Linear(), nn.ReLU(), nn.Linear()]
        self.assertEqual(len(dn.embed_time_fn.timenet), 3)
        self.assertTrue(dn.embed_time_fn.timenet[0].in_features, 13)
        self.assertTrue(dn.embed_time_fn.timenet[0].out_features, 256)
        self.assertTrue(isinstance(dn.embed_time_fn.timenet[1], torch.nn.ReLU))
        self.assertTrue(dn.embed_time_fn.timenet[0].in_features, 256)
        self.assertTrue(dn.embed_time_fn.timenet[0].out_features, 30)
        self.assertEqual(len(dn.skip_layers), 1)
        self.assertEqual(len(dn.skip_layers[0]), 10)  # 5 x (Linear, ReLU)
        # 5x (Linear, ReLU)
        in_features = 93
        for i in range(5):
            self.assertEqual(dn.skip_layers[0][2 * i].in_features, in_features)
            self.assertEqual(dn.skip_layers[0][2 * i].out_features, 256)
            self.assertTrue(isinstance(dn.skip_layers[0][2 * i], torch.nn.Linear))
            self.assertTrue(isinstance(dn.skip_layers[0][2 * i + 1], torch.nn.ReLU))
            in_features = 256
        # 3x (Linear, ReLU)
        self.assertEqual(len(dn.output_linear), 6)
        in_features = 349  # skip layer
        for i in range(3):
            self.assertEqual(dn.output_linear[2 * i].in_features, in_features)
            self.assertEqual(dn.output_linear[2 * i].out_features, 256)
            self.assertTrue(isinstance(dn.output_linear[2 * i], torch.nn.Linear))
            self.assertTrue(isinstance(dn.output_linear[2 * i + 1], torch.nn.ReLU))
            in_features = 256
        print(dn)

        # make sure it works when time network disabled (real world scene)
        network_factory = NetworkFactory(tcnn=False)
        dn = DeformModel(network_factory, t_D=0, t_W=0, t_multires=10).to("cuda")
        print(dn(torch.zeros((1, 3), device="cuda"), torch.zeros((1, 1), device="cuda")))
        self.assertTrue(isinstance(dn.embed_time_fn, PositionalEncoding))
        self.assertEqual(dn.embed_time_fn.n_frequencies, 10)
        self.assertEqual(len(dn.skip_layers), 1)
        self.assertEqual(len(dn.skip_layers[0]), 10)
        self.assertEqual(len(dn.output_linear), 6)
        self.assertEqual(dn.skip_layers[0][0].in_features, 84)
        print(dn)


if __name__ == '__main__':
    unittest.main()

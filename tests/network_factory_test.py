import unittest

import torch
from torch import nn

from internal.utils.network_factory import NetworkFactory, NetworkWithSkipLayers


class NetworkFactoryTestCase(unittest.TestCase):
    def test_tcnn(self):
        factory = NetworkFactory(tcnn=True)

        network = factory.get_network(
            256,
            3,
            2,
            256,
            "ReLU",
            "ReLU",
        )
        self.assertEqual(network.n_input_dims, 256)
        self.assertEqual(network.n_output_dims, 3)
        self.assertEqual(network.network_config["otype"], "CutlassMLP")
        self.assertEqual(network.network_config["n_hidden_layers"], 1)
        self.assertEqual(network.network_config["n_neurons"], 256)
        self.assertEqual(network.network_config["activation"], "ReLU")
        self.assertEqual(network.network_config["output_activation"], "ReLU")

        input = torch.rand((1, 256), device="cuda")
        output = network(input)
        self.assertEqual(output.shape, (1, 3))

        network = factory.get_network(
            256,
            3,
            2,
            256,
            "None",
            "ReLU",
        )
        self.assertEqual(network.network_config["activation"], "None")
        self.assertEqual(network.network_config["output_activation"], "ReLU")

        network = factory.get_network(
            256,
            3,
            1,
            256,
            "ReLU",
            "None",
        )
        self.assertEqual(network.network_config["activation"], "ReLU")
        self.assertEqual(network.network_config["output_activation"], "None")

        input = torch.rand((1, 256), device="cuda")
        output = network(input)
        self.assertEqual(output.shape, (1, 3))

    def test_torch(self):
        factory = NetworkFactory(tcnn=False)

        network = factory.get_network(256, 3, 3, 256, "ReLU", "ReLU").to("cuda")
        # print(network)
        input = torch.rand((1, 256), device="cuda")
        output = network(input)
        self.assertEqual(output.shape, (1, 3))
        self.assertEqual(len(network), 6)
        self.assertTrue(isinstance(network[0], torch.nn.Linear))
        self.assertTrue(isinstance(network[1], torch.nn.ReLU))
        self.assertTrue(isinstance(network[2], torch.nn.Linear))
        self.assertTrue(isinstance(network[3], torch.nn.ReLU))
        self.assertTrue(isinstance(network[4], torch.nn.Linear))
        self.assertTrue(isinstance(network[5], torch.nn.ReLU))

        network = factory.get_network(256, 3, 3, 256, "ReLU", "None").to("cuda")
        # print(network)
        self.assertEqual(len(network), 5)
        self.assertTrue(isinstance(network[0], torch.nn.Linear))
        self.assertTrue(isinstance(network[1], torch.nn.ReLU))
        self.assertTrue(isinstance(network[2], torch.nn.Linear))
        self.assertTrue(isinstance(network[3], torch.nn.ReLU))
        self.assertTrue(isinstance(network[4], torch.nn.Linear))

        network = factory.get_network(256, 3, 3, 256, "None", "ReLU").to("cuda")
        # print(network)
        self.assertEqual(len(network), 4)
        self.assertTrue(isinstance(network[0], torch.nn.Linear))
        self.assertTrue(isinstance(network[1], torch.nn.Linear))
        self.assertTrue(isinstance(network[2], torch.nn.Linear))
        self.assertTrue(isinstance(network[3], torch.nn.ReLU))

    def test_seed(self):
        seed_1 = NetworkFactory(tcnn=True)._get_seed()
        seed_2 = NetworkFactory(tcnn=True)._get_seed()
        self.assertEqual(seed_2 - seed_1, 1)

    def test_skip_layers(self):
        # D = 8
        # input_ch = 3
        # W = 256
        # skips = [2, 4]
        # pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])
        # print(pts_linears)

        factory = NetworkFactory(tcnn=False)
        network = factory.get_network_with_skip_layers(
            n_input_dims=3,
            n_output_dims=16,
            n_layers=8,
            n_neurons=256,
            activation="ReLU",
            output_activation="ReLU",
            skips=[3, 5],
        )
        self.assertEqual(len(network.skip_layers[0]), 6)  # [linear, relu, linear, relu, linear, relu]
        self.assertEqual(len(network.skip_layers[1]), 4)  # [linear, relu, linear, relu]
        self.assertEqual(len(network.output_layers), 6)  # [linear, relu, linear, relu, linear, relu]

        with torch.no_grad():
            input = torch.rand((3, 3))
            output = network(input)
            self.assertEqual(output.shape, (3, 16))


if __name__ == '__main__':
    unittest.main()

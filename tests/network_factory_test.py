import unittest

import torch

from internal.utils.network_factory import NetworkFactory


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


if __name__ == '__main__':
    unittest.main()

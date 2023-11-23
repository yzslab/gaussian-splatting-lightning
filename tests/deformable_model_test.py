import unittest
import torch
from internal.utils.network_factory import NetworkFactory

from internal.models.deform_model import DeformModel


class DeformableModelTestCase(unittest.TestCase):
    def test_deformable_model(self):
        network_factory = NetworkFactory(tcnn=True)
        dn = DeformModel(network_factory, t_D=2, t_W=256, t_multires=10).to("cuda")
        print(dn(torch.zeros((1, 3), device="cuda"), torch.zeros((1, 1), device="cuda")))

        network_factory = NetworkFactory(tcnn=False)
        dn = DeformModel(network_factory, t_D=2, t_W=256, t_multires=10).to("cuda")
        print(dn(torch.zeros((1, 3), device="cuda"), torch.zeros((1, 1), device="cuda")))


if __name__ == '__main__':
    unittest.main()

import unittest
import torch
from internal.models.gaussian import FreezableParameterDict


class FreezableParameterDictTestCase(unittest.TestCase):
    def test_freezable_parameter_dict(self):
        tensors = {
            str(i): torch.rand((3, 3))
            for i in range(16)
        }
        for i in tensors.values():
            self.assertFalse(i.requires_grad)

        parameter_dict = torch.nn.ParameterDict(tensors)
        for i in parameter_dict.values():
            self.assertTrue(isinstance(i, torch.nn.Parameter))
            self.assertTrue(i.requires_grad)

        # untouched
        freezable_parameter_dict = FreezableParameterDict(parameter_dict)
        self.assertEqual(freezable_parameter_dict.keys(), parameter_dict.keys())
        for i in freezable_parameter_dict.values():
            self.assertTrue(i.requires_grad)
        freezable_parameter_dict["frozen"] = torch.rand((3, 3))
        self.assertFalse(freezable_parameter_dict["frozen"].requires_grad)
        freezable_parameter_dict["optimizable"] = torch.rand((3, 3)).requires_grad_(True)
        self.assertTrue(freezable_parameter_dict["optimizable"].requires_grad)

        # freeze by default
        frozen_parameter_dict = FreezableParameterDict(freezable_parameter_dict, new_requires_grad=False)
        self.assertEqual(frozen_parameter_dict.keys(), freezable_parameter_dict.keys())
        for i in frozen_parameter_dict.values():
            self.assertFalse(i.requires_grad)
        frozen_parameter_dict["freeze_new_tensor"] = torch.rand((3, 3)).requires_grad_(True)
        self.assertFalse(frozen_parameter_dict["freeze_new_tensor"].requires_grad)

        # optimizable by default (same as `nn.ParameterDict`)
        optimizable_parameter_dict = FreezableParameterDict(frozen_parameter_dict, new_requires_grad=True)
        self.assertEqual(optimizable_parameter_dict.keys(), frozen_parameter_dict.keys())
        for i in optimizable_parameter_dict.values():
            self.assertTrue(i.requires_grad)
        optimizable_parameter_dict["optimizable_new"] = torch.rand((3, 3))
        self.assertTrue(optimizable_parameter_dict["optimizable_new"].requires_grad)


if __name__ == '__main__':
    unittest.main()

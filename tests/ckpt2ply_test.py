import unittest
import torch
from internal.utils.gaussian_utils import GaussianPlyUtils


class Ckpt2PlyTest(unittest.TestCase):
    def test_ckpt2ply(self):
        ckpt_path = "outputs/lego/0905/checkpoints/epoch=300-step=30000.ckpt"
        ckpt = torch.load(ckpt_path, map_location="cpu")

        output_ply_name = "lego-unittest.ply"

        # save as float
        gaussians_ply_utils = GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"])
        gaussians_ply_utils.to_ply_format().save_to_ply(output_ply_name)
        del gaussians_ply_utils
        gaussians_ply_utils = GaussianPlyUtils.load_from_ply(output_ply_name)
        gaussians_ply_utils = gaussians_ply_utils.to_parameter_structure()
        self.validate_properties(ckpt, gaussians_ply_utils)

        # save as uint8
        # gaussians_ply_utils = GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"])
        # gaussians_ply_utils.to_ply_format().save_to_ply_uint8(output_ply_name)
        # del gaussians_ply_utils
        # gaussians_ply_utils = GaussianPlyUtils.load_from_ply_uint8(output_ply_name)
        # gaussians_ply_utils = gaussians_ply_utils.to_parameter_structure()
        # self.validate_properties(ckpt, gaussians_ply_utils)

    def validate_properties(self, ckpt, ply_utils: GaussianPlyUtils):
        state_dict = ckpt["state_dict"]
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.means"], ply_utils.xyz), "means")
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.shs_dc"], ply_utils.features_dc), "shs_dc")
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.shs_rest"], ply_utils.features_rest), "shs_rest")
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.scales"], ply_utils.scales), "scales")
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.rotations"], ply_utils.rotations), "rotations")
        self.assertTrue(torch.allclose(state_dict["gaussian_model.gaussians.opacities"], ply_utils.opacities), "opacities")


if __name__ == "__main__":
    unittest.main()

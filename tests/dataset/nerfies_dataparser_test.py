import os.path
import unittest
import torch
from internal.configs.dataset import NerfiesParams
from internal.dataparsers.nerfies_dataparser import NerfiesDataparser


class NerfiesDataparserTestCase(unittest.TestCase):
    def test_nerfies_dataparser(self):
        daraparser = NerfiesDataparser(
            path=os.path.expanduser("~/data/DynamicDatasets/HyperNeRF/espresso"),
            output_path="/tmp/HyperNeRF",
            global_rank=0,
            params=NerfiesParams()
        )
        outputs_1x = daraparser.get_outputs()

        daraparser = NerfiesDataparser(
            path=os.path.expanduser("~/data/DynamicDatasets/HyperNeRF/espresso"),
            output_path="/tmp/HyperNeRF",
            global_rank=0,
            params=NerfiesParams(down_sample_factor=2)
        )
        outputs_2x = daraparser.get_outputs()
        self.assertTrue(torch.allclose(outputs_1x.train_set.cameras.fx / 2., outputs_2x.train_set.cameras.fx))
        self.assertTrue(torch.allclose(outputs_1x.train_set.cameras.fy / 2., outputs_2x.train_set.cameras.fy))
        self.assertTrue(torch.allclose(outputs_1x.train_set.cameras.cx / 2., outputs_2x.train_set.cameras.cx))
        self.assertTrue(torch.allclose(outputs_1x.train_set.cameras.cy / 2., outputs_2x.train_set.cameras.cy))
        self.assertTrue(torch.all(outputs_2x.train_set.cameras.time <= 1.))
        self.assertTrue(torch.all(outputs_2x.val_set.cameras.time <= 1.))

        for i in outputs_2x.train_set.image_paths:
            self.assertTrue("rgb/2x/" in i)

        self.assertEqual(outputs_2x.val_set.image_names[0], "000001.png")
        print(outputs_2x.val_set.cameras[0])

        w2c = torch.eye(4)
        w2c[:3, :3] = outputs_2x.val_set.cameras[0].R
        w2c[:3, 3] = outputs_2x.val_set.cameras[0].T
        c2w = torch.linalg.inv(w2c)
        self.assertTrue(torch.allclose(c2w[:3, 3], torch.tensor([
            0.008652675425308145,
            -0.00921293454554443,
            -0.70470877132779
        ])))


if __name__ == '__main__':
    unittest.main()

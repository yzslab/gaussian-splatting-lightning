import unittest
import numpy as np
import torch
import random
import time
from tqdm.auto import tqdm
from internal.utils.sh_utils import eval_sh, eval_sh_decomposed


class SHUtilsTest(unittest.TestCase):
    def test_eval_sh_decomposed(self):
        random.seed(42)
        np.random.seed(42)
        torch.random.manual_seed(42)

        ckpt = torch.load("outputs/lego/glossy/checkpoints/epoch=300-step=30000.ckpt", map_location="cuda")
        assert ckpt["state_dict"]["gaussian_model._active_sh_degree"] == 3

        rgb_shs_dc = ckpt["state_dict"]["gaussian_model.gaussians.shs_dc"].repeat(5, 1, 1).contiguous()  # [N, N_DC, C] = [N, 1, 3]
        rgb_shs_rest = ckpt["state_dict"]["gaussian_model.gaussians.shs_rest"].repeat(5, 1, 1).contiguous()  # [N, N_REST, C] = [N, 15, 3]
        print("rgb_shs_dc.shape={}, rgb_shs_rest.shape={}".format(
            rgb_shs_dc.shape,
            rgb_shs_rest.shape,
        ))
        opacity_shs_dc = ckpt["state_dict"]["gaussian_model.gaussians.opacities"].repeat(5, 1, 1).contiguous()  # [N, N_DC, C] = [N, 1, 1]
        opacity_shs_rest = ckpt["state_dict"]["gaussian_model.gaussians.opacity_rest"].repeat(5, 1, 1).contiguous()  # [N, N_REST, C] = [N, 15, 1]

        vanilla_comsumed_time = 0
        decomposed_comsumed_time = 0

        with torch.no_grad():
            for _ in tqdm(range(2048)):
                dummy_dirs = torch.nn.functional.normalize(torch.rand((rgb_shs_dc.shape[0], 3), device=rgb_shs_dc.device), dim=-1)

                (ref, decomposed), (vanilla_comsumed, decomposed_comsumed) = self.eval_sh(
                    shs_dc=rgb_shs_dc,
                    shs_rest=rgb_shs_rest,
                    viewdirs=dummy_dirs,
                )
                assert torch.allclose(ref, decomposed)
                assert tuple(ref.shape) == tuple(decomposed.shape)
                vanilla_comsumed_time += vanilla_comsumed
                decomposed_comsumed_time += decomposed_comsumed

                (ref, decomposed), (vanilla_comsumed, decomposed_comsumed) = self.eval_sh(
                    shs_dc=opacity_shs_dc,
                    shs_rest=opacity_shs_rest,
                    viewdirs=dummy_dirs,
                )
                assert torch.allclose(ref, decomposed)
                vanilla_comsumed_time += vanilla_comsumed
                decomposed_comsumed_time += decomposed_comsumed

        print("vanilla_comsumed_time={}, decomposed_comsumed_time={}".format(
            vanilla_comsumed_time,
            decomposed_comsumed_time,
        ))

    def eval_sh(self, shs_dc, shs_rest, viewdirs):
        torch.cuda.synchronize()

        started_at = time.time()
        ref = eval_sh(
            deg=3,
            sh=torch.concat([shs_dc, shs_rest], dim=1).transpose(1, 2),  # [N, C, N_DC + N_REST]
            dirs=viewdirs,
        )
        torch.cuda.synchronize()
        vanilla_comsumed = time.time() - started_at

        started_at = time.time()
        decomposed = eval_sh_decomposed(
            deg=3,
            shs_dc=shs_dc,
            shs_rest=shs_rest,
            dirs=viewdirs,
        )
        torch.cuda.synchronize()
        decomposed_comsumed = time.time() - started_at

        return (ref, decomposed), (vanilla_comsumed, decomposed_comsumed)


if __name__ == "__main__":
    unittest.main()

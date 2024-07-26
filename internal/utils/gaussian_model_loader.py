import os
import glob
import torch
from typing import Tuple
from internal.models.gaussian import Gaussian
from internal.renderers import RendererConfig
from internal.renderers.vanilla_renderer import VanillaRenderer


class GaussianModelLoader:
    previous_name_to_new = {
        "_xyz": "means",
        "_features_dc": "shs_dc",
        "_features_rest": "shs_rest",
        "_scaling": "scales",
        "_rotation": "rotations",
        "_opacity": "opacities",
        "_features_extra": "appearance_features",
    }

    previous_state_dict_key_to_new = {
        "_xyz": "gaussians.means",
        "_features_dc": "gaussians.shs_dc",
        "_features_rest": "gaussians.shs_rest",
        "_scaling": "gaussians.scales",
        "_rotation": "gaussians.rotations",
        "_opacity": "gaussians.opacities",
        "_features_extra": "gaussians.appearance_features",
    }

    @staticmethod
    def search_load_file(model_path: str) -> str:
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path) is False:
            return model_path
        # search checkpoint
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        # find checkpoint with max iterations
        load_from = None
        previous_checkpoint_iteration = -1
        for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
            try:
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue
            if checkpoint_iteration > previous_checkpoint_iteration:
                previous_checkpoint_iteration = checkpoint_iteration
                load_from = i

        # not a checkpoint can be found, search point cloud
        if load_from is None:
            previous_point_cloud_iteration = -1
            for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
                try:
                    point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
                except Exception as err:
                    print("error occurred when parsing iteration from {}: {}".format(i, err))
                    continue

                if point_cloud_iteration > previous_point_cloud_iteration:
                    previous_point_cloud_iteration = point_cloud_iteration
                    load_from = os.path.join(i, "point_cloud.ply")

        assert load_from is not None, "not a checkpoint or point cloud can be found"

        return load_from

    @staticmethod
    def filter_state_dict_by_prefix(state_dict, prefix: str, device=None):
        prefix_len = len(prefix)
        new_state_dict = state_dict.__class__()
        for name in state_dict:
            if name.startswith(prefix):
                state = state_dict[name]
                if device is not None:
                    state = state.to(device)
                new_state_dict[name[prefix_len:]] = state
        return new_state_dict

    @classmethod
    def initialize_model_from_checkpoint(cls, checkpoint: dict, device):
        hparams = checkpoint["hyper_parameters"]

        if isinstance(hparams["gaussian"], Gaussian):
            model = hparams["gaussian"].instantiate()
            model_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "gaussian_model.", device=device)
        else:
            # convert previous checkpoint to new format
            sh_degree = hparams["gaussian"].sh_degree
            model_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "gaussian_model.", device=device)
            model_state_dict = {cls.previous_state_dict_key_to_new.get(name): model_state_dict[name] for name in model_state_dict}
            model_state_dict["_active_sh_degree"] = torch.tensor(sh_degree, dtype=torch.int, device=device)

            from internal.models.vanilla_gaussian import VanillaGaussian

            if "gaussians.appearance_features" in model_state_dict:
                if model_state_dict["gaussians.appearance_features"].shape[-1] > 0:
                    from internal.models.appearance_feature_gaussian import AppearanceFeatureGaussian
                    model = AppearanceFeatureGaussian(sh_degree=sh_degree, appearance_feature_dims=model_state_dict["gaussians.appearance_features"].shape[-1]).instantiate()
                else:
                    del model_state_dict["gaussians.appearance_features"]
                    model = VanillaGaussian(sh_degree=sh_degree).instantiate()
            else:
                model = VanillaGaussian(sh_degree=sh_degree).instantiate()

        model.setup_from_number(model_state_dict["gaussians.means"].shape[0])
        model.to(device)
        model.load_state_dict(model_state_dict)

        return model

    @classmethod
    def initialize_renderer_from_checkpoint(cls, checkpoint: dict, stage: str, device):
        hparams = checkpoint["hyper_parameters"]
        # extract state dict of renderer
        renderer = hparams["renderer"]
        renderer_state_dict = cls.filter_state_dict_by_prefix(checkpoint["state_dict"], "renderer.", device=device)
        # load state dict of renderer
        if isinstance(renderer, RendererConfig):
            renderer = renderer.instantiate()
            renderer.setup(stage=stage)
        renderer = renderer.to(device)
        renderer.load_state_dict(renderer_state_dict)
        return renderer

    @classmethod
    def initialize_model_and_renderer_from_checkpoint_file(
            cls,
            checkpoint_path: str,
            device,
            eval_mode: bool = True,
            pre_activate: bool = True,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, dict]:
        stage = "fit"
        if eval_mode is True:
            stage = "validation"

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = cls.initialize_model_from_checkpoint(checkpoint, device)
        renderer = cls.initialize_renderer_from_checkpoint(checkpoint, stage, device)

        if eval_mode is True:
            model.eval()
            renderer.eval()

        if pre_activate is True:
            model.pre_activate_all_properties()

        return model, renderer, checkpoint

    @staticmethod
    def initialize_model_and_renderer_from_ply_file(ply_file_path: str, device, eval_mode: bool = True, pre_activate: bool = True):
        from internal.utils.gaussian_utils import GaussianPlyUtils
        gaussian_ply_utils = GaussianPlyUtils.load_from_ply(ply_file_path).to_parameter_structure()
        model_state_dict = {
            "_active_sh_degree": torch.tensor(gaussian_ply_utils.sh_degrees, dtype=torch.int, device=device),
            "gaussians.means": gaussian_ply_utils.xyz.to(device),
            "gaussians.opacities": gaussian_ply_utils.opacities.to(device),
            "gaussians.shs_dc": gaussian_ply_utils.features_dc.to(device),
            "gaussians.shs_rest": gaussian_ply_utils.features_rest.to(device),
            "gaussians.scales": gaussian_ply_utils.scales.to(device),
            "gaussians.rotations": gaussian_ply_utils.rotations.to(device),
        }

        # use 2DGS if the number of last dim is 2
        if model_state_dict["gaussians.scales"].shape[-1] == 2:
            from internal.models.gaussian_2d import Gaussian2D
            model = Gaussian2D(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
        else:
            from internal.models.vanilla_gaussian import VanillaGaussian
            model = VanillaGaussian(sh_degree=gaussian_ply_utils.sh_degrees).instantiate()
        model.setup_from_number(gaussian_ply_utils.xyz.shape[0])
        model.to(device)
        model.load_state_dict(model_state_dict, strict=False)

        renderer = VanillaRenderer().to(device)

        if eval_mode is True:
            renderer.setup("validation")
            model.eval()
        else:
            renderer.setup("fit")

        if pre_activate is True:
            model.pre_activate_all_properties()

        return model, renderer

    @classmethod
    def search_and_load(
            cls,
            model_path: str,
            device,
            eval_mode: bool = True,
            pre_activate: bool = True,
    ):
        load_from = cls.search_load_file(model_path)
        if load_from.endswith(".ckpt"):
            model, renderer, _ = cls.initialize_model_and_renderer_from_checkpoint_file(
                load_from,
                device=device,
                eval_mode=eval_mode,
                pre_activate=pre_activate,
            )
        elif load_from.endswith(".ply"):
            model, renderer = cls.initialize_model_and_renderer_from_ply_file(
                load_from,
                device=device,
                eval_mode=eval_mode,
                pre_activate=pre_activate,
            )
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer

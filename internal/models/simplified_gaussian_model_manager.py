import torch
import internal.utils.gaussian_utils as gaussian_utils
from .gaussian_model_simplified import GaussianModelSimplified


class SimplifiedGaussianModelManager:
    models: list = None

    def __init__(self, simplified_gaussian_models: list[GaussianModelSimplified], enable_transform: bool, device):
        super().__init__()
        if enable_transform is True:
            self.models = simplified_gaussian_models
        self.device = device

        # calculate total gaussian num
        total_gaussian_num = 0
        model_gaussian_indices = []
        for i in simplified_gaussian_models:
            n = i.get_xyz.shape[0]
            model_gaussian_indices.append((total_gaussian_num, total_gaussian_num + n))
            total_gaussian_num += n

        self.model_gaussian_indices = model_gaussian_indices

        # create tensor
        tensor_initialize_params = {
            "dtype": torch.float,
            "device": device,
        }
        self._xyz = torch.zeros((total_gaussian_num, 3), **tensor_initialize_params)
        self._opacity = torch.zeros((total_gaussian_num, 1), **tensor_initialize_params)
        self._features = torch.zeros([total_gaussian_num] + list(simplified_gaussian_models[0].get_features.shape[1:]), **tensor_initialize_params)
        self._scaling = torch.zeros((total_gaussian_num, 3), **tensor_initialize_params)
        self._rotation = torch.zeros((total_gaussian_num, 4), **tensor_initialize_params)

        # merge gaussians
        for idx, model in enumerate(simplified_gaussian_models):
            begin, end = self.model_gaussian_indices[idx]
            self._xyz[begin:end] = model.get_xyz.to(device)
            self._opacity[begin:end] = model.get_opacity.to(device)
            self._features[begin:end] = model.get_features.to(device)
            self._scaling[begin:end] = model.get_scaling.to(device)
            self._rotation[begin:end] = model.get_rotation.to(device)

        self.active_sh_degree = simplified_gaussian_models[0].active_sh_degree

    def get_model_gaussian_indices(self, idx: int):
        return self.model_gaussian_indices[idx]

    def get_model(self, idx: int) -> GaussianModelSimplified:
        return self.models[idx]

    def transform(
            self,
            idx: int,
            scale: float,
            rx: float,
            ry: float,
            rz: float,
            tx: float,
            ty: float,
            tz: float,
    ):
        model = self.get_model(idx)
        begin, end = self.get_model_gaussian_indices(idx)

        xyz = model.get_xyz.to(self.device)
        # TODO: avoid memory copy if no rotation or scaling happened compared to previous state
        scaling = model.get_scaling.to(self.device)
        rotation = model.get_rotation.to(self.device)

        if scale != 1.:
            xyz, scaling = gaussian_utils.GaussianTransformUtils.rescale(
                xyz,
                scaling,
                scale
            )

        if rx != 0. or ry != 0. or rz != 0.:
            xyz, rotation = gaussian_utils.GaussianTransformUtils.rotate(
                xyz,
                rotation,
                rx,
                ry,
                rz,
            )

        if tx != 0. or ty != 0. or tz != 0.:
            xyz = gaussian_utils.GaussianTransformUtils.translation(
                xyz,
                tx,
                ty,
                tz,
            )

        self._xyz[begin:end] = xyz
        self._scaling[begin:end] = scaling
        self._rotation[begin:end] = rotation

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return self._features

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def max_sh_degree(self):
        return self.active_sh_degree

import os
import numpy as np
import torch
from internal.utils.colmap import rotmat2qvec
from typing import Union
from dataclasses import dataclass
from plyfile import PlyData, PlyElement


@dataclass
class Gaussian:
    sh_degrees: int
    xyz: Union[np.ndarray, torch.Tensor]  # [n, 3]
    opacities: Union[np.ndarray, torch.Tensor]  # [n, 1]
    features_dc: Union[np.ndarray, torch.Tensor]  # [n, 3, 1], or [n, 1, 3]
    features_extra: Union[np.ndarray, torch.Tensor]  # [n, 3, 15], or [n, 15, 3]
    scales: Union[np.ndarray, torch.Tensor]  # [n, 3]
    rotations: Union[np.ndarray, torch.Tensor]  # [n, 4]

    @classmethod
    def load_from_ply(cls, path: str, sh_degrees: int):
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (sh_degrees + 1) ** 2 - 3  # TODO: remove such a assertion
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degrees + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return cls(
            sh_degrees=sh_degrees,
            xyz=xyz,
            opacities=opacities,
            features_dc=features_dc,
            features_extra=features_extra,
            scales=scales,
            rotations=rots,
        )

    def to_parameter_structure(self):
        return Gaussian(
            sh_degrees=self.sh_degrees,
            xyz=torch.tensor(self.xyz, dtype=torch.float),
            opacities=torch.tensor(self.opacities, dtype=torch.float),
            features_dc=torch.tensor(self.features_dc, dtype=torch.float).transpose(1, 2),
            features_extra=torch.tensor(self.features_extra, dtype=torch.float).transpose(1, 2),
            scales=torch.tensor(self.scales, dtype=torch.float),
            rotations=torch.tensor(self.rotations, dtype=torch.float),
        )

    def save_to_ply(self, path: str):
        gaussian = self

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = gaussian.xyz
        normals = np.zeros_like(xyz)
        f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
        # TODO: change sh degree
        if gaussian.sh_degrees > 0:
            f_rest = gaussian.features_extra.reshape((gaussian.features_extra.shape[0], -1))
        else:
            f_rest = np.zeros((f_dc.shape[0], 0))
        opacities = gaussian.opacities
        scale = gaussian.scales
        rotation = gaussian.rotations

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(gaussian.features_dc.shape[1] * gaussian.features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            if gaussian.sh_degrees > 0:
                for i in range(gaussian.features_extra.shape[1] * gaussian.features_extra.shape[2]):
                    l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(gaussian.scales.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(gaussian.rotations.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


class GaussianTransformUtils:
    @staticmethod
    def translation(xyz, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return

        return xyz + torch.tensor([[x, y, z]], device=xyz.device)

    @staticmethod
    def rescale(xyz, scaling, factor: float):
        return xyz * factor, scaling * factor

    @staticmethod
    def rx(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[1, 0, 0],
                             [0, torch.cos(theta), -torch.sin(theta)],
                             [0, torch.sin(theta), torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def ry(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                             [0, 1, 0],
                             [-torch.sin(theta), 0, torch.cos(theta)]], dtype=torch.float)

    @staticmethod
    def rz(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0],
                             [0, 0, 1]], dtype=torch.float)

    @classmethod
    def rotate(cls, xyz, rotation, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        # rotate
        rotation_matrix = cls.rx(x) @ cls.ry(y) @ cls.rz(z)
        xyz, rotation = cls.rotate_by_quaternion(
            xyz,
            rotation,
            rotation_matrix.to(xyz),
        )

        return xyz, rotation

    # @staticmethod
    # def rotate_by_matrix(xyz, rotations, rotation_matrix):
    #     # rotate xyz
    #     xyz = torch.matmul(xyz, rotation_matrix.T)
    #
    #     # rotate via rotation matrix
    #     gaussian_rotation = build_rotation(rotations).to(xyz.device)
    #     gaussian_rotation = rotation_matrix @ gaussian_rotation
    #     xyzw_quaternions = R.from_matrix(gaussian_rotation.cpu().numpy()).as_quat(canonical=False)
    #     wxyz_quaternions = xyzw_quaternions
    #     wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
    #     rotations_from_matrix = wxyz_quaternions
    #
    #     # TODO: rotate shs
    #
    #     return xyz, torch.tensor(rotations_from_matrix).to(xyz)

    @staticmethod
    def quat_multiply(quaternion0, quaternion1):
        w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
        w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
        return torch.concatenate((
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ), dim=-1)

    @classmethod
    def rotate_by_quaternion(cls, xyz, rotations, rotation_matrix):
        # rotate xyz
        xyz = torch.matmul(xyz, rotation_matrix.T)

        # rotate via quaternion
        rotations = torch.nn.functional.normalize(cls.quat_multiply(
            rotations,
            torch.tensor([rotmat2qvec(rotation_matrix.cpu().numpy())]).to(xyz),
        ))

        return xyz, rotations

import os
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
from internal.utils.colmap import rotmat2qvec, qvec2rotmat
from internal.utils.general_utils import build_rotation


@dataclass
class Gaussian:
    sh_degrees: int
    xyz: np.ndarray  # [n, 3]
    opacities: np.ndarray  # [n, 1]
    features_dc: np.ndarray  # [n, 3, 1]
    features_extra: np.ndarray  # [n, 3, 15]
    scales: np.ndarray  # [n, 3]
    rotations: np.ndarray  # [n, 4]

    @staticmethod
    def rx(theta):
        return np.matrix([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    @staticmethod
    def ry(theta):
        return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    @staticmethod
    def rz(theta):
        return np.matrix([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])

    def rescale(self, scale: float):
        if scale != 1.:
            self.xyz *= scale
            self.scales += np.log(scale)

            print("rescaled with factor {}".format(scale))

    def rotate(self, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        rotation_matrix = np.asarray(self.rx(x) @ self.ry(y) @ self.rz(z), dtype=np.float32)

        # rotate xyz
        self.xyz = np.asarray(np.matmul(self.xyz, rotation_matrix.T))

        # rotate gaussian
        # rotate via quaternions, seems not work correctly
        # def quat_multiply(quaternion0, quaternion1):
        #     x0, y0, z0, w0 = np.split(quaternion0, 4, axis=-1)
        #     x1, y1, z1, w1 = np.split(quaternion1, 4, axis=-1)
        #     return np.concatenate(
        #         (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        #          -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        #          x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        #          -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0),
        #         axis=-1)
        #
        # quaternions = rotmat2qvec(rotation_matrix)[np.newaxis, ...]
        # rotations_from_quats = quat_multiply(quaternions, self.rotations)
        # self.rotations = rotations_from_quats

        # rotate via rotation matrix
        gaussian_rotation = build_rotation(torch.from_numpy(self.rotations)).cpu()
        gaussian_rotation = torch.from_numpy(rotation_matrix) @ gaussian_rotation
        xyzw_quaternions = R.from_matrix(gaussian_rotation.numpy()).as_quat(canonical=False)
        wxyz_quaternions = xyzw_quaternions
        wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
        rotations_from_matrix = wxyz_quaternions
        #
        self.rotations = rotations_from_matrix

        # TODO: rotate shs
        print("set sh_degree=0 when rotation transform enabled")
        self.sh_degrees = 0

    def translation(self, x: float, y: float, z: float):
        if x == 0. and y == 0. and z == 0.:
            return

        self.xyz += np.asarray([x, y, z])
        print("translation transform applied")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    # TODO: support degrees > 1
    parser.add_argument("--sh-degrees", "--sh-degree", "-s", type=int, default=3)
    parser.add_argument("--new-sh-degrees", "--ns", type=int, default=-1)

    # translation
    parser.add_argument("--tx", type=float, default=0)
    parser.add_argument("--ty", type=float, default=0)
    parser.add_argument("--tz", type=float, default=0)

    # rotation in euler angeles
    parser.add_argument("--rx", type=float, default=0, help="in radians")
    parser.add_argument("--ry", type=float, default=0, help="in radians")
    parser.add_argument("--rz", type=float, default=0, help="in radians")

    # scale
    parser.add_argument("--scale", type=float, default=1)

    args = parser.parse_args()
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)

    return args


def load_ply(path: str, sh_degrees: int) -> Gaussian:
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
    assert len(extra_f_names) == 3 * (sh_degrees + 1) ** 2 - 3
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

    return Gaussian(
        sh_degrees=sh_degrees,
        xyz=xyz,
        opacities=opacities,
        features_dc=features_dc,
        features_extra=features_extra,
        scales=scales,
        rotations=rots,
    )


def save_ply(gaussian: Gaussian, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = gaussian.xyz
    normals = np.zeros_like(xyz)
    f_dc = gaussian.features_dc.reshape((gaussian.features_dc.shape[0], -1))
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


def main():
    args = parse_args()
    assert args.input != args.output
    assert args.sh_degrees >= 1 and args.sh_degrees <= 3
    assert args.scale > 0
    assert os.path.exists(args.input)

    gaussian = load_ply(args.input, args.sh_degrees)

    if args.new_sh_degrees >= 0:
        gaussian.sh_degrees = args.new_sh_degrees

    gaussian.translation(args.tx, args.ty, args.tz)
    gaussian.rotate(args.rx, args.ry, args.rz)
    gaussian.rescale(args.scale)

    save_ply(gaussian, args.output)

    print(args.output)


main()

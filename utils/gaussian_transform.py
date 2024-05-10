import add_pypath
import os
import argparse
import json
import numpy as np
import torch
from dataclasses import dataclass
from internal.utils.colmap import rotmat2qvec
from internal.utils.rotation import rotation_matrix
from internal.utils.sh_utils import SH2RGB, RGB2SH
import internal.utils.gaussian_utils


@dataclass
class Gaussian(internal.utils.gaussian_utils.Gaussian):
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

    def rotate_by_euler_angles(self, x: float, y: float, z: float):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        rotation_matrix = np.asarray(self.rx(x) @ self.ry(y) @ self.rz(z), dtype=np.float32)

        return self.rotate_by_matrix(rotation_matrix)

    def rotate_by_matrix(self, rotation_matrix, keep_sh_degree: bool = True):
        # rotate xyz
        self.xyz = np.asarray(np.matmul(self.xyz, rotation_matrix.T))

        # rotate gaussian
        # rotate via quaternions
        def quat_multiply(quaternion0, quaternion1):
            w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
            w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
            return np.concatenate((
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            ), axis=-1)

        quaternions = rotmat2qvec(rotation_matrix)[np.newaxis, ...]
        rotations_from_quats = quat_multiply(self.rotations, quaternions)
        self.rotations = rotations_from_quats / np.linalg.norm(rotations_from_quats, axis=-1, keepdims=True)

        # rotate via rotation matrix
        # gaussian_rotation = build_rotation(torch.from_numpy(self.rotations)).cpu()
        # gaussian_rotation = torch.from_numpy(rotation_matrix) @ gaussian_rotation
        # xyzw_quaternions = R.from_matrix(gaussian_rotation.numpy()).as_quat(canonical=False)
        # wxyz_quaternions = xyzw_quaternions
        # wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
        # rotations_from_matrix = wxyz_quaternions
        # self.rotations = rotations_from_matrix

        # TODO: rotate shs
        if keep_sh_degree is False:
            print("set sh_degree=0 when rotation transform enabled")
            self.sh_degrees = 0
        else:
            """
            If the scene has been rotated, 
            in order to get the correct color as without rotation,
            we should rotate the view direction opposite.
            E.g., 
                the scene is rotated by R: means3D' = R @ means3D, 
                then when calculating colors, 
                the correct view direction should be: R.T @ (means3D' - camera_center).
                
            Based on the how the SHs multiple with view directions, we know how to rotate the SHs.
            """
            # rotate sh_degree=1 if exists
            if self.sh_degrees > 0:
                print("rotate sh_degree=1")
                degree_1 = self.features_rest[..., :3]  # [n, 3-rgb, 3-coefficients], 3 coefficients per-channel
                rotation_matrix_inverse = rotation_matrix.T
                rotation_matrix_inverse_reorder = rotation_matrix_inverse[[1, 2, 0], :][:, [1, 2, 0]]
                sign_matrix = np.asarray([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                ], dtype=np.float32)
                rotated_degree_1 = degree_1 @ sign_matrix @ rotation_matrix_inverse_reorder @ sign_matrix
                self.features_rest[..., :3] = rotated_degree_1

                # TODO: rotate higher degree SHs

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

    parser.add_argument("--sh-factor", type=float, default=1.0)

    # auto reorient
    parser.add_argument("--auto-reorient", action="store_true", default=False)
    parser.add_argument("--cameras-json", type=str, default=None)

    parser.add_argument("--appearance-id", type=float, default=-1)

    args = parser.parse_args()
    args.input = os.path.expanduser(args.input)
    args.output = os.path.expanduser(args.output)

    return args


def main():
    args = parse_args()
    assert args.input != args.output
    assert args.sh_degrees >= 0 and args.sh_degrees <= 3
    assert args.scale > 0
    assert os.path.exists(args.input)

    if args.input.endswith(".ply"):
        gaussian = Gaussian.load_from_ply(args.input, args.sh_degrees)
    else:
        checkpoint = torch.load(args.input)
        gaussian = Gaussian.load_from_state_dict(checkpoint["hyper_parameters"]["gaussian"].sh_degree, checkpoint["state_dict"]).to_ply_format()
        if args.appearance_id >= 0.:
            args.new_sh_degree = 0
            with torch.no_grad():
                gray_scale_factor, gamma = checkpoint["hyper_parameters"]["renderer"].appearance_model.get_appearance(args.appearance_id)
            gray_scale_factor = torch.reshape(gray_scale_factor, (1, -1, 1))
            rgb = SH2RGB(gaussian.features_dc).clip(min=0.)  # [n, 3, 1]
            rgb *= gray_scale_factor.cpu().numpy()
            rgb = np.power(rgb + 1e-5, float(gamma.reshape((-1,))[0].cpu()))
            gaussian.features_dc = RGB2SH(rgb)

    if args.auto_reorient is True:
        cameras_json_path = args.cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.input))), "cameras.json")
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = np.zeros(3)
        for i in cameras:
            up += np.asarray(i["rotation"])[:3, 1]
        up = -up / np.linalg.norm(up)
        print("up vector = {}".format(up))

        rotation = rotation_matrix(torch.tensor(up, dtype=torch.float), torch.Tensor([0, 0, 1])).numpy()

        print("rotation matrix = {}".format(rotation))

        gaussian.rotate_by_matrix(rotation)
    else:
        gaussian.rescale(args.scale)
        gaussian.rotate_by_euler_angles(args.rx, args.ry, args.rz)
        gaussian.translation(args.tx, args.ty, args.tz)

    if args.new_sh_degrees >= 0:
        gaussian.sh_degrees = args.new_sh_degrees

    if args.sh_factor != 1.:
        gaussian.features_dc *= args.sh_factor
        gaussian.features_rest *= args.sh_factor

    gaussian.save_to_ply(args.output)

    print(args.output)


main()

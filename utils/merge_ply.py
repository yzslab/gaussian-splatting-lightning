import os
import argparse
import numpy as np
import open3d as o3d

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, nargs="+")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

xyz_list = []
rgb_list = []
with tqdm(args.input) as t:
    for ply in t:
        t.set_description("Loading {}".format(ply))
        point_cloud = o3d.io.read_point_cloud(ply)
        xyz, rgb = np.asarray(point_cloud.points), (np.asarray(point_cloud.colors))
        xyz_list.append(xyz)
        rgb_list.append(rgb)
xyz = np.concatenate(xyz_list, axis=0)
rgb = np.concatenate(rgb_list, axis=0)
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(xyz)
final_pcd.colors = o3d.utility.Vector3dVector(rgb)
o3d.io.write_point_cloud(args.output, final_pcd)

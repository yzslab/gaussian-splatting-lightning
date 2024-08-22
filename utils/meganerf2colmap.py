import add_pypath
import os
import sys
import argparse
import subprocess
import numpy as np
import torch
import sqlite3
from internal.utils import colmap


def array_to_blob(array):
    return array.tostring()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--vocab_tree_path", "-v", type=str,
                        default=os.path.expanduser("~/.cache/colmap/vocab_tree_flickr100K_words256K.bin"))
    args = parser.parse_args()

    assert os.path.exists(args.vocab_tree_path)

    return args


def main():
    args = parse_args()

    coordinates = torch.load(os.path.join(args.path, "coordinates.pt"), map_location="cpu")

    colmap_dir = os.path.join(args.path, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    image_metadata_pairs = []
    for split in ["train", "val"]:
        for i in os.scandir(os.path.join(args.path, split, "rgbs")):
            name_without_ext = i.name.split(".")[0]
            image_metadata_pairs.append((
                i.path,
                os.path.join(args.path, split, "metadata", "{}.pt".format(name_without_ext)),
                i.name,
            ))

    image_dir = os.path.join(colmap_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for i, _, image_name in image_metadata_pairs:
        try:
            os.symlink(i, os.path.join(image_dir, image_name))
        except FileExistsError:
            pass

    colmap_db_path = os.path.join(colmap_dir, "colmap.db")
    assert subprocess.call([
        "colmap",
        "feature_extractor",
        "--database_path", colmap_db_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", "PINHOLE",
    ]) == 0

    colmap_db = sqlite3.connect(colmap_db_path)

    def select_image(image_name: str):
        cur = colmap_db.cursor()
        try:
            return cur.execute("SELECT image_id, camera_id FROM images WHERE name = ?", [image_name]).fetchone()
        finally:
            cur.close()

    def update_camera_params(camera_id: int, params: np.ndarray):
        cur = colmap_db.cursor()
        try:
            cur.execute("UPDATE cameras SET params = ? WHERE camera_id = ?", [
                array_to_blob(params),
                camera_id,
            ])
            colmap_db.commit()
        finally:
            cur.close()

    images = {}
    cameras = {}
    points = {}

    c2w_transform = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float).T
    RDF_TO_DRB_H = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    for _, metadata_path, image_name in image_metadata_pairs:
        metadata = torch.load(metadata_path, map_location="cpu")
        image_id, camera_id = select_image(image_name)
        update_camera_params(camera_id, metadata["intrinsics"].to(torch.float64).numpy())

        c2w = torch.eye(4)
        c2w[:3, :] = metadata["c2w"]

        c2w[:3, 3] *= coordinates["pose_scale_factor"]
        c2w[:3, 3] += coordinates["origin_drb"]

        c2w = torch.linalg.inv(RDF_TO_DRB_H) @ c2w @ c2w_transform @ RDF_TO_DRB_H
        w2c = torch.linalg.inv(c2w)

        images[image_id] = colmap.Image(
            image_id,
            qvec=colmap.rotmat2qvec(w2c[:3, :3].numpy()),
            tvec=w2c[:3, 3].numpy(),
            camera_id=camera_id,
            name=image_name,
            xys=np.array([], dtype=np.float64),
            point3D_ids=np.asarray([], dtype=np.int64),
        )
        cameras[camera_id] = colmap.Camera(
            id=camera_id,
            model="PINHOLE",
            width=metadata["W"],
            height=metadata["H"],
            params=metadata["intrinsics"].to(torch.float64).numpy(),
        )

    colmap_db.close()

    sparse_manually_model_dir = os.path.join(colmap_dir, "sparse_manually")
    os.makedirs(sparse_manually_model_dir, exist_ok=True)
    colmap.write_images_binary(images, os.path.join(sparse_manually_model_dir, "images.bin"))
    colmap.write_cameras_binary(cameras, os.path.join(sparse_manually_model_dir, "cameras.bin"))
    colmap.write_points3D_binary(points, os.path.join(sparse_manually_model_dir, "points3D.bin"))

    assert subprocess.call([
        "colmap",
        "vocab_tree_matcher",
        "--database_path", colmap_db_path,
        "--VocabTreeMatching.vocab_tree_path", args.vocab_tree_path,
    ]) == 0

    sparse_dir = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    assert subprocess.call([
        "colmap",
        "point_triangulator",
        "--database_path", colmap_db_path,
        "--image_path", image_dir,
        "--input_path", sparse_manually_model_dir,
        "--output_path", sparse_dir,
    ]) == 0


def test_main():
    sys.argv = [__file__, os.path.expanduser("~/data/Mega-NeRF/rubble-pixsfm")]
    main()


if __name__ == "__main__":
    main()

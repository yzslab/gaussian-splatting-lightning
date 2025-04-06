"""
Remove duplicated images with colmap's sequential_matcher
"""

import os
import sqlite3
import numpy as np
import argparse


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


class RemoveDuplicatedImages:
    def __init__(self, conn):
        self.conn = conn

    def get_image_name_list(self):
        image_name_list = []
        image_id_to_name = {}
        for i in self.conn.execute("SELECT * FROM images ORDER BY name"):
            image_name_list.append(i[1])
            image_id_to_name[i[0]] = i[1]

        return image_name_list, image_id_to_name

    def get_matched_keypoints(self, image_id_to_name):
        matched_keypoints = {}
        for i in self.conn.execute("SELECT * FROM two_view_geometries"):
            image_id1, image_id2 = pair_id_to_image_ids(i[0])
            image_name_pair = [image_id_to_name[image_id1], image_id_to_name[image_id2]]
            # print("({}, {}) -> {}".format(image_id1, image_id2, image_name_pair))
            if i[3] is None:
                # print("Skip {}".format(image_name_pair))
                continue
            matched_keypoints[tuple(image_name_pair)] = np.frombuffer(i[3], dtype=np.uint32).reshape(i[1], i[2])
        len(matched_keypoints)

        return matched_keypoints

    def get_keypoint_coordinates(self, image_id_to_name):
        keypoint_coordinates = {}
        for i in self.conn.execute("SELECT * FROM keypoints"):
            image_name = image_id_to_name[i[0]]
            keypoint_data = np.frombuffer(i[-1], dtype=np.float32).reshape((i[1], i[2]))
            keypoint_coordinates[image_name] = keypoint_data[:, :2]

        return keypoint_coordinates

    def get_image_pair_offsets(
            self,
            image_name_list,
            matched_keypoints,
            keypoint_coordinates,
            min_offset: float = 100.,
    ):
        image_pair_offsets = {}
        keep_image_name_list = [image_name_list[0]]
        acc_offsets = 0.
        for idx in range(1, len(image_name_list)):
            cur_image_name = image_name_list[idx]
            pre_image_name = image_name_list[idx - 1]
            name_pair = (pre_image_name, cur_image_name)
            try:
                matched = matched_keypoints[name_pair]
            except KeyError:
                name_pair = (cur_image_name, pre_image_name)
                try:
                    matched = matched_keypoints[name_pair]
                except KeyError:
                    # no matching, regard as high offset
                    keep_image_name_list.append(cur_image_name)
                    acc_offsets = 0.
                    continue

            coordinates1 = keypoint_coordinates[name_pair[0]][matched[:, 0]]
            coordinates2 = keypoint_coordinates[name_pair[1]][matched[:, 1]]
            offsets = np.sqrt(np.sum((coordinates2 - coordinates1)**2, axis=-1))

            mean_offset = offsets.mean()
            acc_offsets = acc_offsets + mean_offset
            if acc_offsets > min_offset:
                keep_image_name_list.append(cur_image_name)
                acc_offsets = 0.
            image_pair_offsets[name_pair] = mean_offset
        return image_pair_offsets, keep_image_name_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("colmap_db")
    parser.add_argument("image_dir")
    parser.add_argument("output")
    parser.add_argument("--min-offset", type=float, default=100.)
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.realpath(args.image_dir) != os.path.realpath(args.output)

    conn = sqlite3.connect(args.colmap_db)

    r = RemoveDuplicatedImages(conn)

    image_name_list, image_id_to_name = r.get_image_name_list()
    matched_keypoints = r.get_matched_keypoints(image_id_to_name)
    keypoint_coordinates = r.get_keypoint_coordinates(image_id_to_name)
    image_pair_offsets, keep_image_name_list = r.get_image_pair_offsets(
        image_name_list,
        matched_keypoints,
        keypoint_coordinates,
        min_offset=args.min_offset,
    )

    os.makedirs(args.output, exist_ok=True)

    # remove existing links
    for i in os.scandir(args.output):
        if i.is_symlink():
            os.unlink(i.path)

    for i in keep_image_name_list:
        os.symlink(
            os.path.join(args.image_dir, i),
            os.path.join(args.output, i)
        )

    print("{} of {} images linked to '{}'".format(
        len(keep_image_name_list),
        len(image_name_list),
        args.output,
    ))


if __name__ == "__main__":
    main()

import add_pypath
import os
from glob import glob
from tqdm.auto import tqdm
import numpy as np
import cv2
import torch
import argparse
from utils.distibuted_tasks import configure_arg_parser_v2, get_task_list_with_args_v2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--dist", "-d", type=float, default=0.1,
                        help="Image size ratio. Images with content motion distance below this ratio will be treated as redundant.")
    parser.add_argument("--ratio", "-r", type=float, default=0.3,
                        help="Images when a ratio of keypoints whose motion exceeds the threshold below this will be treated as redundant")
    parser.add_argument("--max-size", type=int, default=1024)
    configure_arg_parser_v2(parser)
    return parser.parse_args()


def load_xfeat():
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
    return xfeat


def extract_keypoints(xfeat, image_dir, image_name, max_size, top_k: int = 4096):
    # load image
    image = cv2.imread(os.path.join(image_dir, image_name))

    # resize if exceed max_size
    image_current_max_size = max(image.shape[0], image.shape[1])
    if max_size > 0 and image_current_max_size > max_size:
        down_size = image_current_max_size / max_size
        image = cv2.resize(
            image,
            dsize=(round(image.shape[1] / down_size), round(image.shape[0] / down_size)),
        )

    # inference
    output = xfeat.detectAndCompute(image, top_k=top_k)[0]
    output.update({'image_size': (image.shape[1], image.shape[0]), "image": image})  # W, H

    return output


def dump_image_list(f, image_list):
    for i in image_list:
        f.write(i)
        f.write("\n")


@torch.no_grad()
def main():
    args = get_args()

    # find images
    cwd = os.getcwd()
    os.chdir(args.image_dir)
    image_name_list = sorted(glob("**/*.[jJ][pP][gG]", recursive=True))
    os.chdir(cwd)
    n_found = len(image_name_list)
    image_name_list = get_task_list_with_args_v2(args, image_name_list)
    n_assigned = len(image_name_list)
    print("{}/{} images".format(n_assigned, n_found))

    # load xfeat
    xfeat = load_xfeat()

    def extract_keypoints_simplified(image_name):
        return extract_keypoints(xfeat, args.image_dir, image_name, max_size=args.max_size)

    # initialize state
    redundant_frames = []
    keyframes = [image_name_list[0]]
    previous_frame_keypoints = extract_keypoints_simplified(image_name_list[0])

    with tqdm(image_name_list[1:]) as t:
        for image_name in t:
            keypoints = extract_keypoints_simplified(image_name)
            xy0, xy1, _ = xfeat.match_lighterglue(previous_frame_keypoints, keypoints)
            normalized_dist = np.linalg.norm(xy0 / np.asarray(previous_frame_keypoints["image_size"]) - xy1 / np.asarray(keypoints["image_size"]), axis=-1)
            over_threshold_mask = normalized_dist > args.dist
            over_threshold_ratio = over_threshold_mask.sum() / (over_threshold_mask.shape[0] + 1)
            if xy0.shape[0] < 32 or over_threshold_ratio > args.ratio:
                tqdm.write(image_name)
                keyframes.append(image_name)
                previous_frame_keypoints = keypoints
                t.set_description("{}/{}".format(len(keyframes), len(image_name_list)))
            else:
                redundant_frames.append(image_name)

    # save
    name_suffix = ""
    if n_found != n_assigned:
        name_suffix = "-{:02d}".format(int(os.environ.get("CURRENT_PROCESSOR_ID", None)))
    abs_image_dir = os.path.abspath(args.image_dir).rstrip(os.path.sep)
    redundant_list_path = "{}-redundant_frames{}.txt".format(abs_image_dir, name_suffix)
    with open(redundant_list_path, "w") as f:
        dump_image_list(f, redundant_frames)
    keyframe_list_path = "{}-keyframes{}.txt".format(abs_image_dir, name_suffix)
    with open(keyframe_list_path, "w") as f:
        dump_image_list(f, keyframes)
    print(redundant_list_path)
    print(keyframe_list_path)


if __name__ == "__main__":
    main()

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
    parser.add_argument("input")
    parser.add_argument("--dist", "-d", type=float, default=0.1,
                        help="Image size ratio. Images with content motion distance below this ratio will be treated as redundant.")
    parser.add_argument("--ratio", "-r", type=float, default=0.3,
                        help="Images when a ratio of keypoints whose motion exceeds the threshold below this will be treated as redundant")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=10,
                        help="The FPS for extracting frames from the video")
    configure_arg_parser_v2(parser)
    return parser.parse_args()


class XFeatUtils:
    def __init__(
            self,
            max_size: int,
            top_k: int = 4096,
    ):
        self.max_size = max_size
        self.top_k = top_k
        self.xfeat = self.load_xfeat()

    def extract_keypoints(self, image):
        image = self.resize_if_oversize(image)

        output = self.xfeat.detectAndCompute(image, top_k=self.top_k)[0]
        output.update({'image_size': (image.shape[1], image.shape[0])})  # W, H

        return output

    def extract_keypoints_with_image_path(self, image_path):
        image = cv2.imread(image_path)
        return image, self.extract_keypoints(image)

    def match_keypoints(self, output0, output1, normalize: bool = True):
        xy0, xy1, indices = self.xfeat.match_lighterglue(output0, output1)

        if normalize:
            xy0 = xy0 / np.asarray(output0["image_size"])
            xy1 = xy1 / np.asarray(output1["image_size"])

        return xy0, xy1, indices

    def resize_if_oversize(self, image):
        max_size = self.max_size
        image_current_max_size = max(image.shape[0], image.shape[1])
        if max_size > 0 and image_current_max_size > max_size:
            down_size = image_current_max_size / max_size
            image = cv2.resize(
                image,
                dsize=(round(image.shape[1] / down_size), round(image.shape[0] / down_size)),
            )

        return image

    @staticmethod
    def load_xfeat():
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        return xfeat


def dump_image_list(f, image_list):
    for i in image_list:
        f.write(i)
        f.write("\n")


def dir_mode(args, is_keyframe):
    # find images
    cwd = os.getcwd()
    os.chdir(args.input)
    image_name_list = sorted(glob("**/*.[jJ][pP][gG]", recursive=True))
    os.chdir(cwd)
    n_found = len(image_name_list)
    image_name_list = get_task_list_with_args_v2(args, image_name_list)
    n_assigned = len(image_name_list)
    print("{}/{} images".format(n_assigned, n_found))

    # load xfeat
    xfeat_utils = XFeatUtils(
        max_size=args.max_size,
    )

    # initialize state
    redundant_frames = []
    keyframes = [image_name_list[0]]
    _, previous_frame_keypoints = xfeat_utils.extract_keypoints_with_image_path(os.path.join(args.input, image_name_list[0]))

    with tqdm(image_name_list[1:]) as t:
        for image_name in t:
            _, keypoints = xfeat_utils.extract_keypoints_with_image_path(os.path.join(args.input, image_name))
            xy0, xy1, _ = xfeat_utils.match_keypoints(previous_frame_keypoints, keypoints)
            if is_keyframe(xy0, xy1):
                tqdm.write(image_name)
                keyframes.append(image_name)
                previous_frame_keypoints = keypoints
                t.set_description("{}/{}".format(len(keyframes), len(image_name_list)))
            else:
                redundant_frames.append(image_name)

    # save image list
    name_suffix = ""
    if n_found != n_assigned:
        name_suffix = "-{:02d}".format(int(os.environ["CURRENT_PROCESSOR_ID"]))
    abs_image_dir = os.path.abspath(args.input).rstrip(os.path.sep)
    redundant_list_path = "{}-redundant_frames{}.txt".format(abs_image_dir, name_suffix)
    with open(redundant_list_path, "w") as f:
        dump_image_list(f, redundant_frames)
    keyframe_list_path = "{}-keyframes{}.txt".format(abs_image_dir, name_suffix)
    with open(keyframe_list_path, "w") as f:
        dump_image_list(f, keyframes)
    print(redundant_list_path)
    print(keyframe_list_path)


def video_mode(args, is_keyframe):
    import subprocess
    import math
    import mediapy
    from mediapy import (
        _get_ffmpeg_path,
        _read_via_local_file,
        _get_video_metadata,
    )
    from utils.common import AsyncImageSaver

    class VideoReader(mediapy.VideoReader):
        def __init__(
            self,
            path_or_url: str,
            read_fps: int = -1,
            *args,
            **kwargs,
        ):
            self.read_fps = read_fps
            super().__init__(path_or_url=path_or_url, *args, **kwargs)

        def __enter__(self) -> 'VideoReader':
            ffmpeg_path = _get_ffmpeg_path()
            try:
                self._read_via_local_file = _read_via_local_file(self.path_or_url)
                # pylint: disable-next=no-member
                tmp_name = self._read_via_local_file.__enter__()

                self.metadata = _get_video_metadata(tmp_name)
                self.num_images, self.shape, self.fps, self.bps = self.metadata
                pix_fmt = self._get_pix_fmt(self.dtype, self.output_format)
                num_channels = {'rgb': 3, 'yuv': 3, 'gray': 1}[self.output_format]
                bytes_per_channel = self.dtype.itemsize
                self._num_bytes_per_image = (
                    math.prod(self.shape) * num_channels * bytes_per_channel
                )

                read_fps_arg = []
                if self.read_fps > 0:
                    read_fps_arg = [
                        '-vf',
                        "fps={}".format(self.read_fps),
                    ]

                command = [
                    ffmpeg_path,
                    '-v',
                    'panic',
                    '-nostdin',
                    '-i',
                    tmp_name,
                ] + read_fps_arg + [
                    '-vcodec',
                    'rawvideo',
                    "-qmin", "1",
                    "-q:v", "1",
                    '-f',
                    'image2pipe',
                    '-pix_fmt',
                    pix_fmt,
                    '-vsync',
                    'vfr',
                    '-',
                ]
                self._popen = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                self._proc = self._popen.__enter__()
            except Exception:
                self.__exit__(None, None, None)
                raise
            return self

    # load xfeat
    xfeat_utils = XFeatUtils(
        max_size=args.max_size,
    )

    output_dir = "{}-keyframes".format(args.input)
    os.makedirs(output_dir, exist_ok=True)
    # remove existing images
    for i in os.scandir(output_dir):
        if i.is_file() and i.name.endswith(".jpg"):
            os.unlink(i.path)

    image_saver = AsyncImageSaver(is_rgb=True)

    def save_keyframe(idx: int, image):
        image_saver.save(
            image,
            path=os.path.join(
                output_dir,
                "{:06d}.jpg".format(idx)
            )
        )
    try:
        with VideoReader(
            args.input,
            read_fps=args.fps,
        ) as reader:
            print(f'Video has {reader.num_images} images with shape={reader.shape},'
                  f' at {reader.fps} frames/sec and {reader.bps} bits/sec.')

            t = tqdm(reader)
            t_iter = iter(t)
            first_frame = next(t_iter)
            previous_frame_keypoints = xfeat_utils.extract_keypoints(first_frame)
            save_keyframe(0, first_frame)
            n_keyframes = 1

            for frame in t_iter:
                keypoints = xfeat_utils.extract_keypoints(frame)
                xy0, xy1, _ = xfeat_utils.match_keypoints(
                    output0=previous_frame_keypoints,
                    output1=keypoints,
                )
                if not is_keyframe(xy0, xy1):
                    continue
                save_keyframe(n_keyframes, frame)
                previous_frame_keypoints = keypoints
                n_keyframes += 1
                t.set_description("{} keyframes extracted".format(n_keyframes))
    finally:
        image_saver.stop()

    print(output_dir)


@torch.no_grad()
def main():
    args = get_args()

    def is_keyframe(xy0, xy1):
        normalized_dist = np.linalg.norm(xy0 - xy1, axis=-1)
        over_threshold_mask = normalized_dist > args.dist
        over_threshold_ratio = over_threshold_mask.sum() / (over_threshold_mask.shape[0] + 1)
        return xy0.shape[0] < 32 or over_threshold_ratio > args.ratio

    if os.path.isdir(args.input):
        dir_mode(args, is_keyframe)
    else:
        video_mode(args, is_keyframe)


if __name__ == "__main__":
    main()

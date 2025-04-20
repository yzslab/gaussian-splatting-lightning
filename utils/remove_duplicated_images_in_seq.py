"""
Remove duplicated images from a video frame sequence
"""

import os
import argparse
import hashlib
from glob import glob
from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, rbd
import torch
from queue import Queue
from threading import Thread
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


class RemoveDuplicatedImages:
    def __init__(
            self,
            image_dir: str,
            feature_output_dir: str,
            device,
    ):
        self.image_dir = image_dir
        self.feature_output_dir = feature_output_dir
        self.device = torch.device(device)

    def create_extractor_and_matcher(self):
        self.extractor = ALIKED(max_num_keypoints=4096).eval().to(device=self.device)
        self.matcher = LightGlue(
            features="aliked",
            depth_confidence=0.9,
            width_confidence=0.95,
        ).eval().to(device=self.device)

    def get_image_name_list(self):
        image_dir = self.image_dir
        image_list = [i[len(image_dir):].lstrip("/") for i in glob(os.path.join(image_dir, "**", "*.[jJ][pP][gG]"), recursive=True)]
        self.image_list = sorted(image_list)

    def get_feature_file_path(self, image_name):
        return os.path.join(self.feature_output_dir, "{}.pt".format(image_name))

    def start_image_loader_thread(self):
        image_queue = Queue(maxsize=16)

        def load_image_to_queue(image_name: str):
            image_queue.put((
                image_name,
                load_image(os.path.join(self.image_dir, image_name)),
            ))

        image_list = []
        for i in self.image_list:
            if os.path.exists(self.get_feature_file_path(i)):
                continue
            image_list.append(i)

        def concurrent_image_loader():
            # max_workers > 6 will strangely freeze the program
            with ThreadPoolExecutor(max_workers=6) as tpe:
                for _ in tpe.map(load_image_to_queue, image_list):
                    pass
                image_queue.put((None, None))

        image_loader_thread = Thread(target=concurrent_image_loader)
        image_loader_thread.start()

        return image_queue, image_loader_thread, image_list

    @torch.no_grad()
    def extract_features(self):
        image_queue, image_loader_thread, image_list = self.start_image_loader_thread()

        with tqdm(image_list, desc="Extracting features...") as t:
            for _ in t:
                image_name, image = image_queue.get()

                image = image.to(device=self.device)
                features = self.extractor.extract(image)
                save_to = self.get_feature_file_path(image_name)
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                torch.save(features, "{}.tmp".format(save_to))
                os.rename("{}.tmp".format(save_to), save_to)

        assert image_queue.get(timeout=1)[0] is None
        image_loader_thread.join()

    @torch.no_grad()
    def remove_duplication(self, min_dist):
        keep_image_list = [self.image_list[0]]

        with torch.no_grad():
            previous = torch.load(
                os.path.join(self.feature_output_dir, "{}.pt".format(self.image_list[0])),
                map_location=self.device,
                weights_only=False,
            )

            for i in tqdm(self.image_list[1:], desc="Filtering..."):
                current = torch.load(
                    os.path.join(self.feature_output_dir, "{}.pt".format(i)),
                    map_location=self.device,
                    weights_only=False,
                )

                matches = self.matcher({
                    "image0": previous,
                    "image1": current,
                })

                feats0, feats1, matches = [rbd(x) for x in [previous, current, matches]]  # remove batch dimension
                matches = matches['matches']  # indices with shape (K,2)
                points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
                points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

                dist = torch.linalg.norm(points0 - points1, dim=-1).mean()
                if dist < min_dist or points0.shape[0] == 0:
                    continue

                keep_image_list.append(i)

                previous = current

        return keep_image_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("output")
    parser.add_argument("--min-offset", type=float, default=256.)
    parser.add_argument("--feature-output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    assert os.path.realpath(args.image_dir) != os.path.realpath(args.output)

    if args.feature_output is None:
        args.feature_output = os.path.join(
            os.path.dirname(__file__), "extracted_image_features",
            hashlib.sha256(os.path.realpath(args.image_dir).encode()).hexdigest(),
        )
    os.makedirs(args.feature_output, exist_ok=True)

    r = RemoveDuplicatedImages(
        image_dir=args.image_dir,
        feature_output_dir=args.feature_output,
        device="cuda",
    )

    r.get_image_name_list()
    r.create_extractor_and_matcher()
    r.extract_features()
    image_list = r.remove_duplication(args.min_offset)

    os.makedirs(args.output, exist_ok=True)

    # remove existing links
    for i in os.scandir(args.output):
        if i.is_symlink():
            os.unlink(i.path)

    for i in image_list:
        os.symlink(
            os.path.join(args.image_dir, i),
            os.path.join(args.output, i)
        )

    print("{} of {} images linked to '{}'".format(
        len(image_list),
        len(r.image_list),
        args.output,
    ))


if __name__ == "__main__":
    main()

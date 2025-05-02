import add_pypath
import os
from glob import glob
import argparse

from common import find_files

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("--extensions", "-e", default=["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"])
args = parser.parse_args()


estimated_depth_dir = os.path.join(os.path.dirname(args.image_dir), "estimated_depths")
found_image_list = find_files(args.image_dir, args.extensions, as_relative_path=True)

counter = 0
image_list_path = os.path.join(os.path.dirname(args.image_dir), "images_without_depth_estimates.txt")
with open(image_list_path, "w") as f:
    for i in found_image_list:
        file_path_without_ext = os.path.join(estimated_depth_dir, "{}".format(i))
        if not os.path.exists("{}.npy".format(file_path_without_ext)) and not os.path.exists("{}.uint16.png".format(file_path_without_ext)):
            f.write("{}\n".format(i))
            counter += 1

print("{}/{}".format(counter, len(found_image_list)))
print("python utils/run_depth_anything_v2.py {} --image_list={}".format(args.image_dir, image_list_path))

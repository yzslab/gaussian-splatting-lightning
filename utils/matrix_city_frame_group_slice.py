import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--slice", type=str, nargs="+",
                    help="start,count")
parser.add_argument("--camera-count", "-c", type=int, default=6)
args = parser.parse_args()
assert args.input != args.output

with open(args.input, "r") as f:
    transforms = json.load(f)

assert len(transforms["frames"]) % args.camera_count == 0

selected_frames = []
for i in args.slice:
    start_count = i.split(",")
    start = int(start_count[0])
    count = int(start_count[1])

    index_left = start * args.camera_count
    index_right = index_left + args.camera_count * count
    selected_frames += transforms["frames"][index_left:index_right]

# check overlap
selected_frame_index = {}
for i in selected_frames:
    frame_index = i["frame_index"]
    assert frame_index not in selected_frame_index
    selected_frame_index[frame_index] = True

transforms["frames"] = selected_frames
with open(args.output, "w") as f:
    json.dump(transforms, f, indent=4, ensure_ascii=False)

import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--camera-count", "-c", type=int, default=6)
    args = parser.parse_args()

    assert args.input != args.output

    return args


def load_json(path: str):
    with open(path, "r") as f:
        transforms = json.load(f)
    return transforms


def build_first_camera_continuous_range(frame_list: list, camera_count: int) -> list:
    frame_count = len(frame_list)

    assert len(frame_list) % camera_count == 0

    group_list = []
    previous_orientation = np.asarray(frame_list[0]["rot_mat"])[:3, :3]
    from_idx = 0
    idx = 0
    while idx < frame_count:
        orientation = np.asarray(frame_list[idx]["rot_mat"])[:3, :3]
        if not np.all(orientation == previous_orientation):
            group_list.append((from_idx, idx - 1))  # record range
            idx += (camera_count - 1) * (idx - from_idx)  # skip to the end of this continuous frames

            from_idx = idx  # reset from_idx to the new continuous frames
            try:
                orientation = np.asarray(frame_list[idx]["rot_mat"])[:3, :3]
            except IndexError:
                break
        previous_orientation = orientation
        idx += 1

    return group_list


def rearrange_frames(first_camera_continuous_frames: list, frame_list: list, camera_count: int) -> list:
    frames = []
    for begin, end in first_camera_continuous_frames:
        for i in range(begin, end + 1):
            continuous_frame_count = end - begin + 1
            for camera_id in range(camera_count):
                frames.append(frame_list[i + camera_id * continuous_frame_count])

    return frames


if __name__ == "__main__":
    args = parse_args()
    transforms = load_json(args.input)
    frame_list = transforms["frames"]
    first_camera_continuous_frames = build_first_camera_continuous_range(frame_list, args.camera_count)
    print(first_camera_continuous_frames)
    rearranged_frames = rearrange_frames(first_camera_continuous_frames, frame_list, args.camera_count)
    assert len(rearranged_frames) == len(frame_list)

    transforms["frames"] = rearranged_frames
    transforms["continuous"] = [i[0] for i in first_camera_continuous_frames]
    with open(args.output, "w") as f:
        json.dump(transforms, f, indent=4, ensure_ascii=False)

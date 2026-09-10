#!/usr/bin/env python
"""Validate the feature contract of a real-world LeRobot dataset."""

import argparse
import json
from pathlib import Path

from _real_common import load_camera_calibration, load_real_dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/real-world")
    parser.add_argument("--image-key", default="observation.images.cam_3")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--decode-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata, input_features, output_features = load_real_dataset(
        dataset_dir,
        args.repo_id,
        args.image_key,
    )
    image_shape = tuple(input_features[args.image_key].shape)
    state_shape = tuple(input_features["observation.state"].shape)
    action_shape = tuple(output_features["action"].shape)
    if state_shape[0] < 3 or action_shape[0] < 3:
        raise ValueError("SCDP needs at least three positional state and action dimensions.")

    report = {
        "dataset_dir": str(dataset_dir),
        "fps": metadata.fps,
        "image_key": args.image_key,
        "image_shape_chw": image_shape,
        "state_shape": state_shape,
        "action_shape": action_shape,
    }
    if args.camera_calibration:
        calibration = load_camera_calibration(args.camera_calibration.expanduser().resolve())
        calibrated_size = calibration.image_size
        dataset_size = image_shape[-2:]
        if calibrated_size != dataset_size:
            raise ValueError(
                f"Calibration image size {calibrated_size} differs from dataset size {dataset_size}."
            )
        report["camera_calibration"] = "compatible"

    if args.decode_frame:
        dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_dir)
        frame = dataset[0]
        report["decoded_image_shape"] = tuple(frame[args.image_key].shape)
        report["decoded_state_shape"] = tuple(frame["observation.state"].shape)
        report["decoded_action_shape"] = tuple(frame["action"].shape)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

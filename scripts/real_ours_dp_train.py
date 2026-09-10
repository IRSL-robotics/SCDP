#!/usr/bin/env python
"""Train calibrated real-world SCDP on a local LeRobot dataset."""

import argparse

from _real_common import run_training


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--image-key", default="observation.images.cam_3")
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--num-epochs", type=int, default=3001)
    parser.add_argument("--save-freq", type=int, default=200)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--crop-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="scdp-real-world")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())

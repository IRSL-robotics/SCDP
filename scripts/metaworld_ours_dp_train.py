#!/usr/bin/env python
"""Train SCDP (the ours_diffusion policy) on a local Meta-World dataset."""

import argparse

from _metaworld_common import run_training


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="assembly")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--repo-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-epochs", type=int, default=1001)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-every-eval", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="scdp-metaworld")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args(), policy_kind="scdp")

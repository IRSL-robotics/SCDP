#!/usr/bin/env python
"""Evaluate an SCDP checkpoint in Meta-World."""

import argparse
from pathlib import Path

from _metaworld_common import evaluate_policy, load_checkpoint, resolve_device, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-name", default="assembly")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-mode", choices=("fixed", "increment"), default="fixed")
    parser.add_argument("--compile-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-dir", type=Path)
    return parser.parse_args()


def main(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    policy, preprocessor, postprocessor, uses_raw_state = load_checkpoint(
        checkpoint=args.checkpoint,
        policy_kind="scdp",
        device=device,
        dataset_dir=args.dataset_dir,
    )
    evaluate_policy(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_name=args.task_name,
        num_episodes=args.episodes,
        seed=args.seed,
        device=device,
        uses_raw_state=uses_raw_state,
        seed_mode=args.seed_mode,
        compile_inference=args.compile_inference,
        compile_mode=args.compile_mode,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main(parse_args())

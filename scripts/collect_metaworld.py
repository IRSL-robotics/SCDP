#!/usr/bin/env python
"""Collect successful Meta-World expert demonstrations as a LeRobot dataset."""

import argparse
from pathlib import Path

import numpy as np
from _metaworld_common import (
    CAMERA_NAME,
    FPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    default_dataset_dir,
    local_repo_id,
    make_env,
    seed_everything,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="assembly")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser.parse_args()


def main(args):
    if args.episodes <= 0 or args.max_attempts <= 0:
        raise ValueError("`episodes` and `max_attempts` must be positive.")
    if args.max_attempts < args.episodes:
        raise ValueError("`max_attempts` cannot be smaller than `episodes`.")

    seed_everything(args.seed)
    dataset_dir = args.dataset_dir or default_dataset_dir(args.task_name)
    repo_id = args.repo_id or local_repo_id(args.task_name)
    if dataset_dir.exists():
        raise FileExistsError(
            f"{dataset_dir} already exists. Choose a new --dataset-dir; this script never overwrites data."
        )

    features = {
        "observation.images.world": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["delta_x", "delta_y", "delta_z", "gripper"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_dir,
        fps=FPS,
        robot_type="metaworld",
        use_videos=True,
        features=features,
        image_writer_threads=args.image_writer_threads,
    )
    env = make_env(args.task_name)
    saved_episodes = 0

    try:
        for attempt in range(args.max_attempts):
            if saved_episodes >= args.episodes:
                break

            observation, _ = env.reset(seed=args.seed + attempt)
            episode_success = False
            terminated = truncated = False

            while not (terminated or truncated):
                action = np.asarray(
                    env.expert_policy.get_action(observation["raw_obs"]),
                    dtype=np.float32,
                )
                dataset.add_frame(
                    {
                        "observation.images.world": np.ascontiguousarray(observation["pixels"][CAMERA_NAME]),
                        "observation.state": np.asarray(observation["agent_pos"], dtype=np.float32),
                        "action": action,
                        "task": env.task_description,
                    }
                )
                observation, _, terminated, truncated, info = env.step(action)
                episode_success = episode_success or bool(info.get("is_success", False))

            if episode_success:
                dataset.save_episode()
                saved_episodes += 1
                print(
                    f"[collect] saved={saved_episodes}/{args.episodes} "
                    f"attempt={attempt + 1} seed={args.seed + attempt}"
                )
            else:
                dataset.clear_episode_buffer(delete_images=True)
                print(f"[collect] rejected attempt={attempt + 1} seed={args.seed + attempt}")
        if saved_episodes < args.episodes:
            raise RuntimeError(
                f"Collected only {saved_episodes}/{args.episodes} successful episodes "
                f"within {args.max_attempts} attempts."
            )
    finally:
        dataset.finalize()
        env.close()

    print(f"[collect] dataset={dataset_dir} repo_id={repo_id}")


if __name__ == "__main__":
    main(parse_args())

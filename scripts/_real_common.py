#!/usr/bin/env python
"""Shared utilities for training and loading real-world DP/SCDP policies."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.ours_diffusion_real.configuration_diffusion import OursDiffusionRealConfig
from lerobot.policies.ours_diffusion_real.modeling_diffusion import OursDiffusionRealPolicy
from lerobot.processor import DeviceProcessorStep

DEFAULT_IMAGE_KEY = "observation.images.cam_3"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Pass --device cpu for a smoke run.")
    return resolved


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)


def load_camera_calibration(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        calibration = json.load(stream)

    required = {
        "camera_image_size",
        "camera_intrinsics",
        "camera_world_position",
        "camera_world_rotation",
    }
    missing = required - set(calibration)
    if missing:
        raise ValueError(f"Camera calibration is missing fields: {sorted(missing)}")
    return {name: calibration[name] for name in required}


def _action_bound(stats: dict[str, Any], name: str) -> tuple[float, ...]:
    values = torch.as_tensor(stats["action"][name], dtype=torch.float32)[:3]
    return tuple(float(value) for value in values.tolist())


def load_real_dataset(
    dataset_dir: Path,
    repo_id: str,
    image_key: str,
):
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot dataset not found at {dataset_dir} (missing meta/info.json).")

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_dir)
    features = dataset_to_policy_features(metadata.features)
    for key in ("observation.state", image_key, "action"):
        if key not in features:
            raise KeyError(f"Dataset does not contain required feature {key!r}.")

    input_features = {
        "observation.state": features["observation.state"],
        image_key: features[image_key],
    }
    output_features = {
        key: value for key, value in features.items() if value.type is FeatureType.ACTION
    }
    if set(output_features) != {"action"}:
        raise ValueError(f"Expected one `action` output feature, got {sorted(output_features)}.")
    return metadata, input_features, output_features


def run_training(args, policy_kind: str) -> None:
    for name in ("num_epochs", "save_freq", "log_freq", "batch_size", "num_inference_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` cannot be negative.")
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError("`max_updates` must be positive when provided.")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    repo_id = args.repo_id or "local/real-world"
    metadata, input_features, output_features = load_real_dataset(
        dataset_dir,
        repo_id,
        args.image_key,
    )

    common_config = {
        "input_features": input_features,
        "output_features": output_features,
        "device": str(device),
        "noise_scheduler_type": "DDIM",
        "num_inference_steps": args.num_inference_steps,
        "crop_shape": tuple(args.crop_shape) if args.crop_shape else None,
        "crop_is_random": args.random_crop,
        "down_dims": tuple(args.down_dims),
    }
    if policy_kind == "scdp":
        if args.camera_calibration is None:
            raise ValueError("Real-world SCDP requires --camera-calibration.")
        calibration = load_camera_calibration(Path(args.camera_calibration).expanduser().resolve())
        config = OursDiffusionRealConfig(
            **common_config,
            **calibration,
            use_separate_rgb_encoder_per_camera=False,
            sample_step=args.sample_step,
            action_min=_action_bound(metadata.stats, "min"),
            action_max=_action_bound(metadata.stats, "max"),
        )
        policy = OursDiffusionRealPolicy(config)
        uses_raw_state = True
    elif policy_kind == "dp":
        config = DiffusionConfig(
            **common_config,
            use_separate_rgb_encoder_per_camera=True,
        )
        policy = DiffusionPolicy(config)
        uses_raw_state = False
    else:
        raise ValueError(f"Unknown policy kind: {policy_kind}")

    policy.to(device)
    policy.train()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)
    delta_timestamps = {
        args.image_key: [index / metadata.fps for index in config.observation_delta_indices],
        "observation.state": [index / metadata.fps for index in config.observation_delta_indices],
        "action": [index / metadata.fps for index in config.action_delta_indices],
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_dir,
        delta_timestamps=delta_timestamps,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise RuntimeError(
            f"The dataset has fewer usable frames than batch size {args.batch_size}. "
            "Reduce --batch-size."
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs") / "real" / policy_kind
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"real-{policy_kind}-seed{args.seed}",
            dir=output_dir,
            config=vars(args),
        )

    updates = 0
    stopped_at_limit = False
    try:
        for epoch in range(args.num_epochs):
            progress = tqdm(dataloader, desc=f"epoch {epoch:04d}", unit="batch", leave=False)
            for batch in progress:
                raw_states = None
                if uses_raw_state:
                    raw_states = batch["observation.state"][:, :, :3].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch, raw_states) if uses_raw_state else policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                updates += 1
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if updates % args.log_freq == 0 or updates == 1:
                    record = {"epoch": epoch, "step": updates, "train/loss": loss.item()}
                    append_jsonl(metrics_path, record)
                    if wandb_run is not None:
                        wandb_run.log(record)
                if args.max_updates is not None and updates >= args.max_updates:
                    stopped_at_limit = True
                    break

            if epoch % args.save_freq == 0:
                save_checkpoint(
                    policy,
                    preprocessor,
                    postprocessor,
                    output_dir / "checkpoints" / f"epoch_{epoch:04d}",
                )
            if stopped_at_limit:
                break

        save_checkpoint(policy, preprocessor, postprocessor, output_dir / "checkpoints" / "final")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print(f"[train] completed; updates={updates}; outputs={output_dir}")


def load_real_checkpoint(
    checkpoint: Path,
    policy_kind: str,
    device: torch.device,
    dataset_dir: Path | None = None,
):
    if policy_kind == "scdp":
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, OursDiffusionRealConfig):
            raise TypeError(f"Expected a real SCDP checkpoint, found {config.type!r}.")
        if config.action_min is None or config.action_max is None:
            if dataset_dir is None:
                raise ValueError("This legacy SCDP checkpoint needs --dataset-dir for action bounds.")
            with (dataset_dir / "meta" / "stats.json").open(encoding="utf-8") as stream:
                stats = json.load(stream)
            config.action_min = tuple(stats["action"]["min"][:3])
            config.action_max = tuple(stats["action"]["max"][:3])
        policy_class = OursDiffusionRealPolicy
        uses_raw_state = True
    elif policy_kind == "dp":
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, DiffusionConfig):
            raise TypeError(f"Expected a Diffusion Policy checkpoint, found {config.type!r}.")
        policy_class = DiffusionPolicy
        uses_raw_state = False
    else:
        raise ValueError(f"Unknown policy kind: {policy_kind}")

    config.device = str(device)
    policy = policy_class.from_pretrained(checkpoint, config=config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
    )
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = str(device)
            step.__post_init__()
    return policy, preprocessor, postprocessor, uses_raw_state

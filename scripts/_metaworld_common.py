#!/usr/bin/env python
"""Shared Meta-World training and evaluation utilities."""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.metaworld import MetaworldEnv
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.ours_diffusion.configuration_diffusion import OursDiffusionConfig
from lerobot.policies.ours_diffusion.modeling_diffusion import OursDiffusionPolicy
from lerobot.processor import DeviceProcessorStep

CAMERA_NAME = "corner2"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
FPS = 80


def normalize_task_name(task_name: str) -> str:
    return task_name if task_name.endswith("-v3") else f"{task_name}-v3"


def short_task_name(task_name: str) -> str:
    return normalize_task_name(task_name).removesuffix("-v3")


def default_dataset_dir(task_name: str) -> Path:
    return Path("data") / "metaworld" / short_task_name(task_name) / "lerobot"


def default_output_dir(task_name: str, policy_kind: str, seed: int) -> Path:
    return Path("outputs") / short_task_name(task_name) / policy_kind / f"seed_{seed}"


def local_repo_id(task_name: str) -> str:
    return f"local/metaworld-{short_task_name(task_name)}"


def resolve_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Pass --device cpu for a smoke run.")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def make_env(task_name: str) -> MetaworldEnv:
    return MetaworldEnv(
        task=normalize_task_name(task_name),
        camera_names=(CAMERA_NAME,),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=IMAGE_WIDTH,
        observation_height=IMAGE_HEIGHT,
    )


def observation_tensors(
    observation: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32, device=device).unsqueeze(0)
    image_array = np.ascontiguousarray(observation["pixels"][CAMERA_NAME])
    image = torch.as_tensor(image_array, dtype=torch.float32, device=device)
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0
    policy_observation = {
        "observation.state": state,
        "observation.images.world": image,
    }
    return policy_observation, state[..., :3]


def stacked_state_history(history: deque[torch.Tensor], n_obs_steps: int) -> torch.Tensor:
    states = list(history)
    if not states:
        raise RuntimeError("State history is empty.")
    states = [states[0]] * (n_obs_steps - len(states)) + states
    return torch.stack(states, dim=1)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(policy, preprocessor, postprocessor, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)


def evaluate_policy(
    *,
    policy,
    preprocessor,
    postprocessor,
    task_name: str,
    num_episodes: int,
    seed: int,
    device: torch.device,
    uses_raw_state: bool,
    seed_mode: str = "increment",
    compile_inference: bool = False,
    compile_mode: str = "reduce-overhead",
    video_dir: Path | None = None,
) -> float:
    if num_episodes <= 0:
        raise ValueError("`num_episodes` must be positive.")
    if seed_mode not in {"fixed", "increment"}:
        raise ValueError(f"Unknown evaluation seed mode: {seed_mode!r}.")
    if compile_inference:
        if not uses_raw_state:
            raise ValueError("Compiled inference is only implemented for SCDP.")
        policy.optimize_for_inference(mode=compile_mode)

    env = make_env(task_name)
    num_successes = 0

    try:
        for episode in range(num_episodes):
            policy.reset()
            episode_seed = seed if seed_mode == "fixed" else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            state_history: deque[torch.Tensor] = deque(maxlen=policy.config.n_obs_steps)
            episode_success = False
            frames = [] if video_dir is not None else None
            if frames is not None:
                frames.append(np.ascontiguousarray(observation["pixels"][CAMERA_NAME]))

            terminated = truncated = False
            while not (terminated or truncated):
                policy_observation, raw_state = observation_tensors(observation, device)
                state_history.append(raw_state)
                policy_observation = preprocessor(policy_observation)

                with torch.inference_mode():
                    if uses_raw_state:
                        raw_states = stacked_state_history(state_history, policy.config.n_obs_steps)
                        action = policy.select_action(policy_observation, raw_states)
                    else:
                        action = policy.select_action(policy_observation)

                action = postprocessor(action)
                action_np = action.squeeze(0).detach().cpu().numpy()
                observation, _, terminated, truncated, info = env.step(action_np)
                episode_success = episode_success or bool(info.get("is_success", False))
                if frames is not None:
                    frames.append(np.ascontiguousarray(observation["pixels"][CAMERA_NAME]))

            num_successes += int(episode_success)
            print(
                f"[eval] episode={episode + 1}/{num_episodes} "
                f"seed={episode_seed} success={int(episode_success)}"
            )

            if frames is not None:
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"episode_{episode:03d}_success_{int(episode_success)}.mp4"
                imageio.mimsave(video_path, frames, fps=FPS)
    finally:
        env.close()

    success_rate = num_successes / num_episodes
    print(f"[eval] success_rate={success_rate:.4f} ({num_successes}/{num_episodes})")
    return success_rate


def _action_bound(stats: dict[str, Any], name: str) -> tuple[float, ...]:
    values = torch.as_tensor(stats["action"][name], dtype=torch.float32)[:3]
    return tuple(float(value) for value in values.tolist())


def _uses_legacy_shared_group_norm(checkpoint: Path) -> bool:
    weights_path = checkpoint / "model.safetensors"
    if not weights_path.is_file():
        return False
    key = "diffusion.rgb_encoder.blockB.res1.normalize.weight"
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        return key in weights.keys()


def _load_dataset(args):
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else default_dataset_dir(args.task_name)
    repo_id = args.repo_id or local_repo_id(args.task_name)
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found at {dataset_dir}. "
            "Run scripts/collect_metaworld.py or pass --dataset-dir."
        )

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_dir)
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
    input_features = {key: value for key, value in features.items() if key not in output_features}
    return dataset_dir, repo_id, metadata, input_features, output_features


def run_training(args, policy_kind: str) -> None:
    positive_args = (
        "num_epochs",
        "eval_freq",
        "log_freq",
        "batch_size",
        "num_inference_steps",
    )
    for name in positive_args:
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive.")
    if args.num_eval_episodes <= 0:
        raise ValueError("`num_eval_episodes` must be positive.")
    if args.scheduler_warmup_steps < 0:
        raise ValueError("scheduler_warmup_steps must be non-negative.")
    if args.optimizer_eps <= 0:
        raise ValueError("optimizer_eps must be positive.")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if args.grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be non-negative.")
    if any(beta < 0 or beta >= 1 for beta in args.optimizer_betas):
        raise ValueError("Each optimizer beta must be in [0, 1).")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset_dir, repo_id, metadata, input_features, output_features = _load_dataset(args)

    common_config = {
        "input_features": input_features,
        "output_features": output_features,
        "device": str(device),
        "noise_scheduler_type": "DDIM",
        "num_inference_steps": args.num_inference_steps,
        "crop_shape": None,
        "down_dims": tuple(args.down_dims),
        "optimizer_lr": args.learning_rate,
        "optimizer_betas": tuple(args.optimizer_betas),
        "optimizer_eps": args.optimizer_eps,
        "optimizer_weight_decay": args.weight_decay,
        "scheduler_name": args.lr_scheduler,
        "scheduler_warmup_steps": args.scheduler_warmup_steps,
    }

    if policy_kind == "scdp":
        config = OursDiffusionConfig(
            **common_config,
            use_separate_rgb_encoder_per_camera=False,
            sample_step=args.sample_step,
            action_min=_action_bound(metadata.stats, "min"),
            action_max=_action_bound(metadata.stats, "max"),
        )
        policy = OursDiffusionPolicy(config)
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
    if uses_raw_state and not getattr(args, "compile_inference", False):
        policy.diffusion.use_reference_inference()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)

    delta_timestamps = {
        "observation.images.world": [index / metadata.fps for index in config.observation_delta_indices],
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
        drop_last=True,
    )
    if len(dataloader) == 0:
        raise RuntimeError(
            f"The dataset has fewer usable frames than batch size {args.batch_size}. "
            "Reduce --batch-size or collect more episodes."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(args.task_name, policy_kind, args.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    optimizer_config = config.get_optimizer_preset()
    optimizer_config.grad_clip_norm = args.grad_clip_norm
    optimizer = optimizer_config.build(policy.parameters())
    scheduler = None
    if args.lr_scheduler != "none":
        scheduler = config.get_scheduler_preset().build(
            optimizer,
            num_training_steps=args.num_epochs * len(dataloader),
        )
    optimizer.zero_grad(set_to_none=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{short_task_name(args.task_name)}-{policy_kind}-seed{args.seed}",
            dir=output_dir,
            config=vars(args),
        )

    best_success_rate = -1.0
    stop_early = False

    def evaluate_checkpoint(epoch: int, step: int) -> bool:
        nonlocal best_success_rate

        policy.eval()
        video_dir = output_dir / "videos" / f"epoch_{epoch:04d}" if args.save_video else None
        success_rate = evaluate_policy(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_name=args.task_name,
            num_episodes=args.num_eval_episodes,
            seed=args.seed if args.eval_seed is None else args.eval_seed,
            device=device,
            uses_raw_state=uses_raw_state,
            seed_mode=args.eval_seed_mode,
            compile_inference=uses_raw_state and getattr(args, "compile_inference", False),
            compile_mode=getattr(args, "compile_mode", "reduce-overhead"),
            video_dir=video_dir,
        )
        eval_record = {
            "epoch": epoch,
            "step": step,
            "eval/success_rate": success_rate,
        }
        append_jsonl(metrics_path, eval_record)
        if wandb_run is not None:
            wandb_run.log(eval_record)

        if args.save_every_eval:
            save_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                output_dir / "checkpoints" / f"epoch_{epoch:04d}",
            )
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            save_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                output_dir / "checkpoints" / f"best_epoch_{epoch:04d}",
            )

        if success_rate >= args.early_stop_success_rate:
            print(f"[train] reached early-stop success rate at epoch {epoch}")
            return True
        policy.train()
        return False

    try:
        for epoch in range(args.num_epochs):
            progress = tqdm(dataloader, desc=f"epoch {epoch:04d}", unit="batch", leave=False)
            for batch_index, batch in enumerate(progress):
                raw_states = None
                if uses_raw_state:
                    raw_states = batch["observation.state"][:, :, :3].to(
                        device=device, dtype=torch.float32, non_blocking=True
                    )

                batch = preprocessor(batch)
                if uses_raw_state:
                    loss, _ = policy.forward(batch, raw_states)
                else:
                    loss, _ = policy.forward(batch)

                loss.backward()
                grad_norm = None
                if optimizer_config.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), optimizer_config.grad_clip_norm, error_if_nonfinite=False
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step = epoch * len(dataloader) + batch_index
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if global_step % args.log_freq == 0:
                    record = {
                        "epoch": epoch,
                        "step": global_step,
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm.item() if grad_norm is not None else None,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    append_jsonl(metrics_path, record)
                    if wandb_run is not None:
                        wandb_run.log(record)

                if epoch % args.eval_freq == 0 and batch_index == 0:
                    stop_early = evaluate_checkpoint(epoch, global_step + 1)
                    if stop_early:
                        break

            if stop_early:
                break

        save_checkpoint(policy, preprocessor, postprocessor, output_dir / "checkpoints" / "final")
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    status = "early-stopped" if stop_early else "completed"
    print(f"[train] {status}; outputs={output_dir}")


def load_checkpoint(
    *,
    checkpoint: Path,
    policy_kind: str,
    device: torch.device,
    dataset_dir: Path | None = None,
):
    if policy_kind == "scdp":
        config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, OursDiffusionConfig):
            raise TypeError(f"Expected an SCDP checkpoint, found {config.type!r}.")
        if _uses_legacy_shared_group_norm(checkpoint):
            config.use_shared_group_norm_in_residual_blocks = True
        if config.action_min is None or config.action_max is None:
            if dataset_dir is None:
                raise ValueError(
                    "This legacy SCDP checkpoint needs --dataset-dir so action bounds can be loaded."
                )
            config.dataset_stats_path = str(dataset_dir / "meta" / "stats.json")
        policy_class = OursDiffusionPolicy
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

    # A checkpoint records its training device. Respect an explicit evaluation override.
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = str(device)
            step.__post_init__()

    return policy, preprocessor, postprocessor, uses_raw_state

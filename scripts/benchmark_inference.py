#!/usr/bin/env python
"""Benchmark legacy and optimized SCDP action-chunk inference on identical weights and inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import einops
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.ours_diffusion.configuration_diffusion import OursDiffusionConfig
from lerobot.policies.ours_diffusion.modeling_diffusion import OursDiffusionPolicy
from lerobot.policies.ours_diffusion_real.configuration_diffusion import OursDiffusionRealConfig
from lerobot.policies.ours_diffusion_real.modeling_diffusion import OursDiffusionRealPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


def legacy_projection(model, raw_states, target):
    device = raw_states.device
    if target == "metaworld":
        camera_position = model.metaworld_cam_world_pos.to(device=device)
        camera_rotation = model.metaworld_cam_world_rot.to(device=device)
        fovy = np.deg2rad(60.0)
        height = width = 224
        fy = (height / 2) / np.tan(fovy / 2)
        fx = fy * (width / height)
        cx, cy = width / 2, height / 2
        camera_states = (raw_states - camera_position) @ camera_rotation
        u = (fx * (camera_states[:, :, 0] / camera_states[:, :, 2]) + cx).unsqueeze(-1)
        v = (fy * (camera_states[:, :, 1] / -camera_states[:, :, 2]) + cy).unsqueeze(-1)
    else:
        camera_position = model.camera_world_position.to(device=device)
        camera_rotation = model.camera_world_rotation.to(device=device)
        fx, fy, cx, cy = model.projection_intrinsics.to(device=device)
        height, width = model.projection_image_size
        camera_states = (raw_states - camera_position) @ camera_rotation
        u = (fx * (camera_states[:, :, 0] / camera_states[:, :, 2]) + cx).unsqueeze(-1)
        v = (fy * (camera_states[:, :, 1] / camera_states[:, :, 2]) + cy).unsqueeze(-1)

    projected = torch.cat([u, v], dim=-1)
    projected[:, :, 0] = 2 * (projected[:, :, 0] / width) - 1
    projected[:, :, 1] = 2 * (projected[:, :, 1] / height) - 1
    return projected


def legacy_movement(model, current_states, delta_states, target):
    if target == "metaworld":
        low = model.metaworld_mocap_low.to(device=delta_states.device)
        high = model.metaworld_mocap_high.to(device=delta_states.device)
        delta_states = torch.clamp(delta_states, min=-1, max=1) * (1.0 / 100)
        return torch.clamp(current_states + delta_states, min=low, max=high)
    return current_states + torch.clamp(delta_states, min=-1, max=1)


def legacy_action_unnormalizer(model, actions):
    action_min = model.action_min.to(device=actions.device)
    action_max = model.action_max.to(device=actions.device)
    actions = (actions + 1.0) / 2.0
    return action_min + actions * (action_max - action_min)


@torch.no_grad()
def legacy_conditional_sample(model, batch, raw_states, noise, target):
    """Pre-optimization scheduling, allocation, projection, and movement loop."""
    device = get_device_from_parameters(model)
    dtype = get_dtype_from_parameters(model)
    batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
    state_cond = model._prepare_state_conditioning(batch)
    image_feature_maps = model._prepare_image_feature_maps(batch)
    sample = noise if noise is not None else torch.randn(
        (batch_size, model.config.horizon, model.config.action_feature.shape[0]),
        dtype=dtype,
        device=device,
    )

    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    for timestep in model.noise_scheduler.timesteps:
        dynamic_states = raw_states
        projected = legacy_projection(model, raw_states, target)
        states = projected.unsqueeze(2)
        for index in range(1, 1 + model.config.sample_step):
            delta = legacy_action_unnormalizer(model, sample[:, index, :3]).unsqueeze(1).repeat(1, 2, 1)
            dynamic_states = legacy_movement(model, dynamic_states, delta, target)
            projected = legacy_projection(model, dynamic_states, target)
            states = torch.cat([states, projected.unsqueeze(2)], dim=2)

        dynamic_cond = model._dynamic_conditioning(image_feature_maps, states, state_cond)
        dynamic_cond = einops.rearrange(
            dynamic_cond,
            "(b s) ... -> b (s ...)",
            b=batch_size,
            s=n_obs_steps,
        )
        model_output = model.unet(
            sample,
            torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
            global_cond=dynamic_cond,
        )
        sample = model.noise_scheduler.step(model_output, timestep, sample).prev_sample
    return sample


def percentile_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "min_ms": float(values.min()),
    }


def cuda_time(callable_) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def paired_cuda_times(reference, optimized, warmup: int, iterations: int) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        reference()
        optimized()
    torch.cuda.synchronize()

    reference_samples = []
    optimized_samples = []
    for index in range(iterations):
        # Alternate order to reduce clock/temperature drift bias.
        if index % 2 == 0:
            reference_samples.append(cuda_time(reference))
            optimized_samples.append(cuda_time(optimized))
        else:
            optimized_samples.append(cuda_time(optimized))
            reference_samples.append(cuda_time(reference))
    return reference_samples, optimized_samples


def make_case(target: str, args):
    height, width = args.image_size
    if target == "metaworld":
        state_dim, action_dim = 4, 4
        input_features = {
            "observation.images.world": PolicyFeature(FeatureType.VISUAL, (3, height, width)),
            "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
        }
        config = OursDiffusionConfig(
            input_features=input_features,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
            device="cuda",
            crop_shape=None,
            down_dims=tuple(args.down_dims),
            noise_scheduler_type="DDIM",
            num_inference_steps=args.num_inference_steps,
            sample_step=args.sample_step,
            action_min=(-0.01, -0.01, -0.01),
            action_max=(0.01, 0.01, 0.01),
        )
        policy = OursDiffusionPolicy(config).cuda().eval()
        raw_states = torch.tensor([[[0.0, 0.6, 0.2]] * config.n_obs_steps], device="cuda")
        image_key = "observation.images.world"
    else:
        state_dim, action_dim = 8, 7
        image_key = "observation.images.cam_3"
        input_features = {
            image_key: PolicyFeature(FeatureType.VISUAL, (3, height, width)),
            "observation.state": PolicyFeature(FeatureType.STATE, (state_dim,)),
        }
        config = OursDiffusionRealConfig(
            input_features=input_features,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
            device="cuda",
            crop_shape=None,
            down_dims=tuple(args.down_dims),
            noise_scheduler_type="DDIM",
            num_inference_steps=args.num_inference_steps,
            sample_step=args.sample_step,
            action_min=(-0.05, -0.05, -0.05),
            action_max=(0.05, 0.05, 0.05),
            camera_image_size=(height, width),
            camera_intrinsics=((width, 0.0, width / 2), (0.0, height, height / 2), (0.0, 0.0, 1.0)),
            camera_world_position=(0.0, 0.0, 0.0),
            camera_world_rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        policy = OursDiffusionRealPolicy(config).cuda().eval()
        raw_states = torch.tensor([[[0.1, 0.1, 1.0]] * config.n_obs_steps], device="cuda")

    batch = {
        OBS_STATE: torch.rand(
            args.batch_size,
            config.n_obs_steps,
            state_dim,
            device="cuda",
        ),
        OBS_IMAGES: torch.rand(
            args.batch_size,
            config.n_obs_steps,
            1,
            3,
            height,
            width,
            device="cuda",
        ),
    }
    raw_states = raw_states.expand(args.batch_size, -1, -1).contiguous()
    noise = torch.randn(
        args.batch_size,
        config.horizon,
        action_dim,
        device="cuda",
    )
    return policy.diffusion, batch, raw_states, noise, image_key


def run_case(target: str, args) -> dict:
    torch.manual_seed(args.seed)
    model, batch, raw_states, noise, _ = make_case(target, args)

    def optimized():
        return model.conditional_sample(batch, raw_states=raw_states, noise=noise)

    def reference():
        return legacy_conditional_sample(model, batch, raw_states, noise, target)

    reference_output = reference()
    optimized_output = optimized()
    torch.cuda.synchronize()
    max_absolute_error = (reference_output - optimized_output).abs().max().item()

    reference_samples, optimized_samples = paired_cuda_times(
        reference, optimized, args.warmup, args.iterations
    )
    reference_stats = percentile_summary(reference_samples)
    optimized_stats = percentile_summary(optimized_samples)
    speedup = reference_stats["p50_ms"] / optimized_stats["p50_ms"]
    return {
        "target": target,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "observation_steps": model.config.n_obs_steps,
        "horizon": model.config.horizon,
        "action_steps": model.config.n_action_steps,
        "sample_step": model.config.sample_step,
        "denoising_steps": model.num_inference_steps,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "max_absolute_error": max_absolute_error,
        "legacy": reference_stats,
        "optimized": optimized_stats,
        "p50_speedup": speedup,
        "p50_latency_reduction_percent": (1 - 1 / speedup) * 100,
        "optimized_amortized_ms_per_action": optimized_stats["p50_ms"] / model.config.n_action_steps,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("metaworld", "real", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--down-dims", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA for synchronized GPU timings.")
    for name in ("batch_size", "num_inference_steps", "sample_step", "warmup", "iterations"):
        if getattr(args, name) <= 0:
            raise ValueError(f"`{name}` must be positive.")

    targets = ("metaworld", "real") if args.target == "both" else (args.target,)
    results = [run_case(target, args) for target in targets]
    report = {"results": results}
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

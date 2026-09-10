#!/usr/bin/env python
"""Run a real-world DP/SCDP checkpoint on RGB image and robot-state observations.

This script only predicts actions. It intentionally contains no robot driver or actuator command.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from _real_common import load_real_checkpoint, resolve_device
from PIL import Image


class RealPolicyRunner:
    """Stateful action-chunk runner suitable for wrapping in a robot control process."""

    def __init__(
        self,
        checkpoint: Path,
        policy_kind: str,
        device: str,
        dataset_dir: Path | None = None,
        compile_inference: bool = True,
        compile_mode: str = "reduce-overhead",
    ):
        self.device = resolve_device(device)
        self.policy, self.preprocessor, self.postprocessor, self.uses_raw_state = load_real_checkpoint(
            checkpoint=checkpoint,
            policy_kind=policy_kind,
            device=self.device,
            dataset_dir=dataset_dir,
        )
        if compile_inference and self.uses_raw_state:
            self.policy.optimize_for_inference(mode=compile_mode)
        self.image_key, self.image_feature = next(iter(self.policy.config.image_features.items()))
        self.state_feature = self.policy.config.robot_state_feature
        self.raw_state_history: deque[torch.Tensor] = deque(maxlen=self.policy.config.n_obs_steps)

    def reset(self) -> None:
        self.policy.reset()
        self.raw_state_history.clear()

    def _image_tensor(self, image: str | Path | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                array = np.array(source.convert("RGB"), copy=True)
            tensor = torch.from_numpy(array)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(np.array(image, copy=True))
        else:
            tensor = image.detach().cpu()

        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3-D RGB image, got shape {tuple(tensor.shape)}.")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected RGB channels, got shape {tuple(tensor.shape)}.")
        expected = tuple(self.image_feature.shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Image shape {tuple(tensor.shape)} does not match checkpoint shape {expected}.")
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max().item() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def step(self, image: str | Path | np.ndarray | torch.Tensor, state) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        expected_state_shape = tuple(self.state_feature.shape)
        if tuple(state_tensor.shape) != expected_state_shape:
            raise ValueError(
                f"State shape {tuple(state_tensor.shape)} does not match checkpoint shape {expected_state_shape}."
            )

        observation = {
            self.image_key: self._image_tensor(image),
            "observation.state": state_tensor,
        }
        raw_state = state_tensor[:3].to(self.device).unsqueeze(0)
        self.raw_state_history.append(raw_state)
        observation = self.preprocessor(observation)

        with torch.inference_mode():
            if self.uses_raw_state:
                history = list(self.raw_state_history)
                history = [history[0]] * (self.policy.config.n_obs_steps - len(history)) + history
                raw_states = torch.stack(history, dim=1)
                action = self.policy.select_action(observation, raw_states)
            else:
                action = self.policy.select_action(observation)
            return self.postprocessor(action).squeeze(0).detach().cpu()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("dp", "scdp"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--dataset-dir", type=Path, help="Only needed for a legacy SCDP checkpoint.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="Single RGB image for a one-step prediction.")
    inputs.add_argument(
        "--input-jsonl",
        type=Path,
        help="JSONL stream with `image`, `state`, and optional `reset`.",
    )
    parser.add_argument("--state", type=float, nargs="+", help="Robot state paired with --image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is not None and args.state is None:
        raise ValueError("--state is required with --image.")
    if args.input_jsonl is not None and args.state is not None:
        raise ValueError("--state can only be used with --image.")

    runner = RealPolicyRunner(
        checkpoint=args.checkpoint.expanduser().resolve(),
        policy_kind=args.policy,
        device=args.device,
        dataset_dir=args.dataset_dir.expanduser().resolve() if args.dataset_dir else None,
        compile_inference=args.compile_inference,
        compile_mode=args.compile_mode,
    )
    if args.image is not None:
        action = runner.step(args.image.expanduser().resolve(), args.state)
        print(json.dumps({"action": action.tolist()}))
        return

    input_path = args.input_jsonl.expanduser().resolve()
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("reset", False):
                runner.reset()
            image_path = Path(record["image"]).expanduser()
            if not image_path.is_absolute():
                image_path = input_path.parent / image_path
            action = runner.step(image_path, record["state"])
            print(json.dumps({"line": line_number, "action": action.tolist()}), flush=True)


if __name__ == "__main__":
    main()

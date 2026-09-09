# SCDP: Meta-World and Real-World Implementation Guide

This repository provides a focused implementation of the SCDP experiments from
`IRSL-robotics/policy_learning`. It includes SCDP (`ours_diffusion`), the standard
Diffusion Policy baseline, Meta-World collection/training/evaluation, and calibrated
real-world training/inference utilities. Datasets, checkpoints, notebooks, videos,
robot drivers, and unrelated workspace artifacts are intentionally excluded.

The code is structured as a small fork of
[LeRobot](https://github.com/huggingface/lerobot) 0.4.1 so the experiments do not
silently change when upstream LeRobot changes.

## Included experiments

| Environment | Policy | Train | Evaluate / infer |
| --- | --- | --- | --- |
| Meta-World | SCDP | `scripts/metaworld_ours_dp_train.py` | `scripts/metaworld_ours_dp_eval.py` |
| Meta-World | Diffusion Policy | `scripts/metaworld_dp_train.py` | `scripts/metaworld_dp_eval.py` |
| Real world | calibrated SCDP | `scripts/real_ours_dp_train.py` | `scripts/real_policy_inference.py --policy scdp` |
| Real world | Diffusion Policy | `scripts/real_dp_train.py` | `scripts/real_policy_inference.py --policy dp` |

Meta-World uses one `224 x 224` RGB observation from the `corner2` camera, a 4-D
agent state, and a 4-D action. The recorded real-world setup uses one `cam_3` RGB
observation, an 8-D robot state, and a 7-D action. SCDP projects the first three state
dimensions into the image and samples multi-scale image features along the predicted
3-D state trajectory.

## Installation

The recorded development environment is Linux, Python 3.10.19, Meta-World 3.0.0,
MuJoCo 3.3.7, PyTorch 2.9.1, torchvision 0.24.1, and CUDA 12.8. Package versions are
pinned in `pyproject.toml` and `uv.lock`.

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone <YOUR_REPOSITORY_URL> SCDP
cd SCDP
uv sync --locked
```

For a headless GPU machine, select EGL before collecting or evaluating Meta-World:

```bash
export MUJOCO_GL=egl
```

A CUDA GPU is expected for the published training defaults. Training on CPU is only
useful for small smoke tests.

## Meta-World workflow

### 1. Collect expert demonstrations

The collector writes successful expert rollouts directly in LeRobot v3 format; no
intermediate HDF5 or NPZ conversion is needed.

```bash
uv run python scripts/collect_metaworld.py \
  --task-name assembly \
  --episodes 20 \
  --seed 0
```

The default output is `data/metaworld/assembly/lerobot`. Collection refuses to
overwrite an existing directory.

### 2. Train both policies

```bash
uv run python scripts/metaworld_dp_train.py \
  --task-name assembly \
  --seed 0

uv run python scripts/metaworld_ours_dp_train.py \
  --task-name assembly \
  --seed 0 \
  --sample-step 8
```

Weights, processor state, configs, and JSONL metrics are written under
`outputs/assembly/{dp,scdp}/seed_0`. A portable best checkpoint is saved whenever
success rate improves, and `checkpoints/final` is always saved. Add
`--save-every-eval`, `--save-video`, or `--wandb` as needed.

### 3. Evaluate checkpoints

```bash
uv run python scripts/metaworld_dp_eval.py \
  --task-name assembly \
  --checkpoint outputs/assembly/dp/seed_0/checkpoints/final \
  --episodes 20 \
  --seed 0

uv run python scripts/metaworld_ours_dp_eval.py \
  --task-name assembly \
  --checkpoint outputs/assembly/scdp/seed_0/checkpoints/final \
  --episodes 20 \
  --seed 0
```

Pass `--video-dir outputs/eval-videos` to save rollout videos.

### Meta-World defaults

| Setting | Value |
| --- | --- |
| Demonstrations | 20 successful expert rollouts |
| Image / camera | `224 x 224`, `corner2` |
| Dataset FPS | 80 |
| Observation steps / horizon / action steps | `2 / 16 / 8` |
| Batch size | 128 |
| Optimizer | Adam, learning rate `1e-4` |
| U-Net dimensions | `128 256 384` |
| Scheduler / inference steps | DDIM / 16 |
| SCDP sample step | 8 |
| Epochs / evaluation interval | 1001 / 200 |
| Evaluation episodes | 20 |

## Real-world workflow

The real-world scripts expect an existing local LeRobot v3 dataset. They do not
contain a robot-specific camera or actuator driver. This keeps the training setup
independent of hardware-specific runtime code and makes the inference boundary explicit.

### Dataset contract

The defaults expect these features:

| Feature | Recorded shape | Meaning |
| --- | --- | --- |
| `observation.images.cam_3` | `(3, H, W)` | fixed RGB camera, float `[0,1]` after decode |
| `observation.state` | `(8,)` | first 3 values are end-effector XYZ in the calibrated world/base frame |
| `action` | `(7,)` | first 3 values are XYZ deltas in the same units/frame |

The state and action dimensions are read from metadata, but SCDP requires at least
three positional dimensions and currently supports exactly one calibrated camera.
Use another image key with `--image-key`.

Validate metadata and decode a real frame before training:

```bash
uv run python scripts/validate_real_dataset.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --decode-frame
```

### Camera calibration

Real SCDP requires a JSON calibration file; the calibration is embedded in every new
checkpoint. `configs/real_camera.example.json` records the original experiment's
fixed camera after resizing `640 x 480` input to `320 x 240`. Treat it as an example,
not as a universal camera calibration.

The required fields are:

```json
{
  "camera_image_size": [240, 320],
  "camera_intrinsics": [[304.325225, 0.0, 163.35962], [0.0, 303.736235, 124.355475], [0.0, 0.0, 1.0]],
  "camera_world_position": [1.322, -0.013, 0.609],
  "camera_world_rotation": [[-0.00662099, 0.47824121, -0.87820357], [0.99996514, 0.00763364, -0.00338195], [0.0050865, -0.87819535, -0.47827509]]
}
```

Intrinsics must be in pixels of the dataset image, not the camera's pre-resize image.
When resizing, scale `fx, cx` by the horizontal ratio and `fy, cy` by the vertical
ratio. When center-cropping before dataset creation, subtract the crop's left/top
offset from `cx/cy`. The validator rejects a calibration whose image size differs
from dataset metadata.

SCDP training only permits no crop or a deterministic center crop. Random cropping
would move pixels without applying the same unknown offset to the projected state and
is therefore rejected. For a center crop passed with `--crop-shape H W`, the policy
updates the principal point internally.

The rotation convention is preserved from the original code:

```text
point_camera = (point_world - camera_world_position) @ camera_world_rotation
```

Verify projected end-effector pixels independently before commanding a robot.

### Train

```bash
uv run python scripts/real_dp_train.py \
  --dataset-dir /path/to/task/lerobot \
  --output-dir outputs/real/dp

uv run python scripts/real_ours_dp_train.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --output-dir outputs/real/scdp \
  --sample-step 8
```

Original real-world defaults are batch size 128, Adam at `1e-4`, U-Net dimensions
`128 256 384`, 3001 epochs, and checkpoint interval 200. Baseline DP defaults to 10
DDIM steps; SCDP defaults to 16. All are CLI options. `--max-updates 1 --batch-size 1
--num-workers 0 --down-dims 32 64 128` is useful for a GPU smoke test.

Real-world training has no automatic success evaluation because no task/robot driver
is included. It records `metrics.jsonl`, periodic checkpoints, and a final checkpoint.

### Checkpoint inference

One observation:

```bash
uv run python scripts/real_policy_inference.py \
  --policy scdp \
  --checkpoint outputs/real/scdp/checkpoints/final \
  --image frame.png \
  --state 0.49 0.00 0.15 1.00 0.00 0.00 0.00 0.00
```

For a persistent sequence, pass `--input-jsonl observations.jsonl`. Each non-empty
line must contain an image path and state; `reset` clears history/action queues at a
new episode:

```json
{"image":"frames/000000.png","state":[0.49,0.00,0.15,1.00,0.00,0.00,0.00,0.00],"reset":true}
{"image":"frames/000001.png","state":[0.49,0.01,0.15,1.00,0.00,0.00,0.00,0.00]}
```

`RealPolicyRunner` in the same script can be imported into a long-running robot
process and accepts a path, NumPy RGB array, or PyTorch tensor. The policy generates
eight actions per diffusion call and returns cached actions on subsequent control
steps. Call `runner.reset()` at every episode boundary.

The script outputs dataset-unit actions only. It deliberately does not send robot
commands. A deployment integration must independently enforce workspace bounds,
velocity/acceleration limits, collision checks, watchdog timeouts, gripper limits,
and an emergency stop.

## Inference optimization and measured speed

The SCDP inference loop is optimized and the real-world policy inherits the same
implementation. It caches scheduler/model timesteps, keeps geometry and action bounds
as device-following buffers, uses `expand` instead of materializing `repeat`, builds
sampled trajectories with one `stack` instead of repeated `cat`, optimizes Meta-World
projection arithmetic, and avoids redundant device transfers. These changes
do not change model weights or outputs.

Measured on 2026-09-09 with an NVIDIA GeForce RTX 5090, PyTorch 2.9.1, batch 1,
`224 x 224` RGB, two observations, horizon 16, sample step 8, and 16 DDIM steps:

| Target | Legacy p50 / p95 | Optimized p50 / p95 | p50 speedup | Output max abs error |
| --- | ---: | ---: | ---: | ---: |
| Meta-World SCDP | 68.770 / 71.955 ms | 65.585 / 68.076 ms | 1.049x (4.63%) | 0.0 |
| Real SCDP | 66.127 / 69.139 ms | 64.182 / 67.548 ms | 1.030x (2.94%) | 0.0 |

The benchmark uses 10 warm-up calls and 50 paired, alternating-order measurements
with synchronized CUDA events. It measures a complete fresh eight-action chunk,
including vision encoding and denoising, but excludes file I/O, preprocessing,
postprocessing, and robot communication.

```bash
uv run python scripts/benchmark_inference.py \
  --target both \
  --warmup 10 \
  --iterations 50
```

The optimization is real but modest: vision and U-Net computation dominate total
latency, so the measured gain is about 3-5%, not a multi-fold speedup. The optimized
fresh-chunk p50 is about 64-66 ms, amortized over eight returned actions.
Using 10 instead of 16 DDIM steps is a larger latency
lever and is exposed by every train config, but it changes inference behavior and
must be evaluated for task success. `torch.compile`, mixed precision, TensorRT, and
CUDA Graphs are intentionally not enabled by default because they add shape/hardware
constraints and require separate numerical and task-success validation.

## Existing datasets and legacy checkpoints

Datasets can live outside this repository:

```bash
uv run python scripts/metaworld_ours_dp_train.py \
  --task-name assembly \
  --dataset-dir /mnt/SSD/metaworld/assembly/lerobot \
  --repo-id local/metaworld-assembly
```

New SCDP checkpoints embed positional action bounds. New real SCDP checkpoints also
embed camera calibration, so both can move without the training dataset. For a legacy
checkpoint containing only `dataset_stats_path`, pass `--dataset-dir` to the relevant
evaluation/inference script. The Meta-World loader also detects the shared-GroupNorm
encoder keys used by early experiment checkpoints and reconstructs that exact encoder
layout before loading weights.

## Implementation notes

- Task names may be supplied with or without the Meta-World `-v3` suffix.
- Training, collection, and evaluation seeds are explicit. CUDA kernels can still
  vary slightly across GPU and driver versions; equal seeds do not guarantee
  bitwise-identical weights.
- Dataset/checkpoint contents are excluded by `.gitignore`.
- The Meta-World projection assumes the original `corner2` pose, 60-degree field of
  view, and mocap bounds. Changing that camera requires changing the policy geometry.
- Meta-World success is read from `info["is_success"]`; time-limit termination alone
  is not success.
- Real-world success must be evaluated in the downstream robot/task integration.

## Validation

```bash
uv sync --locked --extra dev
uv run ruff check scripts tests src/lerobot/policies
uv run pytest
uv build
```

Tests cover CPU-safe policy construction, camera projection, checkpoint-contained
statistics/calibration, config registry round-trips, and baseline normalization. Full
training/evaluation and the speed benchmark require suitable data and a CUDA GPU.

## Repository layout

```text
.
├── configs/
│   └── real_camera.example.json
├── scripts/
│   ├── collect_metaworld.py
│   ├── metaworld_{dp,ours_dp}_{train,eval}.py
│   ├── real_{dp,ours_dp}_train.py
│   ├── real_policy_inference.py
│   ├── validate_real_dataset.py
│   └── benchmark_inference.py
├── src/lerobot/
│   ├── envs/metaworld.py
│   └── policies/{diffusion,ours_diffusion,ours_diffusion_real}/
├── tests/
├── LICENSE
├── NOTICE
├── pyproject.toml
└── uv.lock
```

## Provenance and license

The fork baseline is LeRobot 0.4.1. Experiment code was extracted from
`IRSL-robotics/policy_learning` at commit `1edfb23`, together with the relevant
uncommitted experiment changes present on 2026-09-09. Cleanup removes
machine-specific paths, unrelated `policy_learning` imports, and import-time CUDA
allocations while preserving experiment architecture and defaults.

See `NOTICE` for attribution and `LICENSE` for the Apache License 2.0.
# SCDP

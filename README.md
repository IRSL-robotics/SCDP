# SCDP

Official implementation of **SCDP**, accepted at **CoRL 2026**.

This repository is a focused fork of [LeRobot](https://github.com/huggingface/lerobot)
0.4.1. It provides SCDP and Diffusion Policy baselines for Meta-World, together with
real-world training and inference utilities.

## Setup

Linux, Python 3.10, and a CUDA GPU are recommended. Install
[uv](https://docs.astral.sh/uv/), then run:

```bash
git clone <YOUR_REPOSITORY_URL> SCDP
cd SCDP
uv sync --locked
```

For headless Meta-World runs:

```bash
export MUJOCO_GL=egl
```

## Entry points

| Environment | Policy | Train | Evaluate / infer |
| --- | --- | --- | --- |
| Meta-World | SCDP | `scripts/metaworld_ours_dp_train.py` | `scripts/metaworld_ours_dp_eval.py` |
| Meta-World | Diffusion Policy | `scripts/metaworld_dp_train.py` | `scripts/metaworld_dp_eval.py` |
| Real world | SCDP | `scripts/real_ours_dp_train.py` | `scripts/real_policy_inference.py --policy scdp` |
| Real world | Diffusion Policy | `scripts/real_dp_train.py` | `scripts/real_policy_inference.py --policy dp` |

## Meta-World

Collect successful expert demonstrations in LeRobot v3 format:

```bash
uv run python scripts/collect_metaworld.py \
  --task-name assembly \
  --episodes 20 \
  --seed 0
```

Train the baseline and SCDP:

```bash
uv run python scripts/metaworld_dp_train.py \
  --task-name assembly \
  --seed 0

uv run python scripts/metaworld_ours_dp_train.py \
  --task-name assembly \
  --seed 0 \
  --sample-step 8
```

Evaluate the final checkpoints:

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

Use `--video-dir outputs/eval-videos` to save rollouts. By default, training writes
checkpoints and metrics under `outputs/<task>/{dp,scdp}/seed_<seed>`.

### Default configuration

| Setting | Value |
| --- | --- |
| Demonstrations | 20 successful episodes |
| Image / camera | `224 x 224`, `corner2` |
| Observation steps / horizon / action steps | `2 / 16 / 8` |
| Batch size | 128 |
| Optimizer | Adam, `1e-4` |
| Scheduler / inference steps | DDIM / 16 |
| SCDP sample step | 8 |
| Epochs / evaluation interval | 1001 / 200 |

## Real world

The real-world scripts use a local LeRobot v3 dataset. Robot drivers are intentionally
not included.

| Feature | Expected content |
| --- | --- |
| `observation.images.cam_3` | RGB image from one fixed camera |
| `observation.state` | robot state; first 3 values are end-effector XYZ |
| `action` | action vector; first 3 values are XYZ deltas |

SCDP requires at least three state/action dimensions and a calibrated fixed camera.
Validate the dataset and calibration before training:

```bash
uv run python scripts/validate_real_dataset.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --decode-frame
```

`configs/real_camera.example.json` documents the required intrinsics, camera pose, and
image size. Replace its values with your calibration. Intrinsics must match the stored
image resolution; random crop is unsupported because it invalidates the projection.

Train both policies:

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

Run checkpoint inference on one observation:

```bash
uv run python scripts/real_policy_inference.py \
  --policy scdp \
  --checkpoint outputs/real/scdp/checkpoints/final \
  --image frame.png \
  --state 0.49 0.00 0.15 1.00 0.00 0.00 0.00 0.00
```

Use `--input-jsonl observations.jsonl` for a persistent sequence, or import
`RealPolicyRunner` into the robot process. Call `runner.reset()` at each episode
boundary. The runner returns dataset-unit actions and does not command hardware;
workspace limits, collision checks, watchdogs, and emergency-stop handling remain the
responsibility of the robot integration.

## Inference speed

Inference caches diffusion timesteps, keeps geometry as device buffers, and avoids
redundant allocations and transfers. The real-world policy uses the same optimized
path.

Measured on an NVIDIA GeForce RTX 5090 with PyTorch 2.9.1, batch size 1, 16 DDIM
steps, and 10 warm-up / 50 timed calls:

| Target | Before p50 / p95 | Optimized p50 / p95 | p50 speedup | Max error |
| --- | ---: | ---: | ---: | ---: |
| Meta-World SCDP | 68.770 / 71.955 ms | 65.585 / 68.076 ms | 1.049x | 0.0 |
| Real SCDP | 66.127 / 69.139 ms | 64.182 / 67.548 ms | 1.030x | 0.0 |

```bash
uv run python scripts/benchmark_inference.py \
  --target both \
  --warmup 10 \
  --iterations 50
```

The measured gain is about 3–5%. Each diffusion call produces eight cached actions,
so later control steps reuse the generated chunk. Reducing DDIM steps has a larger
latency impact, but changes policy behavior and should be validated per task.

## Notes

- New checkpoints include normalization statistics; real-world SCDP checkpoints also
  include camera calibration.
- For legacy checkpoints that reference `dataset_stats_path`, pass `--dataset-dir` at
  evaluation or inference time.
- Meta-World projection assumes the original `corner2` camera geometry.
- Datasets, checkpoints, and videos are excluded by `.gitignore`.

## Validation

```bash
uv sync --locked --extra dev
uv run ruff check scripts tests src/lerobot/policies
uv run pytest
uv build
```

Full training, evaluation, and speed benchmarks require suitable data and a CUDA GPU.

## License

This repository is based on LeRobot 0.4.1. See `NOTICE` for attribution and `LICENSE`
for the Apache License 2.0.

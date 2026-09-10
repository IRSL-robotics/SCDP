# SCDP

Official implementation of **Spatially Conditioned Diffusion Policy (SCDP)**,
accepted at **CoRL 2026**.

This repository is a focused fork of [LeRobot](https://github.com/huggingface/lerobot)
0.4.1. It provides SCDP and Diffusion Policy baselines for Meta-World, together with
real-world training and inference utilities.

## Setup

Linux, Python 3.10, and a CUDA GPU are recommended. Install
[uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/IRSL-robotics/SCDP.git
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
| Optimizer | Adam, `1e-4`, betas `(0.9, 0.999)` |
| LR scheduler | None |
| Noise scheduler / inference steps | DDIM / 16 |
| SCDP sample step | 8 |
| Epoch indices / evaluation interval | `0-1000` / 100 |

### Reported result

| Task | SCDP success rate |
| --- | ---: |
| Assembly-v3 | `91.7 ± 2.9%` |

This is the three-seed mean and sample standard deviation of the best evaluated checkpoint.

## Real world

The real-world scripts use a local LeRobot v3 dataset. Robot drivers are intentionally
not included.

| Feature | Expected content |
| --- | --- |
| `observation.images.cam_3` | RGB image from one fixed camera |
| `observation.state` | robot state; first 3 values are end-effector XYZ |
| `action` | action vector; first 3 values are XYZ deltas |

SCDP requires at least three state/action dimensions and a calibrated fixed camera.
Calibration parsing and projection are isolated in
`src/lerobot/policies/ours_diffusion_real/camera_geometry.py`. New calibration JSON
files should use this column-vector pinhole convention:

```text
p_camera_h = T_camera_from_world @ p_world_h
scale * [u, v, 1]^T = K @ [x_camera, y_camera, z_camera]^T
```

Here, `p_world_h = [X, Y, Z, 1]^T`, `p_camera_h` is homogeneous camera space,
and `(u, v)` is a pixel coordinate.

| JSON field | Required convention |
| --- | --- |
| `camera_image_size` | `[height, width]` of the stored image |
| `camera_intrinsics` | Pixel-space `K`, 3x3 with bottom row `[0, 0, 1]` |
| `world_to_camera_matrix` | `T_camera_from_world`, rigid 4x4; rotation orthonormal (det +1), bottom row `[0, 0, 0, 1]` |

The camera frame uses `+x` right, `+y` down, and `+z` forward; projected points must
have positive camera-space `z`. The first three state values, the first three action
deltas, and the transform translation must share one world frame and length unit.
Images must be undistorted, and `K` must match their exact stored resolution. Center
crop is supported; random crop is not.

Replace the values in `configs/real_camera.example.json`, then validate the dataset
and calibration before training:

```bash
uv run python scripts/validate_real_dataset.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --decode-frame
```

The loader and projector can also be used directly in a robot integration:

```python
from lerobot.policies.ours_diffusion_real.camera_geometry import PinholeCameraProjector, load_camera_calibration

projector = PinholeCameraProjector(load_camera_calibration("camera.json"))
grid_xy = projector(points_world)  # [..., 3] world XYZ -> [..., 2] grid_sample coordinates
```

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

Inference reconstructs the recurrent 3-D trajectory once and projects every point with
one batched matrix operation per denoising step. DDIM coefficients and geometry stay
cached on the device. The default `reduce-overhead` path compiles the image encoder and
captures all 16 trajectory-projection, feature-sampling, U-Net, and DDIM-update steps
in one fixed-shape CUDA Graph. Meta-World and real-world policies share this path.

The Meta-World evaluation script and `RealPolicyRunner` enable it by default. Direct
policy users can call `policy.eval()` followed by `policy.optimize_for_inference()`.

Measured on an NVIDIA GeForce RTX 5090 with PyTorch 2.9.1, batch size 1, 16 DDIM
steps, and 10 warm-up / 50 timed calls:

| Target | Before p50 | Optimized p50 | Speedup | Reduction | Max abs. difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Meta-World SCDP | 69.113 ms | 13.886 ms | 4.977x | 79.9% | 0.001154 |
| Real SCDP | 68.143 ms | 13.819 ms | 4.931x | 79.7% | 0.000775 |

```bash
uv run python scripts/benchmark_inference.py \
  --target both \
  --warmup 10 \
  --iterations 50
```

The optimized call amortizes to 1.736 ms per Meta-World action and 1.727 ms per
real-world action because each call produces an eight-action chunk. The first request
compiles each fixed input shape and may take one to two minutes, so initialize
`RealPolicyRunner` once and keep the robot process alive. Use `--compile-mode default`
if CUDA Graph capture is unavailable, or `--no-compile-inference` to disable
compilation. Reducing DDIM steps changes policy behavior and should be validated per
task.

## Notes

- New checkpoints include normalization statistics; real-world SCDP checkpoints also
  include camera calibration.
- For legacy checkpoints that reference `dataset_stats_path`, pass `--dataset-dir` at
  evaluation or inference time.
- Meta-World projection assumes the original `corner2` camera geometry.
- CUDA `grid_sample` backward and diffusion action sampling are stochastic. Compare
  three-seed best-checkpoint aggregates; checkpoint curves and fresh-process rollout
  estimates need not be bitwise identical.
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

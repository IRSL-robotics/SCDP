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
| Epoch indices / evaluation epochs | `0-1000` / `100, 200, ..., 1000` |

### Full benchmark

Run all 50 tasks with seeds `0,1,2`. Add `--collect-missing` the first time if
the 20-demo datasets are not present:

```bash
bash scripts/run_metaworld_benchmark.sh --policy scdp --collect-missing
```

Use `--policy both` to include the Diffusion Policy baseline, or
`--tasks assembly,push` for a subset. The script prints the best-checkpoint
three-seed mean and sample standard deviation, then writes `summary.csv` and
`summary.json` under `outputs/metaworld_benchmark`. Completed runs are skipped;
use `--summarize-only` to aggregate existing metrics without training.

#### Dashboard

Monitor the benchmark from a browser while training is running or after it finishes.
In a separate terminal on the Linux machine running the benchmark, start the
dashboard from the repository root:

```bash
python3 scripts/benchmark_dashboard.py
```

Open <http://127.0.0.1:8080>. By default, the dashboard shows SCDP runs for all
50 tasks and seeds `0,1,2` under `outputs/metaworld_benchmark`. It uses only the
Python standard library and reads existing metrics, logs, checkpoints, and local
process state; generating `summary.csv` or `summary.json` is not required.

- The overview shows completed, running, queued, evaluated, and failed/interrupted
  runs. Overall progress is the fraction of runs that have completed.
- Search by task name or use the status filter. Each seed cell shows the best
  recorded checkpoint success rate. Task means are provisional until every selected
  seed has an evaluation; the sample standard deviation appears once all selected
  seeds have results (with at least two seeds).
- Click a seed cell to see its epoch and evaluation progress, best/latest success
  rates, success-rate and training-loss curves, and recent logs.

The page and open run details refresh every five seconds. Use **새로고침** for an
immediate update. Stop the dashboard with `Ctrl+C`; benchmark training continues
independently.

Match the dashboard options to your benchmark when using a different output root,
policy, task subset, or seeds. For example:

```bash
python3 scripts/benchmark_dashboard.py \
  --output-root outputs/my_benchmark \
  --policy dp \
  --tasks assembly,push \
  --seeds 0,1,2 \
  --port 8081
```

Open <http://127.0.0.1:8081> for this example. The dashboard displays one policy
per server (`scdp` or `dp`); for a benchmark run with `--policy both`, start two
dashboard servers on different ports to view both policies. If you changed the
dataset root or training schedule, also pass matching `--data-root`, `--num-epochs`,
`--eval-freq`, and `--num-eval-episodes` values. Their defaults are
`data/metaworld`, `1001`, `100`, and `20`, respectively. Use
`python3 scripts/benchmark_dashboard.py --help` for all options.

For a benchmark on a remote server, start the dashboard there, then run this SSH
tunnel on your local computer (replace `user@server` with your SSH destination):

```bash
ssh -N -L 8080:127.0.0.1:8080 user@server
```

Keep the tunnel open and visit <http://127.0.0.1:8080> in your local browser. If you
changed the dashboard port, update the forwarding ports and browser URL accordingly.

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

Train SCDP:

```bash
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

## Notes

- New checkpoints include normalization statistics; real-world SCDP checkpoints also
  include camera calibration.
- For legacy checkpoints that reference `dataset_stats_path`, pass `--dataset-dir` at
  evaluation or inference time.
- Meta-World projection assumes the original `corner2` camera geometry.
- CUDA `grid_sample` backward and diffusion action sampling are stochastic. Compare
  three-seed best-checkpoint aggregates; checkpoint curves and fresh-process rollout
  estimates need not be bitwise identical.

## Validation

```bash
uv sync --locked --extra dev
uv run ruff check scripts tests src/lerobot/policies
uv run pytest
uv build
```

Full training and evaluation require suitable data and a CUDA GPU.

## License

This repository is based on LeRobot 0.4.1. See `NOTICE` for attribution and `LICENSE`
for the Apache License 2.0.

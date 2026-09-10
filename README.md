<h1 align="center">SCDP</h1>

<p align="center">
  <strong>Spatially Conditioned Diffusion Policy</strong><br>
  Learning Precise and Robust Manipulation with a Single RGB Camera
</p>

<p align="center">
  Seoyoon Kim · Kanghyun Kim · Dongwoo Ko · Yeong Jin Heo · Min Jun Kim
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.14535"><img src="https://img.shields.io/badge/arXiv-2606.14535-b31b1b" alt="arXiv:2606.14535"></a>
  <img src="https://img.shields.io/badge/CoRL-2026-2563eb" alt="Accepted at CoRL 2026">
</p>

<p align="center">
  <a href="#setup">Setup</a> ·
  <a href="#meta-world">Meta-World</a> ·
  <a href="#real-world">Real world</a> ·
  <a href="#citation">Citation</a>
</p>

Official implementation of SCDP, with Meta-World benchmarks and real-world training
and inference. Built on [LeRobot](https://github.com/huggingface/lerobot) 0.4.1.

## Setup

Linux, Python 3.10, and a CUDA GPU are recommended. Install
[uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/IRSL-robotics/SCDP.git
cd SCDP
uv sync --locked
```

For headless Meta-World runs, set `export MUJOCO_GL=egl`.

## Meta-World

### 1. Collect demonstrations

Collect 20 successful expert episodes in LeRobot v3 format:

```bash
uv run python scripts/collect_metaworld.py \
  --task-name assembly \
  --episodes 20 \
  --seed 0
```

### 2. Train

```bash
# Diffusion Policy baseline
uv run python scripts/metaworld_dp_train.py \
  --task-name assembly \
  --seed 0

# SCDP
uv run python scripts/metaworld_ours_dp_train.py \
  --task-name assembly \
  --seed 0 \
  --sample-step 8
```

### 3. Evaluate

```bash
# Diffusion Policy baseline
uv run python scripts/metaworld_dp_eval.py \
  --task-name assembly \
  --checkpoint outputs/assembly/dp/seed_0/checkpoints/final \
  --episodes 20 \
  --seed 0

# SCDP
uv run python scripts/metaworld_ours_dp_eval.py \
  --task-name assembly \
  --checkpoint outputs/assembly/scdp/seed_0/checkpoints/final \
  --episodes 20 \
  --seed 0
```

Use `--video-dir outputs/eval-videos` to save rollouts. By default, training writes
checkpoints and metrics under `outputs/<task>/{dp,scdp}/seed_<seed>`.

<details>
<summary>Default configuration</summary>

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

</details>

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

**Dashboard:** Run `python3 scripts/benchmark_dashboard.py` in a separate terminal
and open <http://127.0.0.1:8080> to monitor progress and results.

## Real world

Use a local LeRobot v3 dataset with fixed-camera RGB images
(`observation.images.cam_3`), robot state (`observation.state`, first 3 values: XYZ),
and actions (`action`, first 3 values: XYZ deltas).

Update [configs/real_camera.example.json](configs/real_camera.example.json) with your
image size, camera intrinsics, and world-to-camera transform. Use the same world
frame and length unit for state, action deltas, and calibration translation.

```bash
# Validate the dataset and calibration
uv run python scripts/validate_real_dataset.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --decode-frame

# Train SCDP
uv run python scripts/real_ours_dp_train.py \
  --dataset-dir /path/to/task/lerobot \
  --camera-calibration configs/real_camera.example.json \
  --output-dir outputs/real/scdp

# Run inference on one observation
uv run python scripts/real_policy_inference.py \
  --policy scdp \
  --checkpoint outputs/real/scdp/checkpoints/final \
  --image frame.png \
  --state 0.49 0.00 0.15 1.00 0.00 0.00 0.00 0.00
```

Inference returns actions in dataset units; robot control must be integrated separately.

<details>
<summary>Development and implementation notes</summary>

### Notes

- New checkpoints include normalization statistics; real-world SCDP checkpoints also
  include camera calibration.
- For legacy checkpoints that reference `dataset_stats_path`, pass `--dataset-dir` at
  evaluation or inference time.
- Meta-World projection assumes the original `corner2` camera geometry.
- CUDA `grid_sample` backward and diffusion action sampling are stochastic. Compare
  three-seed best-checkpoint aggregates; checkpoint curves and fresh-process rollout
  estimates need not be bitwise identical.

### Validation

```bash
uv sync --locked --extra dev
uv run ruff check scripts tests src/lerobot/policies
uv run pytest
uv build
```

Full training and evaluation require suitable data and a CUDA GPU.

</details>

## License

Released under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for LeRobot attribution.

## Citation

If you use SCDP in your research, please cite our [paper](https://arxiv.org/abs/2606.14535):

```bibtex
@misc{kim2026scdp,
  title         = {Spatially Conditioned Diffusion Policy: Learning Precise and
                   Robust Manipulation with a Single {RGB} Camera},
  author        = {Seoyoon Kim and Kanghyun Kim and Dongwoo Ko and
                   Yeong Jin Heo and Min Jun Kim},
  year          = {2026},
  eprint        = {2606.14535},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.14535}
}
```

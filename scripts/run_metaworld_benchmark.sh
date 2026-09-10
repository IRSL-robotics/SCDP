#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLICY="scdp"
DATA_ROOT="data/metaworld"
OUTPUT_ROOT="outputs/metaworld_benchmark"
TASK_FILTER="all"
SEED_CSV="0,1,2"
COLLECT_MISSING=0
SUMMARY_ONLY=0
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_metaworld_benchmark.sh [options] [-- TRAIN_ARGS...]

Runs every selected Meta-World task with three seeds and summarizes the best
evaluation success rate from each training run.

Options:
  --policy scdp|dp|both     Policies to train (default: scdp)
  --tasks all|TASKS         Comma-separated task names (default: all 50)
  --seeds SEEDS             Comma-separated seeds (default: 0,1,2)
  --data-root PATH          Dataset root (default: data/metaworld)
  --output-root PATH        Result root (default: outputs/metaworld_benchmark)
  --collect-missing         Collect 20 expert demos for missing datasets
  --summarize-only         Do not train; summarize existing metrics
  -h, --help                Show this help

Arguments after -- are forwarded to each training script. For example:
  bash scripts/run_metaworld_benchmark.sh --tasks assembly,push -- \
    --num-epochs 1001 --eval-freq 100
EOF
}

while (($#)); do
  case "$1" in
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --tasks)
      TASK_FILTER="$2"
      shift 2
      ;;
    --seeds)
      SEED_CSV="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --collect-missing)
      COLLECT_MISSING=1
      shift
      ;;
    --summarize-only)
      SUMMARY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$POLICY" in
  scdp)
    POLICIES=("scdp")
    ;;
  dp)
    POLICIES=("dp")
    ;;
  both)
    POLICIES=("scdp" "dp")
    ;;
  *)
    echo "--policy must be scdp, dp, or both." >&2
    exit 2
    ;;
esac

IFS=',' read -r -a SEEDS <<< "$SEED_CSV"
if (("${#SEEDS[@]}" != 3)); then
  echo "--seeds must contain exactly three comma-separated seeds." >&2
  exit 2
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
done

TASK_CONFIG="src/lerobot/envs/metaworld_config.json"
mapfile -t ALL_TASKS < <(
  uv run python -c \
    'import json, sys; data=json.load(open(sys.argv[1])); print(*[x.removesuffix("-v3") for x in data["TASK_NAME_TO_ID"]], sep="\n")' \
    "$TASK_CONFIG"
)
if (("${#ALL_TASKS[@]}" != 50)); then
  echo "Expected 50 Meta-World tasks, found ${#ALL_TASKS[@]}." >&2
  exit 1
fi

declare -A KNOWN_TASKS=()
for task in "${ALL_TASKS[@]}"; do
  KNOWN_TASKS["$task"]=1
done

if [[ "$TASK_FILTER" == "all" ]]; then
  TASKS=("${ALL_TASKS[@]}")
else
  IFS=',' read -r -a TASKS <<< "$TASK_FILTER"
  for index in "${!TASKS[@]}"; do
    TASKS[$index]="${TASKS[$index]%-v3}"
    if [[ -z "${KNOWN_TASKS[${TASKS[$index]}]+known}" ]]; then
      echo "Unknown Meta-World task: ${TASKS[$index]}" >&2
      exit 2
    fi
  done
fi

DATA_ROOT="$(realpath -m "$DATA_ROOT")"
OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ((SUMMARY_ONLY == 0)); then
  for task in "${TASKS[@]}"; do
    dataset_dir="$DATA_ROOT/$task/lerobot"
    if [[ -f "$dataset_dir/meta/info.json" ]]; then
      continue
    fi
    if ((COLLECT_MISSING == 0)); then
      echo "Missing dataset: $dataset_dir" >&2
      echo "Run with --collect-missing or collect it separately." >&2
      exit 1
    fi
    if [[ -e "$dataset_dir" ]]; then
      echo "Incomplete dataset directory exists: $dataset_dir" >&2
      echo "Move it aside before collecting again." >&2
      exit 1
    fi

    echo "[benchmark] collecting task=$task"
    uv run python scripts/collect_metaworld.py \
      --task-name "$task" \
      --dataset-dir "$dataset_dir" \
      --episodes 20 \
      --seed 0
  done

  for policy in "${POLICIES[@]}"; do
    if [[ "$policy" == "scdp" ]]; then
      train_script="scripts/metaworld_ours_dp_train.py"
    else
      train_script="scripts/metaworld_dp_train.py"
    fi

    for task in "${TASKS[@]}"; do
      dataset_dir="$DATA_ROOT/$task/lerobot"
      for seed in "${SEEDS[@]}"; do
        run_dir="$OUTPUT_ROOT/$task/$policy/seed_$seed"
        final_weights="$run_dir/checkpoints/final/model.safetensors"
        metrics_path="$run_dir/metrics.jsonl"

        if [[ -f "$final_weights" && -f "$metrics_path" ]]; then
          echo "[benchmark] skip completed policy=$policy task=$task seed=$seed"
          continue
        fi
        if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -print -quit | grep -q .; then
          echo "Incomplete run directory exists: $run_dir" >&2
          echo "Move it aside or choose another --output-root." >&2
          exit 1
        fi

        mkdir -p "$run_dir"
        echo "[benchmark] train policy=$policy task=$task seed=$seed"
        uv run python "$train_script" \
          --task-name "$task" \
          --dataset-dir "$dataset_dir" \
          --output-dir "$run_dir" \
          --seed "$seed" \
          "${EXTRA_TRAIN_ARGS[@]}" 2>&1 | tee "$run_dir/train.log"
      done
    done
  done
fi

policy_csv="$(IFS=,; echo "${POLICIES[*]}")"
task_csv="$(IFS=,; echo "${TASKS[*]}")"

uv run python - "$OUTPUT_ROOT" "$policy_csv" "$SEED_CSV" "$task_csv" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
policies = sys.argv[2].split(",")
seeds = [int(value) for value in sys.argv[3].split(",")]
tasks = sys.argv[4].split(",")

scores = {}
missing = []
for policy in policies:
    for task in tasks:
        for seed in seeds:
            metrics_path = output_root / task / policy / f"seed_{seed}" / "metrics.jsonl"
            if not metrics_path.is_file():
                missing.append(str(metrics_path))
                continue

            evaluations = []
            with metrics_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON at {metrics_path}:{line_number}"
                        ) from error
                    if "eval/success_rate" in record:
                        evaluations.append(float(record["eval/success_rate"]))

            if not evaluations:
                missing.append(f"{metrics_path} (no evaluation records)")
                continue
            scores[(policy, task, seed)] = max(evaluations)

if missing:
    print("Cannot summarize an incomplete benchmark:", file=sys.stderr)
    for path in missing:
        print(f"  - {path}", file=sys.stderr)
    raise SystemExit(1)

output_root.mkdir(parents=True, exist_ok=True)
summary = {"selection": "best evaluated checkpoint", "seeds": seeds, "policies": {}}
csv_rows = []

print()
print("Best evaluated success rate (%)")
header = ["policy", "task", *[f"seed {seed}" for seed in seeds], "mean", "sample std"]
print("  ".join(f"{name:>12}" for name in header))

for policy in policies:
    task_results = {}
    seed_macro = {seed: [] for seed in seeds}
    for task in tasks:
        values = [scores[(policy, task, seed)] for seed in seeds]
        mean = statistics.mean(values)
        sample_std = statistics.stdev(values)
        task_results[task] = {
            "by_seed": {str(seed): value for seed, value in zip(seeds, values, strict=True)},
            "mean": mean,
            "sample_std": sample_std,
        }
        for seed, value in zip(seeds, values, strict=True):
            seed_macro[seed].append(value)

        rendered = [
            policy,
            task,
            *[f"{100 * value:.1f}" for value in values],
            f"{100 * mean:.1f}",
            f"{100 * sample_std:.1f}",
        ]
        print("  ".join(f"{value:>12}" for value in rendered))
        csv_rows.append([policy, task, *values, mean, sample_std])

    macro_by_seed = {
        str(seed): statistics.mean(seed_macro[seed])
        for seed in seeds
    }
    macro_values = list(macro_by_seed.values())
    overall = {
        "by_seed": macro_by_seed,
        "mean": statistics.mean(macro_values),
        "sample_std": statistics.stdev(macro_values),
    }
    summary["policies"][policy] = {"tasks": task_results, "all_tasks_macro": overall}

    rendered = [
        policy,
        "ALL_TASKS",
        *[f"{100 * value:.1f}" for value in macro_values],
        f"{100 * overall['mean']:.1f}",
        f"{100 * overall['sample_std']:.1f}",
    ]
    print("  ".join(f"{value:>12}" for value in rendered))
    csv_rows.append(
        [policy, "ALL_TASKS", *macro_values, overall["mean"], overall["sample_std"]]
    )

csv_path = output_root / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["policy", "task", *[f"seed_{seed}" for seed in seeds], "mean", "sample_std"])
    writer.writerows(csv_rows)

json_path = output_root / "summary.json"
json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved {csv_path}")
print(f"Saved {json_path}")
PY

#!/usr/bin/env python3
"""Read-only, dependency-free local dashboard for Meta-World benchmark runs."""

import argparse
import csv
import io
import json
import math
import re
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = {"collecting", "starting", "training", "evaluating", "finalizing"}
TRAIN_SCRIPTS = {"scdp": "metaworld_ours_dp_train.py", "dp": "metaworld_dp_train.py"}
LAUNCHERS = {"run_metaworld_benchmark_gpu_mig.sh", "run_metaworld_benchmark_2gpu.sh"}
PROGRESS = re.compile(r"epoch\s+(\d+):\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)")
EPISODE = re.compile(r"\[eval\] episode=(\d+)/(\d+)")


def iso_time(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat() if timestamp else None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def option(args, name, default=None):
    # The last option wins, as in argparse (launcher defaults may be overridden).
    result = default
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            result = args[index + 1]
        elif arg.startswith(name + "="):
            result = arg.split("=", 1)[1]
    return result


def tail(path, limit=16384):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_tsv(path):
    try:
        return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))
    except OSError:
        return []


def processes():
    """Inspect argv only; never import training code or initialize CUDA."""
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            if any(Path(arg).name in {*TRAIN_SCRIPTS.values(), *LAUNCHERS,
                                     "run_metaworld_benchmark.sh", "collect_metaworld.py"}
                   for arg in args[:3]):
                found[int(entry.name)] = args
        except (OSError, ValueError):
            continue
    return found


class MetricsCache:
    """Consume only appended complete JSONL lines; handle restarts and partial writes."""

    def __init__(self):
        self.files = {}

    def read(self, path):
        try:
            stat = path.stat()
        except OSError:
            self.files.pop(path, None)
            return {"epoch": None, "step": None, "loss": None, "evaluations": [], "loss_history": []}
        state = self.files.get(path)
        identity = (stat.st_dev, stat.st_ino)
        if state is None or state["identity"] != identity or stat.st_size < state["size"]:
            state = {"identity": identity, "size": 0, "offset": 0, "epoch": None, "step": None,
                     "loss": None, "evaluations": [], "loss_history": []}
            self.files[path] = state
        try:
            with path.open("rb") as stream:
                stream.seek(state["offset"])
                chunk = stream.read()
        except OSError:
            return state
        end = chunk.rfind(b"\n") + 1
        state["offset"] += end
        state["size"] = stat.st_size
        for line in chunk[:end].splitlines():
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            epoch, step = record.get("epoch"), record.get("step")
            if not finite(epoch) or epoch < 0:
                continue
            state["epoch"] = int(epoch)
            if finite(step) and step >= 0:
                state["step"] = int(step)
            loss = record.get("train/loss")
            if finite(loss):
                state["loss"] = loss
                state["loss_history"].append({"epoch": int(epoch), "step": state["step"], "loss": loss})
            success = record.get("eval/success_rate")
            if finite(success) and 0 <= success <= 1:
                state["evaluations"].append({"epoch": int(epoch), "success_rate": success})
        return state


class Dashboard:
    def __init__(self, output_root, data_root, tasks, seeds, policy="scdp", num_epochs=1001,
                 eval_freq=100, num_eval_episodes=20):
        self.output_root = Path(output_root).resolve()
        self.data_root = Path(data_root).resolve()
        self.tasks, self.seeds, self.policy = tasks, seeds, policy
        self.num_epochs, self.eval_freq = num_epochs, eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.metrics = MetricsCache()
        self.lock = threading.Lock()
        self.cached_at = 0
        self.cached = None
        self.details = {}

    def matches_root(self, args, flag, default):
        value = Path(option(args, flag, str(default)))
        if not value.is_absolute():
            value = ROOT / value
        return value.resolve() == self.output_root

    def launcher_data(self, procs):
        if self.policy != "scdp":
            return None, {}, {}, {}, False
        directories = sorted(path for path in self.output_root.glob("launcher_*") if path.is_dir())
        # Directory names contain launch timestamps; use mtime to compare old/new naming conventions.
        latest = max(directories, key=lambda p: p.stat().st_mtime, default=None)
        assignments, results, devices = {}, {}, {}
        if latest:
            devices = {row["slot"]: row.get("device") for row in read_tsv(latest / "devices.tsv")
                       if row.get("slot")}
            for row in read_tsv(latest / "assignments.tsv"):
                try:
                    key = (row["task"], int(row["seed"]))
                    row["pid"] = int(row["pid"])
                    assignments[key] = row
                except (KeyError, ValueError, TypeError):
                    continue
            for row in read_tsv(latest / "results.tsv"):
                try:
                    results[(row["task"], int(row["seed"]))] = int(row["exit_code"])
                except (KeyError, ValueError, TypeError):
                    continue
        active = any(any(Path(arg).name in LAUNCHERS for arg in args[:3])
                     and self.matches_root(args, "--output-root", ROOT / "outputs/metaworld_benchmark")
                     for args in procs.values())
        return latest, assignments, results, devices, active

    def build_run(self, task, seed, procs, assignment, exit_code, devices):
        run_dir = self.output_root / task / self.policy / f"seed_{seed}"
        metrics_path = run_dir / "metrics.jsonl"
        metrics = self.metrics.read(metrics_path)
        args = next((args for args in procs.values()
                     if TRAIN_SCRIPTS[self.policy] in [Path(arg).name for arg in args[:3]]
                     and self.matches_run(args, task, seed, run_dir)), None)
        worker_args = procs.get(assignment.get("pid"), [])
        worker_alive = ("run_metaworld_benchmark.sh" in [Path(arg).name for arg in worker_args[:3]]
                        and option(worker_args, "--tasks") == task
                        and option(worker_args, "--seeds") == str(seed)
                        and self.matches_root(worker_args, "--output-root", self.output_root))
        live = args is not None or worker_alive
        num_epochs = int(option(args or worker_args, "--num-epochs", self.num_epochs))
        eval_freq = int(option(args or worker_args, "--eval-freq", self.eval_freq))
        eval_total = int(option(args or worker_args, "--num-eval-episodes", self.num_eval_episodes))
        train_log = run_dir / "train.log"
        log_path = train_log
        if not train_log.is_file() and assignment.get("log"):
            candidate = Path(assignment["log"]).resolve()
            if candidate.is_relative_to(self.output_root):
                log_path = candidate
        log = tail(log_path)
        progress = list(PROGRESS.finditer(log))
        frame = progress[-1] if progress else None
        epoch = metrics["epoch"]
        fraction = 0.0
        if frame and (epoch is None or int(frame[1]) >= epoch):
            epoch = int(frame[1])
            fraction = int(frame[2]) / max(1, int(frame[3]))
        evaluations = metrics["evaluations"]
        latest = evaluations[-1] if evaluations else None
        eval_epoch = latest["epoch"] if latest else None
        early_stopped = "[train] early-stopped;" in log or "reached early-stop success rate" in log
        final_exists = (run_dir / "checkpoints/final/model.safetensors").is_file() and metrics_path.is_file()
        # While a process is alive, a final weight file can still be being written.
        completed = final_exists and (not live or "[train] early-stopped;" in log or "[train] completed;" in log)
        eval_done = None
        if completed:
            status = "completed"
        elif live:
            if args is None and (self.data_root / ".benchmark_locks" / f"{task}.collecting").exists():
                status = "collecting"
            elif epoch is None:
                status = "starting"
            elif early_stopped or final_exists:
                status = "finalizing"
            elif epoch > 0 and epoch % eval_freq == 0 and (eval_epoch is None or eval_epoch < epoch):
                status = "evaluating"
                episodes = list(EPISODE.finditer(log))
                # Only count episodes after the current epoch's training frame.
                current = [match for match in episodes if frame is None or match.start() > frame.start()]
                eval_done = int(current[-1][1]) if current else 0
                eval_total = int(current[-1][2]) if current else eval_total
            else:
                status = "training"
        elif exit_code is not None and exit_code != 0:
            status = "failed"
        elif assignment or metrics_path.exists() or train_log.exists():
            status = "interrupted"
        else:
            status = "queued"
        updated = []
        for path in (metrics_path, log_path):
            try:
                updated.append(path.stat().st_mtime)
            except OSError:
                pass
        run = {
            "task": task, "seed": seed, "status": status, "epoch": epoch, "max_epoch": num_epochs - 1,
            "progress_pct": 100.0 if completed else min(99.9, 100 * ((epoch or 0) + fraction) / num_epochs),
            "step": metrics["step"], "loss": metrics["loss"],
            "best_success": max((item["success_rate"] for item in evaluations), default=None),
            "latest_success": latest["success_rate"] if latest else None, "eval_epoch": eval_epoch,
            "eval_episodes_done": eval_done, "eval_episodes_total": eval_total if eval_done is not None else None,
            "device": devices.get(assignment.get("slot")), "updated_at": iso_time(max(updated, default=0)),
            "early_stopped": early_stopped, "exit_code": exit_code,
        }
        history = metrics["loss_history"]
        stride = max(1, math.ceil(len(history) / 300))
        sampled = history[::stride]
        if history and sampled[-1] is not history[-1]:
            sampled.append(history[-1])
        self.details[(task, seed)] = {
            **run, "evaluations": list(evaluations), "loss_history": sampled,
            "log_tail": re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log).replace("\r", "\n").strip()[-12000:],
        }
        return run

    @staticmethod
    def matches_run(args, task, seed, run_dir):
        output = Path(option(args, "--output-dir", ""))
        if not output.is_absolute():
            output = ROOT / output
        return (output.resolve() == run_dir and option(args, "--task-name", "").removesuffix("-v3") == task
                and option(args, "--seed", "0") == str(seed))

    def snapshot(self):
        with self.lock:
            if self.cached is not None and time.monotonic() - self.cached_at < 2:
                return self.cached
            procs = processes()
            directory, assignments, results, devices, launcher_active = self.launcher_data(procs)
            tasks, counts = [], Counter()
            evaluated = 0
            for task in self.tasks:
                runs = [self.build_run(task, seed, procs, assignments.get((task, seed), {}),
                                       results.get((task, seed)), devices) for seed in self.seeds]
                counts.update(run["status"] for run in runs)
                scores = [run["best_success"] for run in runs if run["best_success"] is not None]
                evaluated += len(scores)
                tasks.append({"task": task, "runs": runs,
                              "completed": sum(run["status"] == "completed" for run in runs),
                              "evaluated": len(scores), "best_mean": statistics.mean(scores) if scores else None,
                              "best_std": statistics.stdev(scores) if len(scores) == len(self.seeds) > 1 else None})
            total = len(self.tasks) * len(self.seeds)
            self.cached = {
                "updated_at": iso_time(time.time()), "output_root": str(self.output_root), "policy": self.policy,
                "seeds": self.seeds, "num_epochs": self.num_epochs, "eval_freq": self.eval_freq,
                "launcher": {"name": directory.name if directory else None,
                             "active": launcher_active, "slots": len(devices)},
                "summary": {"total": total, "completed": counts["completed"],
                            "running": sum(counts[status] for status in ACTIVE), "queued": counts["queued"],
                            "failed": counts["failed"], "interrupted": counts["interrupted"], "evaluated": evaluated,
                            "completed_tasks": sum(task["completed"] == len(self.seeds) for task in tasks),
                            "completion_pct": 100 * counts["completed"] / total},
                "tasks": tasks,
            }
            self.cached_at = time.monotonic()
            return self.cached

    def detail(self, task, seed):
        self.snapshot()
        with self.lock:
            return self.details[(task, seed)]


def handler_for(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in ("/", "/index.html"):
                self.respond(200, Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self.respond_json(dashboard.snapshot())
            elif url.path == "/api/run":
                query = parse_qs(url.query)
                try:
                    task, seed = query["task"][0], int(query["seed"][0])
                except (KeyError, ValueError, IndexError):
                    self.respond_json({"error": "task and seed are required"}, 400)
                    return
                if task not in dashboard.tasks or seed not in dashboard.seeds:
                    self.respond_json({"error": "Unknown task or seed"}, 404)
                    return
                self.respond_json(dashboard.detail(task, seed))
            else:
                self.respond_json({"error": "Not found"}, 404)

        def respond_json(self, value, status=200):
            self.respond(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                         "application/json; charset=utf-8")

        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            if len(args) < 2 or str(args[1]) != "200":
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/metaworld_benchmark")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/metaworld")
    parser.add_argument("--policy", choices=("scdp", "dp"), default="scdp")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--num-epochs", type=int, default=1001)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    args = parser.parse_args()
    config = json.loads((ROOT / "src/lerobot/envs/metaworld_config.json").read_text())
    known = [task.removesuffix("-v3") for task in config["TASK_NAME_TO_ID"]]
    tasks = known if args.tasks == "all" else [task.strip().removesuffix("-v3") for task in args.tasks.split(",")]
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in known for task in tasks):
        parser.error("--tasks must contain unique Meta-World task names")
    try:
        seeds = [int(seed) for seed in args.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        parser.error("--seeds must be unique nonnegative integers")
    if min(args.num_epochs, args.eval_freq, args.num_eval_episodes) <= 0:
        parser.error("epoch and evaluation counts must be positive")
    dashboard = Dashboard(args.output_root, args.data_root, tasks, seeds, args.policy,
                          args.num_epochs, args.eval_freq, args.num_eval_episodes)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(dashboard))
    print(f"Benchmark dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"Reading {args.output_root} ({len(tasks)} tasks × {len(seeds)} seeds)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

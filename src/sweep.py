"""Sweep driver.

`prepare`  builds every compressed representation from runs/baseline/best.pt, writes
           runs/queue.txt + runs/index.csv, then launches the first job.
`launch-next`  launches the next not-yet-completed job (one at a time, since we have a
           single GPU). Call this when the autopilot wakes you on a job completion.
`status`   prints queue progress.

A run is "completed" iff runs/<run_name>/summary.json exists. Jobs are launched via
`exp` so they survive disconnects and the autopilot can see them.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import torch

from .compress import compress, total_bytes
from .data import get_loaders
from .models import resnet20
from .utils import load_config

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "/workspace/bin/exp"
VENV_PY = sys.executable  # the venv python running this module
QUEUE_PATH = "runs/queue.txt"
INDEX_PATH = "runs/index.csv"
ATTEMPTS_PATH = "runs/attempts.json"
COMPRESSED_DIR = "compressed"
MAX_ATTEMPTS = 3


def knob_str(knob):
    if isinstance(knob, (list, tuple)):
        return "_".join(str(x) for x in knob)
    return str(knob)


def enumerate_runs(sweep_cfg):
    """Yield (technique, knob, knobstr) for every (technique, knob) combination."""
    for entry in sweep_cfg["techniques"]:
        tech = entry["technique"]
        if "knob_pairs" in entry:
            for pair in entry["knob_pairs"]:
                yield tech, tuple(pair), knob_str(pair)
        else:
            for knob in entry["knobs"]:
                yield tech, knob, knob_str(knob)


def build_one(sd, technique, knob, train_loader, device, seed):
    comp = compress(sd, technique, knob, train_loader=train_loader,
                    device=device, seed=seed)
    cbytes, bbytes = total_bytes(comp, sd)
    return {
        "technique": technique,
        "knob": knob,
        "compressed_bytes": int(cbytes),
        "baseline_bytes": int(bbytes),
        "compression_ratio": cbytes / bbytes,
        "compressed": comp,
    }


def prepare(args):
    cfg = load_config(args.config)
    sweep_cfg = load_config(args.sweep)
    seed = cfg.get("seed", 42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.baseline_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # snip/fisher need a train loader; build once.
    train_loader, _, _ = get_loaders(batch_size=cfg["batch_size"],
                                     num_workers=cfg.get("num_workers", 4))

    os.makedirs(COMPRESSED_DIR, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Build all compressed reps, grouped per technique so we can pick distill extras.
    per_tech = {}   # tech -> list of (knobstr, ratio, path, knob)
    index_rows = []
    for tech, knob, ks in enumerate_runs(sweep_cfg):
        # move model+loader to device only when needed (snip/fisher/aqlm calibration)
        need_dev = tech in ("snip", "fisher_prune", "aqlm")
        dev = device if need_dev else torch.device("cpu")
        blob = build_one(sd, tech, knob, train_loader if need_dev else None,
                         dev, seed)
        name = f"{tech}_{ks}"
        path = os.path.join(COMPRESSED_DIR, f"{name}.pt")
        torch.save(blob, path)
        per_tech.setdefault(tech, []).append(
            {"knobstr": ks, "ratio": blob["compression_ratio"],
             "path": path, "knob": knob})
        index_rows.append({"run_name": name, "technique": tech, "knob": ks,
                           "mode": "plain", "compressed": path,
                           "compressed_bytes": blob["compressed_bytes"],
                           "compression_ratio": f"{blob['compression_ratio']:.6f}"})
        print(f"built {name:>24}  ratio={blob['compression_ratio']:.4f}", flush=True)

    # Queue: all plain runs (in technique order), then distill extras.
    queue = []
    for tech, knob, ks in enumerate_runs(sweep_cfg):
        name = f"{tech}_{ks}"
        path = os.path.join(COMPRESSED_DIR, f"{name}.pt")
        queue.append(f"{name}|{path}|plain")

    n_extra = int(sweep_cfg.get("distill_extras_per_technique", 2))
    for tech, items in per_tech.items():
        most_compressed = sorted(items, key=lambda d: d["ratio"])[:n_extra]
        for it in most_compressed:
            name = f"{tech}_{it['knobstr']}_distill"
            queue.append(f"{name}|{it['path']}|distill")
            index_rows.append({"run_name": name, "technique": tech,
                               "knob": it["knobstr"], "mode": "distill",
                               "compressed": it["path"],
                               "compressed_bytes": "",
                               "compression_ratio": f"{it['ratio']:.6f}"})

    with open(QUEUE_PATH, "w") as f:
        f.write("\n".join(queue) + "\n")
    with open(INDEX_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_name", "technique", "knob", "mode",
                                          "compressed", "compressed_bytes",
                                          "compression_ratio"])
        w.writeheader()
        w.writerows(index_rows)

    print(f"\nprepared {len(queue)} runs ({len(index_rows)} index rows) -> {QUEUE_PATH}",
          flush=True)
    # Launch the first job.
    launch_next(args)


def _read_queue():
    with open(QUEUE_PATH) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    runs = []
    for ln in lines:
        name, path, mode = ln.split("|")
        runs.append({"name": name, "path": path, "mode": mode})
    return runs


def _completed(name):
    return os.path.exists(os.path.join("runs", name, "summary.json"))


def _exp_status(name):
    try:
        out = subprocess.run([EXP, "status", name], cwd=PROJECT_DIR,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[-1].strip() if out.stdout.strip() else ""
    except Exception:
        return ""


def _attempts():
    if os.path.exists(ATTEMPTS_PATH):
        with open(ATTEMPTS_PATH) as f:
            return json.load(f)
    return {}


def _save_attempts(a):
    with open(ATTEMPTS_PATH, "w") as f:
        json.dump(a, f)


def _launch(run, args):
    name, path, mode = run["name"], run["path"], run["mode"]
    cmd = [EXP, name, "--", VENV_PY, "-m", "src.retrain",
           "--compressed", path, "--mode", mode,
           "--config", args.config, "--baseline-dir", args.baseline_dir,
           "--run-name", name]
    print(f"[sweep] launching {name} (mode={mode})", flush=True)
    subprocess.run(cmd, cwd=PROJECT_DIR)


def launch_next(args):
    runs = _read_queue()

    # 1) if any queued job is currently running, wait.
    for r in runs:
        if not _completed(r["name"]) and _exp_status(r["name"]) == "RUNNING":
            print(f"[sweep] still running: {r['name']} — waiting", flush=True)
            return 0

    # 2) launch the first incomplete job that hasn't exceeded the crash budget.
    attempts = _attempts()
    for r in runs:
        if _completed(r["name"]):
            continue
        n = attempts.get(r["name"], 0)
        if n >= MAX_ATTEMPTS:
            print(f"[sweep] SKIP {r['name']} — {n} attempts, no recovery summary "
                  f"(needs human)", flush=True)
            continue
        attempts[r["name"]] = n + 1
        _save_attempts(attempts)
        _launch(r, args)
        return 0

    # 3) nothing left to launch.
    remaining = [r["name"] for r in runs if not _completed(r["name"])]
    if remaining:
        print(f"[sweep] NO LAUNCHABLE JOBS; stuck on: {remaining}", flush=True)
        return 2
    print("[sweep] QUEUE EMPTY — all runs completed", flush=True)
    return 3


def drain(args, poll=20):
    """Drive the whole queue to completion, one job at a time.

    Each tick calls launch_next: it waits while a job is RUNNING, otherwise launches
    the next incomplete job (relaunching crashed ones up to MAX_ATTEMPTS). Returns when
    the queue is empty (rc 3) or stuck at the attempt cap (rc 2). Designed to be run as
    its own `exp` job so it survives disconnects.
    """
    print(f"[sweep] drain start (poll={poll}s)", flush=True)
    while True:
        rc = launch_next(args)
        if rc == 3:
            print("[sweep] drain: QUEUE EMPTY — done", flush=True)
            return 3
        if rc == 2:
            print("[sweep] drain: STUCK (remaining jobs at attempt cap)", flush=True)
            return 2
        time.sleep(poll)


def status(args):
    runs = _read_queue()
    done = [r["name"] for r in runs if _completed(r["name"])]
    pending = [r["name"] for r in runs if not _completed(r["name"])]
    print(f"completed {len(done)}/{len(runs)}")
    for r in runs:
        mark = "DONE" if _completed(r["name"]) else _exp_status(r["name"]) or "pending"
        print(f"  [{mark:>8}] {r['name']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "launch-next", "status", "drain"])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    ap.add_argument("--baseline-dir", default="runs/baseline")
    ap.add_argument("--tag", default="",
                    help="suffix for queue/index/attempts files so multiple sweeps coexist")
    args = ap.parse_args()

    if args.tag:
        global QUEUE_PATH, INDEX_PATH, ATTEMPTS_PATH
        QUEUE_PATH = f"runs/queue_{args.tag}.txt"
        INDEX_PATH = f"runs/index_{args.tag}.csv"
        ATTEMPTS_PATH = f"runs/attempts_{args.tag}.json"

    if args.command == "prepare":
        prepare(args)
    elif args.command == "launch-next":
        sys.exit(launch_next(args))
    elif args.command == "drain":
        sys.exit(drain(args))
    elif args.command == "status":
        sys.exit(status(args))


if __name__ == "__main__":
    main()

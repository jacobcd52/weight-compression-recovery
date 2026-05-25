"""Retraining harness (dense-from-init recovery).

Loads a compressed file produced by the sweep, reconstructs the best dense
approximation, loads it into a fresh ResNet-20, and retrains with the same
optimizer state machine as the baseline (AdamW + warmup-cosine, bf16) — except
the cosine schedule is rescaled to the 10% step budget so the LR decays within
the short run.

Stops at the recovery threshold (baseline_test_acc - margin) or the budget cap,
whichever comes first. Writes runs/<run-name>/summary.json.
"""
import argparse
import json
import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .data import get_loaders
from .compress import reconstruct
from .engine import build_optimizer, build_scheduler, train_loop
from .models import resnet20
from .utils import CSVLogger, get_device, load_config, set_seed


def load_teacher(baseline_dir, device):
    ckpt = torch.load(os.path.join(baseline_dir, "best.pt"),
                      map_location=device, weights_only=False)
    teacher = resnet20().to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compressed", required=True, help="path to compressed/<name>.pt")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["plain", "distill"], default="plain")
    ap.add_argument("--baseline-dir", default=None,
                    help="dir with baseline best.pt + summary.json (default runs/baseline)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    baseline_dir = args.baseline_dir or (
        "runs/baseline_smoke" if args.smoke else "runs/baseline")
    with open(os.path.join(baseline_dir, "summary.json")) as f:
        base = json.load(f)

    blob = torch.load(args.compressed, map_location="cpu", weights_only=False)
    technique = blob["technique"]
    knob = blob["knob"]
    compressed_bytes = int(blob["compressed_bytes"])
    compression_ratio = float(blob["compression_ratio"])
    did_distill = args.mode == "distill"

    run_name = args.run_name or (
        os.path.splitext(os.path.basename(args.compressed))[0]
        + ("_distill" if did_distill else ""))
    if args.smoke:
        run_name += "_smoke"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Reconstruct dense init from compressed bits.
    init_sd = reconstruct(blob["compressed"])
    model = resnet20().to(device)
    model.load_state_dict(init_sd)

    # Data
    train_loader, test_loader, _ = get_loaders(
        batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 4),
        smoke=args.smoke)
    steps_per_epoch = len(train_loader)

    # Budget + threshold from the *real* baseline metrics.
    baseline_steps = base["baseline_steps"]
    baseline_acc = base["baseline_test_acc"]
    budget_steps = max(steps_per_epoch,
                       int(math.ceil(cfg["budget_fraction"] * baseline_steps)))
    if args.smoke:
        budget_steps = 2 * steps_per_epoch
    threshold = baseline_acc - cfg["recovery_margin_pp"]
    warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, budget_steps, warmup_steps)

    teacher = None
    if did_distill:
        teacher = load_teacher(baseline_dir, device)

    tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    csv_logger = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                           ["epoch", "step", "test_acc", "lr"])

    # init accuracy before any retraining (informative)
    from .utils import eval_full
    init_acc = eval_full(model, test_loader, device)

    print(f"[retrain] run={run_name} tech={technique} knob={knob} mode={args.mode} "
          f"ratio={compression_ratio:.4f} init_acc={init_acc:.2f} "
          f"baseline_acc={baseline_acc:.2f} threshold={threshold:.2f} "
          f"budget_steps={budget_steps} (of {baseline_steps})", flush=True)

    results = train_loop(
        model, train_loader, test_loader, device,
        optimizer=optimizer, scheduler=scheduler, total_steps=budget_steps,
        tb_writer=tb, csv_logger=csv_logger, threshold=threshold,
        teacher=teacher, T=cfg["distill_T"], alpha=cfg["distill_alpha"])

    recovery_steps = results["recovery_steps"]
    recovery_fraction = (recovery_steps / baseline_steps
                         if recovery_steps is not None
                         else cfg["budget_fraction"])  # DNR plotted at the cap

    summary = {
        "run_name": run_name,
        "technique": technique,
        "knob": knob,
        "mode": args.mode,
        "did_distill": did_distill,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        "init_acc": init_acc,
        "recovered": results["recovered"],
        "recovery_steps": recovery_steps,
        "recovery_fraction": recovery_fraction,
        "final_test_acc": results["best_acc"],
        "baseline_test_acc": baseline_acc,
        "baseline_steps": baseline_steps,
        "budget_steps": budget_steps,
        "smoke": args.smoke,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tb.close()
    status = "RECOVERED" if results["recovered"] else "DNR"
    print(f"[retrain] DONE {status} {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()

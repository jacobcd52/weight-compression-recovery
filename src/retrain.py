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
    # Recovery metric is TEST LOSS: recover when full-test CE < baseline test loss.
    baseline_steps = base["baseline_steps"]
    baseline_acc = base["baseline_test_acc"]
    baseline_loss = base["baseline_test_loss"]
    budget_steps = max(steps_per_epoch,
                       int(math.ceil(cfg["budget_fraction"] * baseline_steps)))
    if args.smoke:
        budget_steps = 2 * steps_per_epoch
    loss_threshold = baseline_loss + cfg.get("recovery_loss_margin", 0.0)
    # Short warmup for the short recovery budget (configurable; default ~100 steps).
    warmup_steps = cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_steps = cfg["warmup_epochs"] * steps_per_epoch

    import shutil
    from .utils import eval_full
    teacher = load_teacher(baseline_dir, device) if did_distill else None
    lr_grid = cfg.get("lr_grid", [cfg["lr"]])

    init_acc, init_loss = eval_full(model, test_loader, device)
    print(f"[retrain] run={run_name} tech={technique} knob={knob} mode={args.mode} "
          f"ratio={compression_ratio:.4f} init_acc={init_acc:.2f} init_loss={init_loss:.4f} "
          f"baseline_loss={baseline_loss:.4f} loss_threshold={loss_threshold:.4f} "
          f"warmup={warmup_steps} budget_steps={budget_steps} lr_grid={lr_grid}", flush=True)

    # Recovery cost = MIN over a small LR sweep ("how cheaply can the attacker recover").
    # The retraining recipe is part of the method; for now we sweep only the (cosine) peak LR.
    lr_results, best = [], None
    for lr in lr_grid:
        model.load_state_dict(init_sd)                  # fresh init for each LR
        cfg_lr = dict(cfg); cfg_lr["lr"] = lr
        opt = build_optimizer(model, cfg_lr)
        sch = build_scheduler(opt, budget_steps, warmup_steps)
        csv_path = os.path.join(run_dir, f"metrics_lr{lr:.0e}.csv")
        clog = CSVLogger(csv_path, ["epoch", "step", "test_acc", "test_loss", "lr"])
        res = train_loop(
            model, train_loader, test_loader, device,
            optimizer=opt, scheduler=sch, total_steps=budget_steps,
            tb_writer=None, csv_logger=clog, loss_threshold=loss_threshold,
            teacher=teacher, T=cfg["distill_T"], alpha=cfg["distill_alpha"],
            eval_every_steps=cfg.get("eval_every_steps"), log_prefix=f"lr{lr:.0e} ")
        rf = res["recovery_steps"] / baseline_steps if res["recovered"] else None
        lr_results.append({"lr": lr, "recovered": res["recovered"],
                           "recovery_fraction": rf, "best_loss": res["best_loss"]})
        # prefer recovered (then min recovery_fraction), else min best_loss
        key = (0 if res["recovered"] else 1,
               rf if rf is not None else float("inf"), res["best_loss"])
        if best is None or key < best[0]:
            best = (key, lr, res, csv_path)

    _, best_lr, results, best_csv = best
    shutil.copyfile(best_csv, os.path.join(run_dir, "metrics.csv"))  # best-LR curve for report

    recovery_steps = results["recovery_steps"]
    recovery_fraction = (recovery_steps / baseline_steps
                         if results["recovered"] else cfg["budget_fraction"])

    summary = {
        "run_name": run_name,
        "technique": technique,
        "knob": knob,
        "mode": args.mode,
        "did_distill": did_distill,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": compression_ratio,
        "init_acc": init_acc,
        "init_loss": init_loss,
        "recovered": results["recovered"],
        "recovery_steps": recovery_steps,
        "recovery_fraction": recovery_fraction,
        "best_lr": best_lr,
        "lr_results": lr_results,
        "final_test_acc": results["best_acc"],
        "final_test_loss": results["best_loss"],
        "baseline_test_acc": baseline_acc,
        "baseline_test_loss": baseline_loss,
        "baseline_steps": baseline_steps,
        "budget_steps": budget_steps,
        "smoke": args.smoke,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    status = "RECOVERED" if results["recovered"] else "DNR"
    print(f"[retrain] DONE {status} best_lr={best_lr} "
          f"recovery_fraction={recovery_fraction:.4f}", flush=True)


if __name__ == "__main__":
    main()

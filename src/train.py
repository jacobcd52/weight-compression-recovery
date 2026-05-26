"""Baseline training entry point. Config-driven.

Saves runs/<name>/{best.pt,final.pt}, logs per-step loss to TensorBoard and
per-epoch test accuracy to TB + CSV, and writes runs/<name>/summary.json.
"""
import argparse
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .data import get_loaders
from .engine import build_optimizer, build_scheduler, train_loop
from .models import resnet20, count_params
from .utils import (CSVLogger, compressible_keys, fp32_bytes_of_keys,
                    get_device, load_config, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs, 1024-image subset, for a fast pipeline check")
    ap.add_argument("--lr", type=float, default=None, help="override cfg lr (for LR sweep)")
    ap.add_argument("--run-name", default=None, help="override run_name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.lr is not None:
        cfg["lr"] = args.lr
    run_name = (args.run_name if args.run_name else
                ("baseline_smoke" if args.smoke else cfg.get("run_name", "baseline")))
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(cfg.get("seed", 42))
    device = get_device()

    epochs = 2 if args.smoke else cfg["epochs"]
    warmup_epochs = cfg["warmup_epochs"]
    batch_size = cfg["batch_size"]

    train_loader, test_loader, _ = get_loaders(
        batch_size=batch_size, num_workers=cfg.get("num_workers", 4),
        smoke=args.smoke)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    model = resnet20(num_classes=10, norm=cfg.get("norm", "batch"),
                     groups=cfg.get("gn_groups", 8)).to(device)
    params = count_params(model)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    csv_logger = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                           ["epoch", "step", "test_acc", "test_loss", "lr"])

    best_path = os.path.join(run_dir, "best.pt")

    def on_best(m, epoch, step, acc, loss):  # best = lowest test loss
        torch.save({"state_dict": m.state_dict(), "epoch": epoch, "step": step,
                    "test_acc": acc, "test_loss": loss}, best_path)

    print(f"[train] run={run_name} device={device} epochs={epochs} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} params={params:,}",
          flush=True)

    results = train_loop(
        model, train_loader, test_loader, device,
        optimizer=optimizer, scheduler=scheduler, total_steps=total_steps,
        tb_writer=tb, csv_logger=csv_logger, on_best=on_best)

    torch.save({"state_dict": model.state_dict(), "epoch": epochs - 1,
                "step": results["steps_taken"], "test_acc": results["final_acc"],
                "test_loss": results["final_loss"]},
               os.path.join(run_dir, "final.pt"))

    # Byte accounting reference for the compression ratio denominator.
    sd = model.state_dict()
    ckeys = compressible_keys(sd)
    weight_bytes_fp32 = fp32_bytes_of_keys(sd, ckeys)

    summary = {
        "run_name": run_name,
        "norm": cfg.get("norm", "batch"),
        "baseline_test_acc": results["best_acc"],
        "baseline_final_acc": results["final_acc"],
        "baseline_test_loss": results["best_loss"],     # recovery target (recover below this)
        "baseline_final_loss": results["final_loss"],
        "baseline_steps": total_steps,
        "baseline_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "params_count": params,
        "weight_bytes_fp32": weight_bytes_fp32,
        "n_compressible_keys": len(ckeys),
        "smoke": args.smoke,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tb.close()
    print("[train] DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

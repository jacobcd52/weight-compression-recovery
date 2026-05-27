"""Train the 11M victim from scratch, single-pass, undertrained-by-construction.

Logs train/eval loss to <out>/metrics.csv, saves the victim checkpoint + summary
(target loss = final eval loss, denominator = step count), and plots the loss curve.

    python -m src.llm.train_ref --steps 4000 --batch-size 64 --lr 3e-3
Smoke: python -m src.llm.train_ref --steps 20 --eval-every 10 --out runs/llm_smoke
"""
import argparse
import csv
import json
import math
import os
import time

import torch

from .data import Batcher, load_packed, prepare
from .model import build_model, compressible_names, eval_loss, fp32_bytes, get_weights


def lr_at(step, peak, warmup, total, floor_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--cache-dir", default="data/simplestories")
    ap.add_argument("--out", default="runs/llm_victim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_arr, eval_arr, meta = load_packed(args.cache_dir)
    tb = Batcher(train_arr, args.batch_size, device, seed=args.seed)
    eb = Batcher(eval_arr, args.batch_size, device, seed=1)

    model = build_model(device, seed=args.seed)
    nparam = sum(p.numel() for p in model.parameters())
    cw = get_weights(model)
    print(f"[victim] {nparam/1e6:.2f}M params, {len(cw)} compressible matrices, "
          f"{fp32_bytes(cw)/1e6:.1f} MB fp32 | {args.steps} steps x bs {args.batch_size} "
          f"= {args.steps*args.batch_size*meta['seqlen']/1e6:.0f}M tokens", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rows = []
    t0 = time.time(); run_loss = None
    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        x = tb.batch()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(input_ids=x, labels=x)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % args.eval_every == 0 or step == args.steps - 1:
            el = eval_loss(model, eb, device, max_batches=args.eval_batches)
            dt = time.time() - t0
            tps = (step + 1) * args.batch_size * meta["seqlen"] / max(dt, 1e-9)
            rows.append(dict(step=step, train_loss=round(run_loss, 4), eval_loss=round(el, 4),
                             lr=round(lr_at(step, args.lr, args.warmup, args.steps), 6),
                             elapsed_s=round(dt, 1)))
            print(f"  step {step:5d}/{args.steps}  train {run_loss:.4f}  eval {el:.4f}  "
                  f"lr {rows[-1]['lr']:.2e}  {tps/1e3:.0f}k tok/s  {dt:.0f}s", flush=True)

    final_eval = eval_loss(model, eb, device, max_batches=None)
    torch.save({"state_dict": model.state_dict(), "config": model.config.to_dict()},
               os.path.join(args.out, "victim.pt"))
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = dict(steps=args.steps, batch_size=args.batch_size, seqlen=meta["seqlen"],
                   tokens=args.steps * args.batch_size * meta["seqlen"], lr=args.lr,
                   warmup=args.warmup, weight_decay=args.weight_decay,
                   target_loss=round(final_eval, 4), final_train_loss=round(run_loss, 4),
                   fp32_bytes=fp32_bytes(cw), n_compressible=len(cw),
                   nparam=nparam, elapsed_s=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"[victim] DONE. final eval loss (target) = {final_eval:.4f} over {args.steps} steps "
          f"in {summary['elapsed_s']:.0f}s -> {args.out}", flush=True)

    # curves by default
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        st = [r["step"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(st, [r["train_loss"] for r in rows], label="train loss", color="#5ab1ef")
        ax.plot(st, [r["eval_loss"] for r in rows], label="eval loss", color="#f0a030", lw=2)
        ax.axhline(final_eval, ls="--", c="#7d3a3a", lw=1, label=f"final eval (target) {final_eval:.3f}")
        ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss")
        ax.set_title(f"11M SimpleStories victim — from-scratch ({args.steps} steps, undertrained)")
        ax.legend(); ax.grid(alpha=.15)
        fig.tight_layout(); fig.savefig("figures/llm_baseline_curve.png", dpi=130)
        print("[victim] wrote figures/llm_baseline_curve.png", flush=True)
    except Exception as e:
        print("[victim] plot skipped:", e, flush=True)


if __name__ == "__main__":
    main()

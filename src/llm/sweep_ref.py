"""Train the 11M victim with an LR sweep (like the ResNet baseline) and save intermediate
checkpoints. All LRs train concurrently in one process from the SAME init on the SAME batches
each step (a clean LR ablation; the small 11M model leaves plenty of GPU room). The lowest
final-eval-loss LR becomes the victim; its intermediate checkpoints are kept so we can pick the
target-loss / denominator point later.

    python -m src.llm.sweep_ref --lrs 1e-3 3e-3 6e-3 1e-2 --steps 3000 --batch-size 64
Smoke: python -m src.llm.sweep_ref --lrs 1e-3 3e-3 --steps 20 --eval-every 10 \
           --batch-size 32 --cache-dir data/ss_smoke --out runs/llm_sweep_smoke
"""
import argparse
import csv
import json
import os
import shutil
import time

import torch

from .data import Batcher, load_packed
from .model import build_model, eval_loss, fp32_bytes, get_weights
from .train_ref import lr_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 6e-3, 1e-2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--ckpt-fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cache-dir", default="data/simplestories")
    ap.add_argument("--out", default="runs/llm_victim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_arr, eval_arr, meta = load_packed(args.cache_dir)
    tb = Batcher(train_arr, args.batch_size, device, seed=args.seed)
    eb = Batcher(eval_arr, args.batch_size, device, seed=1)
    def _snap(f):  # land checkpoints on eval steps so their recorded eval_loss is exact
        if f >= 1.0:
            return args.steps - 1
        return max(0, min(args.steps - 1,
                          int(round(f * args.steps / args.eval_every)) * args.eval_every))
    ckpt_steps = sorted({_snap(f) for f in args.ckpt_fracs})

    K = len(args.lrs)
    models, opts = [], []
    for k in range(K):
        m = build_model(device, seed=args.seed)          # SAME init for every LR
        models.append(m)
        opts.append(torch.optim.AdamW(m.parameters(), lr=args.lrs[k], betas=(0.9, 0.95),
                                      weight_decay=args.weight_decay))
    cw = get_weights(models[0])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[sweep] {K} LRs {args.lrs} | 11M x{K}, {len(cw)} matrices, {fp32_bytes(cw)/1e6:.1f} MB "
          f"fp32 | {args.steps} steps x bs {args.batch_size} = "
          f"{args.steps*args.batch_size*meta['seqlen']/1e6:.0f}M tokens/LR | ckpt@{ckpt_steps}",
          flush=True)

    rows = []            # per (step, lr) eval rows
    run_loss = [None] * K
    t0 = time.time()
    for m in models:
        m.train()
    for step in range(args.steps):
        x = tb.batch()                                   # shared batch across LRs
        for k in range(K):
            for g in opts[k].param_groups:
                g["lr"] = lr_at(step, args.lrs[k], args.warmup, args.steps)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = models[k](input_ids=x, labels=x)
            opts[k].zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
            opts[k].step()
            l = out.loss.item()
            run_loss[k] = l if run_loss[k] is None else 0.98 * run_loss[k] + 0.02 * l
        do_eval = step % args.eval_every == 0 or step == args.steps - 1
        if do_eval:
            evs = [eval_loss(models[k], eb, device, max_batches=args.eval_batches) for k in range(K)]
            dt = time.time() - t0
            for k in range(K):
                rows.append(dict(step=step, lr=args.lrs[k], train_loss=round(run_loss[k], 4),
                                 eval_loss=round(evs[k], 4), elapsed_s=round(dt, 1)))
            tps = (step + 1) * args.batch_size * meta["seqlen"] * K / max(dt, 1e-9)
            best = min(range(K), key=lambda k: evs[k])
            print(f"  step {step:5d}/{args.steps}  " +
                  "  ".join(f"lr{args.lrs[k]:.0e}:{evs[k]:.3f}" for k in range(K)) +
                  f"  | best lr{args.lrs[best]:.0e}  {tps/1e3:.0f}k tok/s(all)  {dt:.0f}s", flush=True)
        if step in ckpt_steps:
            for k in range(K):
                d = os.path.join(args.out, f"lr_{args.lrs[k]:.0e}")
                os.makedirs(d, exist_ok=True)
                ev_here = next(r["eval_loss"] for r in reversed(rows) if r["lr"] == args.lrs[k])
                torch.save({"state_dict": models[k].state_dict(), "step": step,
                            "lr": args.lrs[k], "eval_loss": ev_here,
                            "config": models[k].config.to_dict()},
                           os.path.join(d, f"ckpt_{step}.pt"))

    # pick best LR by final full eval loss
    final = {k: eval_loss(models[k], eb, device, max_batches=None) for k in range(K)}
    best = min(range(K), key=lambda k: final[k])
    best_lr = args.lrs[best]
    print(f"[sweep] final eval: " + "  ".join(f"lr{args.lrs[k]:.0e}:{final[k]:.4f}" for k in range(K)) +
          f"  -> BEST lr{best_lr:.0e} = {final[best]:.4f}", flush=True)

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # promote the best LR: victim.pt + its intermediate checkpoints; record the checkpoint manifest
    best_dir = os.path.join(args.out, f"lr_{best_lr:.0e}")
    torch.save({"state_dict": models[best].state_dict(), "config": models[best].config.to_dict(),
                "lr": best_lr, "step": args.steps}, os.path.join(args.out, "victim.pt"))
    manifest = []
    for cs in ckpt_steps:
        p = os.path.join(best_dir, f"ckpt_{cs}.pt")
        if os.path.exists(p):
            d = torch.load(p, map_location="cpu", weights_only=False)
            manifest.append({"step": cs, "eval_loss": d["eval_loss"],
                             "path": os.path.relpath(p, args.out)})
    # prune the non-best LRs' checkpoints (keep their metrics rows)
    for k in range(K):
        if k != best:
            shutil.rmtree(os.path.join(args.out, f"lr_{args.lrs[k]:.0e}"), ignore_errors=True)

    summary = dict(lrs=args.lrs, best_lr=best_lr, final_eval=final[best],
                   final_eval_by_lr={f"{args.lrs[k]:.0e}": round(final[k], 4) for k in range(K)},
                   target_loss=round(final[best], 4), steps=args.steps, batch_size=args.batch_size,
                   seqlen=meta["seqlen"], tokens=args.steps * args.batch_size * meta["seqlen"],
                   warmup=args.warmup, weight_decay=args.weight_decay,
                   fp32_bytes=fp32_bytes(cw), n_compressible=len(cw),
                   checkpoints=manifest, elapsed_s=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(f"[sweep] DONE in {summary['elapsed_s']:.0f}s. victim=best lr {best_lr:.0e}, "
          f"target loss {final[best]:.4f}, {len(manifest)} intermediate ckpts -> {args.out}",
          flush=True)

    _plot(rows, args, best_lr, final[best])


def _plot(rows, args, best_lr, target):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        cmap = plt.cm.viridis
        for i, lr in enumerate(args.lrs):
            r = [x for x in rows if x["lr"] == lr]
            st = [x["step"] for x in r]
            c = cmap(i / max(1, len(args.lrs) - 1))
            lw = 2.6 if lr == best_lr else 1.3
            lab = f"lr {lr:.0e}" + ("  (victim)" if lr == best_lr else "")
            ax.plot(st, [x["eval_loss"] for x in r], color=c, lw=lw, label=lab)
        ax.axhline(target, ls="--", c="#7d3a3a", lw=1, label=f"target (victim final) {target:.3f}")
        ax.set_xlabel("step"); ax.set_ylabel("eval cross-entropy loss")
        ax.set_title("11M SimpleStories victim — LR sweep (from-scratch, single-pass)")
        ax.legend(fontsize=9); ax.grid(alpha=.15)
        fig.tight_layout(); fig.savefig("figures/llm_baseline_curve.png", dpi=130)
        print("[sweep] wrote figures/llm_baseline_curve.png", flush=True)
    except Exception as e:
        print("[sweep] plot skipped:", e, flush=True)


if __name__ == "__main__":
    main()

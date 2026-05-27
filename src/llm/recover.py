"""LLM recovery evaluator. Given a compressed reconstruction of the victim's 2D weights, build
the retrain init (victim's dense RMSNorm + reconstructed matrices), then retrain on SimpleStories
across a small LR sweep until the eval loss reaches the victim's target loss, capped at a compute
budget. Recovery cost = (first step any LR crosses) / denominator. 0 = reconstruction already at
target (free recovery); DNR = no LR crossed within budget.

All LRs in the sweep train concurrently from the same init; a model is frozen once it crosses,
so a recovery costs roughly min-over-LRs of the compute, not the sum.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from .data import Batcher, load_packed
from .model import build_config, eval_loss, set_weights
from .train_ref import lr_at
from transformers import LlamaForCausalLM


def _fresh_from(victim_sd, recon, device):
    cfg = build_config()
    m = LlamaForCausalLM(cfg)
    m.load_state_dict(victim_sd)            # dense RMSNorm etc. (stolen verbatim)
    if recon is not None:
        set_weights(m, recon)               # overwrite the 2D matrices with the reconstruction
    return m.to(device)


def evaluate_recovery(recon, victim_sd, *, target_loss, denom_steps, budget_steps, lr_grid,
                      warmup, eval_every, eval_batches, train_batcher, eval_batcher, device,
                      weight_decay=0.1, abort_frac=0.34, abort_margin=2.0, log_prefix=""):
    """Returns dict(recovered, recovery_fraction, best_lr, step0_loss, cross_step, curve)."""
    K = len(lr_grid)
    models = [_fresh_from(victim_sd, recon, device) for _ in range(K)]
    opts = [torch.optim.AdamW(m.parameters(), lr=lr_grid[k], betas=(0.9, 0.95),
                              weight_decay=weight_decay) for k, m in enumerate(models)]
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for m in models:
        m.train()

    # step-0 reconstruction quality (free-recovery check)
    step0 = eval_loss(models[0], eval_batcher, device, max_batches=eval_batches)
    cross = [None] * K
    if step0 <= target_loss:
        return dict(recovered=True, recovery_fraction=0.0, best_lr=None, step0_loss=round(step0, 4),
                    cross_step=0, curve=[(0, {f"{lr:.0e}": step0 for lr in lr_grid})])

    active = [True] * K
    curve = [(0, {f"{lr:.0e}": step0 for lr in lr_grid})]
    best_seen = step0
    abort_step = max(eval_every, int(abort_frac * budget_steps))
    aborted = False
    for step in range(budget_steps):
        x = train_batcher.batch()
        for k in range(K):
            if not active[k]:
                continue
            for g in opts[k].param_groups:
                g["lr"] = lr_at(step, lr_grid[k], warmup, budget_steps)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = models[k](input_ids=x, labels=x)
            opts[k].zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
            opts[k].step()
        if (step + 1) % eval_every == 0 or step == budget_steps - 1:
            snap = {}
            for k in range(K):
                if not active[k]:
                    snap[f"{lr_grid[k]:.0e}"] = None
                    continue
                el = eval_loss(models[k], eval_batcher, device, max_batches=eval_batches)
                snap[f"{lr_grid[k]:.0e}"] = round(el, 4)
                best_seen = min(best_seen, el)
                if el <= target_loss and cross[k] is None:
                    cross[k] = step + 1
                    active[k] = False
            curve.append((step + 1, snap))
            if not any(active):
                break
            # early-abort provably-destroyed inits: if past the abort point and the best loss any
            # LR has reached is still far above target, no LR will recover within budget.
            if step + 1 >= abort_step and best_seen > target_loss + abort_margin:
                aborted = True
                break

    crossed = [(k, cross[k]) for k in range(K) if cross[k] is not None]
    if crossed:
        bk, bs = min(crossed, key=lambda t: t[1])
        return dict(recovered=True, recovery_fraction=bs / denom_steps, best_lr=lr_grid[bk],
                    step0_loss=round(step0, 4), cross_step=bs, curve=curve)
    return dict(recovered=False, recovery_fraction=budget_steps / denom_steps, best_lr=None,
                step0_loss=round(step0, 4), cross_step=None, curve=curve)


# ----- seed-frontier runner (no API spend) -----------------------------------
def _seed_codes():
    """Hand-written seeds + the group-NF2 family discovered on ResNet (model-agnostic numpy)."""
    from ..autoresearch.seeds import SEEDS
    codes = {n: code for n, (fam, code) in SEEDS.items()}
    # pull the best ResNet-evolved scheme if the archive is present
    arch = "runs/autoresearch/archive.json"
    if os.path.exists(arch):
        e = json.load(open(arch))["entries"]
        rec = [x for x in e if x.get("recovered")]
        if rec:
            best = min(rec, key=lambda x: x["ratio"])
            codes[f"resnet_best_{best['name']}"] = best["code"]
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-dir", default="runs/llm_victim")
    ap.add_argument("--cache-dir", default="data/simplestories")
    ap.add_argument("--target-ckpt-step", type=int, default=None,
                    help="use this intermediate checkpoint's loss as the recovery target and its "
                         "step as the denominator (ResNet-style undertrained target). Default: "
                         "victim final loss / full step count.")
    ap.add_argument("--budget-fraction", type=float, default=0.05)
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="runs/llm_seedfrontier")
    args = ap.parse_args()

    from .candidate_llm import export_victim_weights, run_candidate, load_recon
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = json.load(open(os.path.join(args.victim_dir, "summary.json")))
    train_arr, eval_arr, meta = load_packed(args.cache_dir)
    tb = Batcher(train_arr, args.batch_size, device, seed=123)
    eb = Batcher(eval_arr, args.batch_size, device, seed=1)
    victim_sd = torch.load(os.path.join(args.victim_dir, "victim.pt"),
                           map_location="cpu", weights_only=False)["state_dict"]

    # The compressed weights always come from the FINAL victim. The recovery TARGET is an
    # undertrained checkpoint's loss (ResNet epoch-30 analog) to stay off the flattened tail of
    # the loss curve, with the denominator = that checkpoint's step count. Target is measured on
    # the SAME eval subset used for crossing detection (else a subset bias inflates DNRs).
    if args.target_ckpt_step is not None:
        c = next(c for c in summary["checkpoints"] if c["step"] == args.target_ckpt_step)
        tgt_sd = torch.load(os.path.join(args.victim_dir, c["path"]),
                            map_location="cpu", weights_only=False)["state_dict"]
        denom = args.target_ckpt_step
    else:
        tgt_sd = victim_sd
        denom = summary["steps"]
    budget_steps = int(round(args.budget_fraction * denom))
    tmodel = _fresh_from(tgt_sd, None, device)
    target = eval_loss(tmodel, eb, device, max_batches=args.eval_batches)
    del tmodel
    if device.type == "cuda":
        torch.cuda.empty_cache()

    wnpz = os.path.join(args.out, "victim_weights.npz")
    _, fp32_bytes = export_victim_weights(os.path.join(args.victim_dir, "victim.pt"), wnpz)
    print(f"[seedfrontier] target={target:.4f} (ckpt_step={args.target_ckpt_step}, "
          f"eval_batches={args.eval_batches}, victim_final={summary['target_loss']:.4f}) "
          f"denom={denom} budget={budget_steps} fp32={fp32_bytes/1e6:.1f}MB "
          f"lr_grid={args.lr_grid}", flush=True)

    results = []
    for name, code in _seed_codes().items():
        t0 = time.time()
        r = run_candidate(code, wnpz)
        if not r["ok"]:
            print(f"  {name:22} BUILD FAIL: {r['error'][:80]}", flush=True)
            results.append(dict(name=name, valid=False, ratio=1.0, recovered=False,
                                recovery_fraction=1.0, error=r["error"][:200]))
            continue
        recon = load_recon(r["recon_path"])
        ev = evaluate_recovery(recon, victim_sd, target_loss=target, denom_steps=denom,
                               budget_steps=budget_steps, lr_grid=args.lr_grid, warmup=args.warmup,
                               eval_every=args.eval_every, eval_batches=args.eval_batches,
                               train_batcher=tb, eval_batcher=eb, device=device)
        os.remove(r["recon_path"])
        tag = (f"REC@{ev['recovery_fraction']*100:.2f}%" if ev["recovered"] else "DNR")
        print(f"  {name:22} ratio={r['ratio']:.4f} step0={ev['step0_loss']:.3f} {tag} "
              f"(best_lr={ev['best_lr']}) {time.time()-t0:.0f}s", flush=True)
        results.append(dict(name=name, valid=True, ratio=r["ratio"], **{k: ev[k] for k in
                       ["recovered", "recovery_fraction", "best_lr", "step0_loss", "cross_step"]},
                       curve=ev["curve"]))

    json.dump(dict(target=target, target_ckpt_step=args.target_ckpt_step,
                   victim_final_loss=summary["target_loss"], eval_batches=args.eval_batches,
                   denom=denom, budget_steps=budget_steps, budget_fraction=args.budget_fraction,
                   fp32_bytes=fp32_bytes, lr_grid=args.lr_grid, results=results),
              open(os.path.join(args.out, "frontier.json"), "w"), indent=1)
    rec = [r for r in results if r.get("recovered")]
    print(f"[seedfrontier] DONE. {len(rec)}/{len(results)} recovered. "
          + ("best ratio %.4f" % min((r["ratio"] for r in rec), default=float('nan'))), flush=True)


if __name__ == "__main__":
    main()

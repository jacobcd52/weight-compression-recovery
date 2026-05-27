"""Diagnostic: overlay low-rank RECOVERY retraining curves on the baseline's FROM-SCRATCH curve.

The point (per user): retraining from a low-rank init should look like normal training, not
something pathological. Since a low-rank reconstruction of these (full-rank) weights has a loss
~as bad as random init, retraining it should track training-from-scratch and converge toward the
victim loss given enough steps. If instead it stalls far above the from-scratch curve, that would
signal a bug. Uses a wider LR sweep (incl. 3e-3) and a long budget so LR is not the limiter.

    python -m src.llm.lowrank_diag
"""
import csv
import json
import os

import torch

from .data import Batcher, load_packed
from .model import eval_loss
from .recover import _fresh_from, evaluate_recovery
from .candidate_llm import export_victim_weights, run_candidate, load_recon
from ..autoresearch.seeds import SEEDS

METHODS = ["seed_lowrank", "seed_lowrank_r8", "seed_lowrank_r2"]  # frac0.25, rank8, rank2


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = "runs/llm_seedfrontier"; os.makedirs(out, exist_ok=True)
    train, ev, meta = load_packed("data/simplestories")
    tb = Batcher(train, 64, dev, seed=123); eb = Batcher(ev, 64, dev, seed=1)
    summ = json.load(open("runs/llm_victim/summary.json"))
    victim_sd = torch.load("runs/llm_victim/victim.pt", map_location="cpu",
                           weights_only=False)["state_dict"]
    # victim final loss on the same 32-batch eval (for reference lines)
    vfull = eval_loss(_fresh_from(victim_sd, None, dev), eb, dev, max_batches=32)
    torch.cuda.empty_cache()

    # baseline from-scratch eval curve (best LR = 1e-3), from the sweep metrics
    rows = list(csv.DictReader(open("runs/llm_victim/metrics.csv")))
    base = [(int(r["step"]), float(r["eval_loss"])) for r in rows if abs(float(r["lr"]) - 1e-3) < 1e-9]
    base.sort()

    budget, lr_grid = 1200, [3e-4, 1e-3, 3e-3]
    wnpz = os.path.join(out, "victim_weights.npz")
    export_victim_weights("runs/llm_victim/victim.pt", wnpz)
    print(f"victim final(32b)={vfull:.4f}; baseline-from-scratch reaches "
          f"{base[-1][1]:.3f} by step {base[-1][0]}", flush=True)

    curves = {}
    for name in METHODS:
        r = run_candidate(SEEDS[name][1], wnpz)
        recon = load_recon(r["recon_path"]); os.remove(r["recon_path"])
        res = evaluate_recovery(recon, victim_sd, target_loss=vfull, denom_steps=summ["steps"],
                                budget_steps=budget, lr_grid=lr_grid, warmup=50, eval_every=50,
                                eval_batches=32, train_batcher=tb, eval_batcher=eb, device=dev,
                                abort_margin=1e9)
        xs, ys = [], []
        for step, snap in res["curve"]:
            vals = [v for v in snap.values() if v is not None]
            if vals:
                xs.append(step); ys.append(min(vals))
        curves[name] = dict(ratio=r["ratio"], xs=xs, ys=ys)
        print(f"  {name:16} ratio={r['ratio']:.4f} recon={ys[0]:.2f} -> end {ys[-1]:.3f} "
              f"(over {budget} steps, LR-swept {lr_grid})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot([s for s, _ in base], [l for _, l in base], "-", color="k", lw=2.6,
            label="BASELINE from-scratch (random init, lr 1e-3)")
    cmap = plt.cm.cool
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["xs"], c["ys"], "--o", ms=3, color=cmap(i / 2),
                label=f"{name.replace('seed_','')} retrain (r={c['ratio']:.3f})")
    ax.axhline(vfull, ls=":", c="#7d3a3a", lw=1.5, label=f"victim loss {vfull:.2f}")
    ax.set_xlabel("step (from-scratch step, or retraining step)")
    ax.set_ylabel("eval loss"); ax.set_title(
        "Low-rank recovery vs. training from scratch\n"
        "(low-rank init ~ as bad as random; retraining tracks from-scratch = sane, not a bug)")
    ax.legend(fontsize=9); ax.grid(alpha=.15)
    fig.tight_layout(); os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/llm_lowrank_vs_baseline.png", dpi=130); plt.close(fig)
    print("wrote figures/llm_lowrank_vs_baseline.png", flush=True)


if __name__ == "__main__":
    main()

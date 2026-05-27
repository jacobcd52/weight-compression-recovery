"""Capture and plot recovery RETRAINING curves for representative compression schemes — to tell a
bug (diverging / erratic / NaN loss) from a merely-bad method (smooth descent that plateaus above
target). Saves figures/llm_retrain_curves.png + runs/llm_seedfrontier/retrain_curves.json.

    python -m src.llm.retrain_curves
"""
import json
import os

import torch

from .data import Batcher, load_packed
from .model import build_config, eval_loss
from .recover import _fresh_from, evaluate_recovery
from .candidate_llm import export_victim_weights, run_candidate, load_recon
from ..autoresearch.seeds import SEEDS

# good (recover) + bad (DNR) methods; the bad ones are what we want to inspect for bugs
METHODS = ["seed_quant4", "seed_kmeans4", "seed_quant2", "seed_magsparse",
           "seed_lowrank", "seed_lowrank_r8", "seed_lowrank_r2"]
GOOD = {"seed_quant4", "seed_kmeans4"}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = "runs/llm_seedfrontier"; os.makedirs(out, exist_ok=True)
    train, ev, meta = load_packed("data/simplestories")
    tb = Batcher(train, 64, dev, seed=123); eb = Batcher(ev, 64, dev, seed=1)
    summ = json.load(open("runs/llm_victim/summary.json"))
    victim_sd = torch.load("runs/llm_victim/victim.pt", map_location="cpu",
                           weights_only=False)["state_dict"]
    c2250 = next(c for c in summ["checkpoints"] if c["step"] == 2250)
    tgt_sd = torch.load(os.path.join("runs/llm_victim", c2250["path"]),
                        map_location="cpu", weights_only=False)["state_dict"]
    tmodel = _fresh_from(tgt_sd, None, dev)
    target = eval_loss(tmodel, eb, dev, max_batches=32); del tmodel
    torch.cuda.empty_cache()
    denom = 2250
    wnpz = os.path.join(out, "victim_weights.npz")
    export_victim_weights("runs/llm_victim/victim.pt", wnpz)
    print(f"target(2250,32b)={target:.4f} denom={denom}", flush=True)

    budget, lr_grid = 250, [1e-4, 3e-4, 1e-3]
    curves = {}
    for name in METHODS:
        r = run_candidate(SEEDS[name][1], wnpz)
        recon = load_recon(r["recon_path"]); os.remove(r["recon_path"])
        ev_res = evaluate_recovery(recon, victim_sd, target_loss=target, denom_steps=denom,
                                   budget_steps=budget, lr_grid=lr_grid, warmup=20, eval_every=10,
                                   eval_batches=32, train_batcher=tb, eval_batcher=eb, device=dev,
                                   abort_margin=1e9)   # never abort: we want the full curve
        # min-over-LRs loss per step
        xs, ys = [], []
        for step, snap in ev_res["curve"]:
            vals = [v for v in snap.values() if v is not None]
            if vals:
                xs.append(step); ys.append(min(vals))
        curves[name] = dict(ratio=r["ratio"], recovered=ev_res["recovered"],
                            recovery_fraction=ev_res["recovery_fraction"], xs=xs, ys=ys)
        tag = f"REC@{ev_res['recovery_fraction']*100:.1f}%" if ev_res["recovered"] else "DNR"
        print(f"  {name:18} ratio={r['ratio']:.4f} recon={ev_res['step0_loss']:.2f} "
              f"-> end {ys[-1]:.3f}  {tag}", flush=True)
    json.dump(dict(target=target, denom=denom, budget=budget, lr_grid=lr_grid, curves=curves),
              open(os.path.join(out, "retrain_curves.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = plt.cm.tab10
    for i, (name, c) in enumerate(curves.items()):
        good = name in GOOD
        ax.plot(c["xs"], c["ys"], "-" if good else "--", color=cmap(i % 10), lw=2.2 if good else 1.7,
                marker="o", ms=3,
                label=f"{name.replace('seed_','')} (r={c['ratio']:.3f}, "
                      f"{'REC '+format(c['recovery_fraction']*100,'.1f')+'%' if c['recovered'] else 'DNR'})")
    ax.axhline(target, ls=":", c="k", lw=1.6, label=f"target {target:.2f}")
    ax.set_xlabel("retraining step"); ax.set_ylabel("eval loss (min over LR sweep)")
    ax.set_title("Recovery retraining curves — bug check\n(smooth monotone descent = bad method, not a bug)")
    ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig("figures/llm_retrain_curves.png", dpi=130); plt.close(fig)
    print("wrote figures/llm_retrain_curves.png", flush=True)


if __name__ == "__main__":
    main()

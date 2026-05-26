"""Run the whole compression sweep through the vmap ensemble evaluator.

For each (technique, knob) it builds the compressed init from the GroupNorm baseline, then
trains all (init x LR-grid) configs vectorized in model-chunks, recording each model's
recovery cost = min over its LR-grid. Writes plot-compatible summaries to runs/<name>/.

Usage:
  python -m src.sweep_ensemble --config configs/retrain.yaml --sweep configs/sweep_promising.yaml \
      --baseline-dir runs/baseline_gn --chunk-models 8
"""
import argparse
import json
import os

import torch

from .compress import compress, reconstruct, total_bytes, overhead_bytes
from .data import get_loaders
from .ensemble import run_ensemble
from .sweep import enumerate_runs
from .utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain.yaml")
    ap.add_argument("--sweep", default="configs/sweep_promising.yaml")
    ap.add_argument("--baseline-dir", default="runs/baseline_gn")
    ap.add_argument("--chunk-models", type=int, default=8,
                    help="models per vmap chunk (x len(lr_grid) configs run together)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sweep_cfg = load_config(args.sweep)
    seed = cfg.get("seed", 42)
    norm = "group"
    lr_grid = sorted(cfg.get("lr_grid", [cfg["lr"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = json.load(open(os.path.join(args.baseline_dir, "summary.json")))
    baseline_loss = base["baseline_test_loss"]
    baseline_steps = base["baseline_steps"]
    budget_steps = int(round(cfg["budget_fraction"] * baseline_steps))
    warmup_steps = int(cfg.get("warmup_steps", cfg.get("warmup_epochs", 5) * base["steps_per_epoch"]))
    eval_every = int(cfg.get("eval_every_steps", 20))

    ckpt = torch.load(os.path.join(args.baseline_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    train_loader, test_loader, _ = get_loaders(batch_size=cfg["batch_size"],
                                              num_workers=cfg.get("num_workers", 4))

    # 1) build compressed reps + reconstructed inits for every (technique, knob)
    models = []   # list of dict(name, technique, knob, ratio, amort_ratio, cbytes, init)
    for tech, knob, ks in enumerate_runs(sweep_cfg):
        need_dev = tech in ("snip", "fisher_prune", "aqlm")
        comp = compress(sd, tech, knob, train_loader=(train_loader if need_dev else None),
                        device=device, seed=seed, norm=norm)
        cbytes, bbytes = total_bytes(comp, sd)
        amort = (cbytes - overhead_bytes(comp)) / bbytes
        models.append({"name": f"{tech}_{ks}", "technique": tech, "knob": knob,
                       "ratio": cbytes / bbytes, "amort_ratio": amort,
                       "cbytes": cbytes, "init": reconstruct(comp)})
        print(f"built {tech}_{ks:<10} ratio={cbytes/bbytes:.4f}", flush=True)

    # 2) run the ensemble in model-chunks (each chunk: G models x len(lr_grid) configs)
    K = len(lr_grid)
    G = max(1, args.chunk_models)
    per_model_results = {}
    for c0 in range(0, len(models), G):
        chunk = models[c0:c0 + G]
        inits, lrs, owner = [], [], []
        for mi, mdl in enumerate(chunk):
            for lr in lr_grid:
                inits.append(mdl["init"]); lrs.append(lr); owner.append(mi)
        print(f"[ensemble] chunk {c0//G}: {len(chunk)} models x {K} LRs = {len(inits)} configs",
              flush=True)
        res = run_ensemble(inits, lrs, baseline_loss=baseline_loss,
                           baseline_steps=baseline_steps, budget_steps=budget_steps,
                           warmup_steps=warmup_steps, eval_every=eval_every,
                           train_loader=train_loader, test_loader=test_loader,
                           device=device, norm=norm, num_classes=10,
                           weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]),
                           log_prefix=f" c{c0//G}")
        for mi, mdl in enumerate(chunk):
            rs = [res[j] for j in range(len(res)) if owner[j] == mi]
            per_model_results[mdl["name"]] = rs

    # 3) write plot-compatible per-model summaries
    for mdl in models:
        rs = per_model_results[mdl["name"]]
        recovered = any(r["recovered"] for r in rs)
        rec = [r for r in rs if r["recovered"]]
        if rec:
            best = min(rec, key=lambda r: r["recovery_fraction"])
            rf, best_lr, rec_steps = best["recovery_fraction"], best["lr"], best["recovery_steps"]
        else:
            rf, best_lr, rec_steps = cfg["budget_fraction"], min(rs, key=lambda r: r["best_loss"])["lr"], None
        run_dir = os.path.join("runs", mdl["name"])
        os.makedirs(run_dir, exist_ok=True)
        summary = {
            "run_name": mdl["name"], "technique": mdl["technique"], "knob": mdl["knob"],
            "mode": "plain", "did_distill": False, "norm": norm, "via": "ensemble",
            "compressed_bytes": mdl["cbytes"], "compression_ratio": mdl["ratio"],
            "amortized_ratio": mdl["amort_ratio"],
            "recovered": recovered, "recovery_steps": rec_steps, "recovery_fraction": rf,
            "best_lr": best_lr,
            "lr_results": [{"lr": r["lr"], "recovered": r["recovered"],
                            "recovery_fraction": r["recovery_fraction"],
                            "best_loss": r["best_loss"]} for r in rs],
            "final_test_loss": min(r["best_loss"] for r in rs),
            "final_test_acc": 0.0,
            "baseline_test_loss": baseline_loss, "baseline_test_acc": base["baseline_test_acc"],
            "baseline_steps": baseline_steps, "budget_steps": budget_steps, "smoke": False,
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    nrec = sum(any(r["recovered"] for r in per_model_results[m["name"]]) for m in models)
    print(f"[ensemble] DONE: {len(models)} models, {nrec} recovered", flush=True)


if __name__ == "__main__":
    main()

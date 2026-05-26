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
    ap.add_argument("--only", default=None,
                    help="comma-separated run names (technique_knob) to run a subset")
    ap.add_argument("--target-epoch", type=int, default=None,
                    help="recovery target = baseline test loss at this epoch (from its curve); "
                         "the denominator for recovery_fraction is then target-epoch x steps/epoch")
    ap.add_argument("--budget-fraction", type=float, default=None,
                    help="max retrain steps as a fraction of the denominator (overrides cfg)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    cfg = load_config(args.config)
    sweep_cfg = load_config(args.sweep)
    seed = cfg.get("seed", 42)
    norm = "group"
    lr_grid = sorted(cfg.get("lr_grid", [cfg["lr"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = json.load(open(os.path.join(args.baseline_dir, "summary.json")))
    spe = base["steps_per_epoch"]
    if args.target_epoch is not None:
        # target = baseline test loss at target_epoch (from its curve); denominator = that epoch's steps
        import csv as _csv
        mrows = list(_csv.DictReader(open(os.path.join(args.baseline_dir, "metrics.csv"))))
        baseline_loss = float(mrows[args.target_epoch - 1]["test_loss"])
        baseline_steps = args.target_epoch * spe
        print(f"[target] epoch {args.target_epoch}: loss={baseline_loss:.4f}, denom={baseline_steps} steps",
              flush=True)
    else:
        baseline_loss = base["baseline_test_loss"]
        baseline_steps = base["baseline_steps"]
    budget_fraction = args.budget_fraction if args.budget_fraction is not None else cfg["budget_fraction"]
    budget_steps = int(round(budget_fraction * baseline_steps))
    warmup_steps = int(cfg.get("warmup_steps", 5 * spe))
    eval_every = int(cfg.get("eval_every_steps", 20))

    ckpt = torch.load(os.path.join(args.baseline_dir, "best.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    train_loader, test_loader, _ = get_loaders(batch_size=cfg["batch_size"],
                                              num_workers=cfg.get("num_workers", 4))

    # 1) build compressed reps + reconstructed inits for every (technique, knob)
    models = []   # list of dict(name, technique, knob, ratio, amort_ratio, cbytes, init)
    for tech, knob, ks in enumerate_runs(sweep_cfg):
        if only is not None and f"{tech}_{ks}" not in only:
            continue
        need_dev = tech in ("snip", "fisher_prune", "aqlm")
        comp = compress(sd, tech, knob, train_loader=(train_loader if need_dev else None),
                        device=device, seed=seed, norm=norm)
        cbytes, bbytes = total_bytes(comp, sd)
        amort = (cbytes - overhead_bytes(comp)) / bbytes
        models.append({"name": f"{tech}_{ks}", "technique": tech, "knob": knob,
                       "ratio": cbytes / bbytes, "amort_ratio": amort,
                       "cbytes": cbytes, "init": reconstruct(comp)})
        print(f"built {tech}_{ks:<10} ratio={cbytes/bbytes:.4f}", flush=True)

    # 2) WIDE-BATCH: chunk G models, vmap all their LRs together (G*K configs) to saturate the
    # GPU. stop_on_any=False -> each config is masked (frozen) when it crosses; the chunk runs to
    # budget (small, so cheap). Per-method recovery = min over its LRs; per-method curve = the
    # min-loss / max-acc envelope over its LR indices.
    from .utils import CSVLogger
    K = len(lr_grid)
    G = max(1, args.chunk_models)
    per_model_results, per_model_curves = {}, {}
    done = 0
    for c0 in range(0, len(models), G):
        chunk = models[c0:c0 + G]
        inits, lrs, owner = [], [], []
        for mi, mdl in enumerate(chunk):
            for lr in lr_grid:
                inits.append(mdl["init"]); lrs.append(lr); owner.append(mi)
        res, curve = run_ensemble(inits, lrs, baseline_loss=baseline_loss,
                           baseline_steps=baseline_steps, budget_steps=budget_steps,
                           warmup_steps=warmup_steps, eval_every=eval_every,
                           train_loader=train_loader, test_loader=test_loader,
                           device=device, norm=norm, num_classes=10,
                           weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]),
                           stop_on_any=False, log_prefix=f" c{c0//G}")
        for mi, mdl in enumerate(chunk):
            idxs = [j for j in range(len(owner)) if owner[j] == mi]
            per_model_results[mdl["name"]] = [res[j] for j in idxs]
            mc = [(st, min(losses[j] for j in idxs), max(accs[j] for j in idxs))
                  for (st, losses, accs) in curve]
            per_model_curves[mdl["name"]] = mc
            run_dir = os.path.join("runs", mdl["name"]); os.makedirs(run_dir, exist_ok=True)
            clog = CSVLogger(os.path.join(run_dir, "metrics.csv"),
                             ["epoch", "step", "test_acc", "test_loss", "lr"])
            for (st, lo, ac) in mc:
                clog.log(epoch=st // spe, step=st, test_acc=ac, test_loss=lo, lr=0)
            done += 1
            rec = [r for r in per_model_results[mdl["name"]] if r["recovered"]]
            tag = (f"REC@{min(r['recovery_fraction'] for r in rec)*100:.2f}%" if rec else "DNR")
            print(f"[ensemble] {done}/{len(models)} {mdl['name']:<22} ratio={mdl['ratio']:.4f} {tag}",
                  flush=True)

    # 3) write plot-compatible per-model summaries
    for mdl in models:
        rs = per_model_results[mdl["name"]]
        recovered = any(r["recovered"] for r in rs)
        rec = [r for r in rs if r["recovered"]]
        if rec:
            best = min(rec, key=lambda r: r["recovery_fraction"])
            rf, best_lr, rec_steps = best["recovery_fraction"], best["lr"], best["recovery_steps"]
        else:
            rf, best_lr, rec_steps = budget_fraction, min(rs, key=lambda r: r["best_loss"])["lr"], None
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
            "init_loss": per_model_curves[mdl["name"]][0][1],
            "baseline_test_loss": baseline_loss, "baseline_test_acc": base["baseline_test_acc"],
            "baseline_steps": baseline_steps, "budget_steps": budget_steps, "smoke": False,
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    nrec = sum(any(r["recovered"] for r in per_model_results[m["name"]]) for m in models)
    print(f"[ensemble] DONE: {len(models)} models, {nrec} recovered", flush=True)


if __name__ == "__main__":
    main()

"""Auto-research controller: evolve per-tensor compression schemes against the vmap-ensemble
fitness. Loop: sample parents from the archive -> LLM proposes a new scheme -> sandbox-build +
measure honest bytes -> evaluate a wave in one wide vmap batch -> update the Pareto archive.

Seed-only (no LLM, no spend):  python -m src.autoresearch.run --proposals 0
Pilot:                         python -m src.autoresearch.run --proposals 150 --wave 6
"""
import argparse
import csv as _csv
import json
import os
import random
import time

import torch

from ..data import get_loaders
from ..ensemble import run_ensemble
from ..utils import load_config
from .archive import Archive
from .candidate import run_candidate
from .propose import build_prompt, propose
from .seeds import SEEDS


def evaluate_wave(cands, baseline_path, *, baseline_loss, baseline_steps, budget_steps,
                  warmup_steps, lr_grid, train_loader, test_loader, device, weight_decay,
                  betas, chunk=32):
    """cands: list of {name, family, code}. Builds each init (sandboxed), then scores all valid
    candidates' (init x LR) configs in wide vmap chunks. Returns cands annotated with metrics."""
    built = []
    for c in cands:
        r = run_candidate(c["code"], baseline_path)
        c = dict(c)
        if not r["ok"]:
            c.update(recovered=False, recovery_fraction=baseline_steps and 1.0, ratio=1.0,
                     valid=False, error=r["error"]); built.append(c); continue
        c.update(ratio=r["ratio"], init_path=r["init_path"], valid=True)
        built.append(c)
    valid = [c for c in built if c.get("valid")]

    K = len(lr_grid)
    G = max(1, chunk // K)                       # candidates per chunk
    for c0 in range(0, len(valid), G):
        grp = valid[c0:c0 + G]
        inits, lrs, owner = [], [], []
        for mi, c in enumerate(grp):
            sd = torch.load(c["init_path"], map_location="cpu", weights_only=False)["init"]
            for lr in lr_grid:
                inits.append(sd); lrs.append(lr); owner.append(mi)
        res, _ = run_ensemble(inits, lrs, baseline_loss=baseline_loss,
                              baseline_steps=baseline_steps, budget_steps=budget_steps,
                              warmup_steps=warmup_steps, eval_every=20,
                              train_loader=train_loader, test_loader=test_loader,
                              device=device, weight_decay=weight_decay, betas=betas,
                              stop_on_any=False, log_prefix=" wave")
        for mi, c in enumerate(grp):
            rs = [res[j] for j in range(len(owner)) if owner[j] == mi]
            rec = [r for r in rs if r["recovered"]]
            if rec:
                b = min(rec, key=lambda r: r["recovery_fraction"])
                c.update(recovered=True, recovery_fraction=b["recovery_fraction"], best_lr=b["lr"])
            else:
                c.update(recovered=False, recovery_fraction=budget_steps / baseline_steps,
                         best_lr=min(rs, key=lambda r: r["best_loss"])["lr"])
        for c in grp:
            if os.path.exists(c.get("init_path", "")):
                os.remove(c["init_path"])
    return built


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/retrain_gn.yaml")
    ap.add_argument("--baseline-dir", default="runs/baseline_gn")
    ap.add_argument("--target-epoch", type=int, default=30)
    ap.add_argument("--budget-fraction", type=float, default=0.05)
    ap.add_argument("--proposals", type=int, default=0, help="LLM proposals to make (0 = seeds only)")
    ap.add_argument("--wave", type=int, default=6, help="proposals built + evaluated per wave")
    ap.add_argument("--backend", default="claude_p", choices=["claude_p", "api"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--archive", default="runs/autoresearch/archive.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = json.load(open(os.path.join(args.baseline_dir, "summary.json")))
    spe = base["steps_per_epoch"]
    mrows = list(_csv.DictReader(open(os.path.join(args.baseline_dir, "metrics.csv"))))
    target_loss = float(mrows[args.target_epoch - 1]["test_loss"])
    baseline_steps = args.target_epoch * spe
    budget_steps = int(round(args.budget_fraction * baseline_steps))
    warmup_steps = int(cfg.get("warmup_steps", 100))
    lr_grid = sorted(cfg.get("lr_grid", [cfg["lr"]]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_path = os.path.join(args.baseline_dir, "best.pt")
    train_loader, test_loader, _ = get_loaders(batch_size=cfg["batch_size"],
                                              num_workers=cfg.get("num_workers", 4))
    ev = dict(baseline_loss=target_loss, baseline_steps=baseline_steps, budget_steps=budget_steps,
              warmup_steps=warmup_steps, lr_grid=lr_grid, train_loader=train_loader,
              test_loader=test_loader, device=device, weight_decay=cfg["weight_decay"],
              betas=tuple(cfg["betas"]))
    print(f"[autoresearch] target_loss={target_loss:.4f} denom={baseline_steps} budget={budget_steps} "
          f"lr_grid={lr_grid} backend={args.backend}", flush=True)

    arch = Archive(args.archive, budget_fraction=args.budget_fraction).load()

    # seed the archive (evaluate the hand-written seeds once)
    if not arch.entries:
        seeds = [{"name": n, "family": fam, "code": code} for n, (fam, code) in SEEDS.items()]
        for c in evaluate_wave(seeds, baseline_path, **ev):
            arch.add({"name": c["name"], "code": c["code"], "family": c["family"],
                      "ratio": c["ratio"], "recovered": c.get("recovered", False),
                      "recovery_fraction": c["recovery_fraction"],
                      "best_lr": c.get("best_lr"), "gen": 0, "parents": []})
            tag = f"REC@{c['recovery_fraction']*100:.2f}%" if c.get("recovered") else "DNR"
            print(f"[seed] {c['name']:18} ratio={c.get('ratio',1):.4f} {tag} "
                  f"{'' if c.get('valid', True) else 'INVALID:'+str(c.get('error',''))[:80]}",
                  flush=True)
        arch.save()

    # search loop
    rng = random.Random(0)
    made = 0
    while made < args.proposals:
        wave = []
        while len(wave) < args.wave and made < args.proposals:
            parents = arch.sample_parents(k=3, rng=rng)
            best_ratio = min((e["ratio"] for e in arch.entries if e.get("recovered")), default=None)
            prompt = build_prompt(parents, target_loss, args.budget_fraction * 100, best_ratio)
            code, err = propose(prompt, backend=args.backend, model=args.model)
            made += 1
            if code:
                wave.append({"name": f"cand_{made}", "family": "evolved", "code": code,
                             "parents": [p["name"] for p in parents]})
            else:
                print(f"[propose {made}] failed: {err}", flush=True)
        if not wave:
            continue
        for c in evaluate_wave(wave, baseline_path, **ev):
            arch.add({"name": c["name"], "code": c["code"], "family": c["family"],
                      "ratio": c.get("ratio", 1.0), "recovered": c.get("recovered", False),
                      "recovery_fraction": c["recovery_fraction"], "best_lr": c.get("best_lr"),
                      "gen": 1, "parents": c.get("parents", []), "valid": c.get("valid", False),
                      "error": c.get("error")})
            tag = f"REC@{c['recovery_fraction']*100:.2f}%" if c.get("recovered") else "DNR"
            print(f"[cand {made}] {c['name']:10} ratio={c.get('ratio',1):.4f} {tag} "
                  f"{'' if c.get('valid') else 'INVALID'}", flush=True)
        arch.save()

    front = arch.pareto_front()
    print(f"\n[autoresearch] DONE. {len(arch.entries)} evaluated, frontier {len(front)} points:")
    for e in front:
        print(f"  {e['name']:14} ratio={e['ratio']:.4f} recovery@{e['recovery_fraction']*100:.2f}%")


if __name__ == "__main__":
    main()

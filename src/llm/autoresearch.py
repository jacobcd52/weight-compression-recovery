"""LLM auto-research: evolve weight-compression schemes against the 11M-victim recovery metric.

Diversity is maintained three ways (per user): (1) specialized proposer roles — scalar-quant,
vector-quant, entropy/structure, hybrid — each with a focused prompt; (2) a MAP-Elites archive
binned by compression ratio; (3) a mix of proposer models (Opus for the hard vector-quant role,
Sonnet for the cheaper breadth roles). Fitness = recovery cost from src/llm/recover, ratio from
the honest zlib byte count. Spend is tracked and capped.

    python -m src.llm.autoresearch --proposals 120 --wave 6 --max-spend 20
"""
import argparse
import json
import os
import random
import time

import torch

from .data import Batcher, load_packed
from .model import eval_loss
from .recover import _fresh_from, evaluate_recovery, _seed_codes
from .candidate_llm import export_victim_weights, run_candidate, load_recon
from ..autoresearch.archive import Archive
from ..autoresearch.propose import parse_code, propose

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"

# rough API price estimate ($ per call, ~2k in / ~1.2k out tokens)
COST = {SONNET: 0.02, OPUS: 0.11}

LLM_SPEC = """You are inventing a per-tensor WEIGHT-COMPRESSION scheme for stealing a trained language
model cheaply. Write ONE self-contained Python snippet using ONLY numpy (as np) and math:

    KNOBS = {{ ... }}                      # small scalar defaults only
    def compress_tensor(w, **knobs):       # w: float32 2D weight matrix (shape preserved by caller)
        return payload                     # picklable dict of small numpy arrays / scalars
    def reconstruct_tensor(payload, shape):
        return np.ndarray(shape, float32)  # the dense reconstruction

The compressed size is measured OBJECTIVELY as len(zlib.compress(pickle(payload))) — you cannot
fake it; lower-entropy, smaller payloads win. After decompression the model is briefly RETRAINED on
the training data, so your reconstruction must be a good enough STARTING POINT that it recovers the
original loss in as little retraining as possible. The score is: lowest compression ratio that
still RECOVERS within the retrain budget.

Hard rules: deterministic; pure numpy+math; NO file/network/OS access, NO eval/exec/import beyond
numpy & math, NO loading external data, NO giant constant arrays.

WHAT WE LEARNED ON THIS MODEL (use it):
- The matrices are the attention (q/k/v/o) and MLP (gate/up/down) projections plus the tied
  token embedding. They are NEAR-FULL-RANK and DENSE.
- => LOW-RANK (SVD) and SPARSITY are BAD here: they destroy the model (loss worse than random) and
  need almost a full retrain. DO NOT propose plain low-rank or magnitude pruning.
- => QUANTIZATION is the family that works. The best recovering scheme so far is GROUP-WISE 2-bit
  with a NormalFloat codebook at compression ratio ~0.05.
- Promising directions: smaller group sizes with cheaply-coded (delta/quantized) scales; better
  codebooks (NormalFloat / learned / per-group); MIXED precision (a few outlier weights/columns in
  higher precision, the rest very low bit); ADDITIVE / MULTI-CODEBOOK VECTOR QUANTIZATION (encode
  sub-vectors as a SUM of entries from several learned codebooks — AQLM style); entropy-friendly
  index coding so zlib shrinks the payload further.

GOAL: beat ratio {best_ratio} while STILL RECOVERING. Push the ratio as low as you can.
"""

SPECIALIZATIONS = [
    {"name": "scalar", "model": SONNET, "focus":
     "Specialize in SCALAR quantization: group-wise / per-channel low-bit (1-3 bit) quant with "
     "smart codebooks (NormalFloat, per-group learned levels), cheaply-stored scales (delta-coded, "
     "8-bit-quantized), and outlier/mixed-precision handling. Push bits-per-weight down."},
    {"name": "vq", "model": OPUS, "focus":
     "Specialize in VECTOR / MULTI-CODEBOOK quantization (AQLM / 'Aggressive Compression Enables "
     "LLM Weight Theft' style). Split each matrix's rows into sub-vectors and encode each as a SUM "
     "of entries from M learned codebooks (residual/additive VQ), or product quantization. This is "
     "the strongest known method — implement it CAREFULLY and CORRECTLY (good codebook learning, "
     "right index bit-width, tiny codebooks amortized over the matrix). Make it actually recover."},
    {"name": "entropy", "model": SONNET, "focus":
     "Specialize in making the quantized representation LOW-ENTROPY so the objective zlib byte count "
     "shrinks: cluster values, delta/zigzag-code indices and scales, exploit the bell-shaped weight "
     "distribution, pack bits tightly. Combine with a solid quantizer."},
    {"name": "hybrid", "model": SONNET, "focus":
     "COMBINE / STACK the strongest ideas from the PARENT schemes below into one scheme (e.g. "
     "group-wise scaling + a NormalFloat or vector-quantized codebook + entropy-friendly index "
     "coding). Aim to inherit the best of each parent and beat them both."},
]


def build_llm_prompt(parents, spec, best_ratio):
    head = LLM_SPEC.format(best_ratio=f"{best_ratio:.4f}" if best_ratio else "0.05")
    parts = [head, "\nYOUR SPECIALIZATION: " + spec["focus"]]
    if parents:
        parts.append("\nParent schemes to mutate / combine (name | ratio | recovery cost):")
        for p in parents:
            rc = (f"{p['recovery_fraction']*100:.1f}%" if p.get("recovered") else "DID-NOT-RECOVER")
            parts.append(f"\n# === {p['name']}  ratio={p['ratio']:.4f}  recovery={rc} ===\n"
                         + p["code"].strip())
    parts.append("\nReturn ONLY one ```python code block with KNOBS, compress_tensor, "
                 "reconstruct_tensor. Make it correct and aim for a LOWER ratio that still recovers.")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-dir", default="runs/llm_victim")
    ap.add_argument("--cache-dir", default="data/simplestories")
    ap.add_argument("--target-ckpt-step", type=int, default=2250)
    ap.add_argument("--budget-fraction", type=float, default=0.15)   # smaller than frontier, for speed
    ap.add_argument("--lr-grid", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--proposals", type=int, default=120)
    ap.add_argument("--wave", type=int, default=6)
    ap.add_argument("--max-spend", type=float, default=20.0)
    ap.add_argument("--backend", default="api")
    ap.add_argument("--archive", default="runs/llm_autoresearch/archive.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.archive), exist_ok=True)
    summ = json.load(open(os.path.join(args.victim_dir, "summary.json")))
    train_arr, eval_arr, meta = load_packed(args.cache_dir)
    tb = Batcher(train_arr, args.batch_size, device, seed=123)
    eb = Batcher(eval_arr, args.batch_size, device, seed=1)
    victim_sd = torch.load(os.path.join(args.victim_dir, "victim.pt"),
                           map_location="cpu", weights_only=False)["state_dict"]
    c = next(c for c in summ["checkpoints"] if c["step"] == args.target_ckpt_step)
    tgt_sd = torch.load(os.path.join(args.victim_dir, c["path"]),
                        map_location="cpu", weights_only=False)["state_dict"]
    target = eval_loss(_fresh_from(tgt_sd, None, device), eb, device, max_batches=args.eval_batches)
    denom = args.target_ckpt_step
    budget_steps = int(round(args.budget_fraction * denom))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    wnpz = os.path.join(os.path.dirname(args.archive), "victim_weights.npz")
    _, fp32_bytes = export_victim_weights(os.path.join(args.victim_dir, "victim.pt"), wnpz)
    ev = dict(target_loss=target, denom_steps=denom, budget_steps=budget_steps,
              lr_grid=sorted(args.lr_grid), warmup=args.warmup, eval_every=args.eval_every,
              eval_batches=args.eval_batches, train_batcher=tb, eval_batcher=eb, device=device)
    print(f"[llm-autoresearch] target={target:.4f} denom={denom} budget={budget_steps} "
          f"({args.budget_fraction*100:.0f}%) lr_grid={sorted(args.lr_grid)} "
          f"max_spend=${args.max_spend} proposals<={args.proposals}", flush=True)

    arch = Archive(args.archive, budget_fraction=args.budget_fraction).load()

    def score_and_add(name, code, family, gen, parents):
        r = run_candidate(code, wnpz)
        if not r["ok"]:
            arch.add({"name": name, "code": code, "family": family, "ratio": 1.0,
                      "recovered": False, "recovery_fraction": 1.0, "gen": gen,
                      "parents": parents, "valid": False, "error": r["error"][:200]})
            return f"INVALID ({r['error'][:60]})"
        recon = load_recon(r["recon_path"])
        res = evaluate_recovery(recon, victim_sd, **ev)
        os.remove(r["recon_path"])
        arch.add({"name": name, "code": code, "family": family, "ratio": r["ratio"],
                  "recovered": res["recovered"], "recovery_fraction": res["recovery_fraction"],
                  "step0_loss": res["step0_loss"], "best_lr": res["best_lr"], "gen": gen,
                  "parents": parents, "valid": True})
        return (f"ratio={r['ratio']:.4f} recon={res['step0_loss']:.2f} " +
                (f"REC@{res['recovery_fraction']*100:.2f}%" if res["recovered"] else "DNR"))

    if not arch.entries:
        for name, code in _seed_codes().items():
            fam = "vq" if "vq" in name else ("evolved" if name.startswith("resnet") else "seed")
            print(f"[seed] {name:22} {score_and_add(name, code, fam, 0, [])}", flush=True)
        arch.save()

    rng = random.Random(0)
    made = 0; spend = 0.0; si = 0
    while made < args.proposals and spend < args.max_spend:
        wave = []
        while len(wave) < args.wave and made < args.proposals and spend < args.max_spend:
            spec = SPECIALIZATIONS[si % len(SPECIALIZATIONS)]; si += 1
            parents = arch.sample_parents(k=2, rng=rng)
            best_ratio = min((e["ratio"] for e in arch.entries if e.get("recovered")), default=None)
            code, err = propose(build_llm_prompt(parents, spec, best_ratio),
                                backend=args.backend, model=spec["model"])
            made += 1; spend += COST.get(spec["model"], 0.05)
            if code:
                wave.append((f"cand_{made}", code, spec["name"], [p["name"] for p in parents]))
            else:
                print(f"[propose {made}/{spec['name']}] failed: {err}", flush=True)
        for name, code, fam, par in wave:
            print(f"[{made} {fam:7}] {name:9} {score_and_add(name, code, fam, 1, par)}  "
                  f"(spend ~${spend:.2f})", flush=True)
        arch.save()

    front = arch.pareto_front()
    print(f"\n[llm-autoresearch] DONE. {len(arch.entries)} evaluated, ~${spend:.2f} spent, "
          f"frontier {len(front)} pts:", flush=True)
    for e in front:
        print(f"  {e['name']:14} [{e.get('family','?')}] ratio={e['ratio']:.4f} "
              f"recovery@{e['recovery_fraction']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()

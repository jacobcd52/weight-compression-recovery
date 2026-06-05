"""Stage-1 cheap search on Pythia-1.4B: Opus-heavy proposers evolve activation-aware quantizers,
scored by RECONSTRUCTION LOSS (the victim's eval loss right after decompression, NO retraining) at
the lowest compression ratio. This is cheap (~minutes/candidate, dominated by the compress build),
so a big search is affordable at 1.5B scale; recon loss is a strong proxy for recovery cost (see
11M data) and is exactly what AQLM/GPTQ optimise. Stage-2 (src/pythia/recover.py) then validates
the Pareto-best with real retraining.

    python -m src.pythia.search_recon --proposals 100 --wave 5 --max-spend 30
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from .data import load_packed, Batcher
from .model import load_model, get_weights, set_weights, fp32_bytes, eval_loss
from ..llm.candidate_aware import run_candidate, load_recon
from ..llm.seeds_aware import AWARE_SEEDS
from ..autoresearch.archive import Archive
from ..autoresearch.propose import propose
from ..autoresearch.seeds import SEEDS

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"
COST = {SONNET: 0.02, OPUS: 0.12}

SPEC = """You are inventing a per-tensor WEIGHT-COMPRESSION scheme to steal a 1.4B language model
(GPT-NeoX / Pythia). THREAT MODEL: the attacker is INSIDE the data center — full access to weights,
the GPU, the dataset, and per-layer ACTIVATION STATISTICS (the input Hessian H=E[xx^T]).

OBJECTIVE (this stage): make the decompressed model's RECONSTRUCTION LOSS as LOW as possible at the
LOWEST compression ratio — i.e. compress so well that the model barely degrades BEFORE any
retraining. (Low recon loss strongly predicts cheap recovery.) The matrices are near-full-rank and
dense, so quantization / vector-quantization is the family that works; minimise the ACTIVATION-
WEIGHTED error  E = sum_rows (w_r-ŵ_r) @ H @ (w_r-ŵ_r), not raw MSE.

Write ONE self-contained snippet using numpy (np), math, and torch (incl. GPU):
    KNOBS = {{ ... }}
    def compress_tensor(w, H=None, **knobs):   # w float32 (out,in); H float32 (in,in) or None
        return payload
    def reconstruct_tensor(payload, shape):     # PURE: payload-only
        return np.ndarray(shape, float32)
Cost is len(zlib(pickle(payload))) (torch tensors -> numpy first). Banned: file/net/os, eval/exec,
*.load/*.save. For big matrices CHUNK GPU ops so you don't OOM (the embeddings are 50304x2048).

WHAT WE VERIFIED: naive residual-VQ needs ~8 bits/weight for near-lossless; at 2 bits it gives huge
error. The paper reaches near-lossless at ~1-2 bits via BETTER algorithms — implement them:
  * GPTQ: sequential column quant with inverse-Hessian error feedback (Cholesky of H), group-wise.
  * AQLM: additive multi-codebook VQ with H-WEIGHTED code assignment (beam search) + ALTERNATING
    optimisation (re-assign codes <-> gradient-fine-tune codebooks on E), many iterations.
  * H-driven MIXED precision: more bits on the input directions H says matter.

COMPUTE BUDGET (critical): compress_tensor is called on ALL ~98 matrices of the 1.4B in sequence
and the WHOLE model must finish within ~12 minutes, or the candidate TIMES OUT and scores NOTHING.
Two matrices are HUGE (embed_in / embed_out: 50304x2048 ~ 103M params) and the MLP matrices are
8192-dim — so BOUND your per-matrix cost: cap k-means/beam iterations, codebook size, and fine-tune
steps; CHUNK all GPU ops; and use a cheaper setting (or fewer iterations) for the giant embeddings.
Reference: a working AQLM seed (chunked k-means + ~120 codebook fine-tune steps) processes the whole
model in ~5 min — stay within ~2x that. Unbounded beam search over the embeddings WILL time out.
BEAT the current best recon loss at low ratio, but make sure it FINISHES.
"""

SPECS = [
    {"name": "aqlm", "model": OPUS, "focus": "Full AQLM: additive multi-codebook VQ, H-weighted "
     "(beam-search) code assignment, ALTERNATE between re-assigning codes and gradient-fine-tuning "
     "codebooks on the H-weighted error E, many rounds. Drive bits/weight down at low recon loss."},
    {"name": "gptq", "model": OPUS, "focus": "GPTQ: sequential column quantization with inverse-"
     "Hessian error feedback (Cholesky of H), group-wise low-bit, cheaply-coded scales."},
    {"name": "creative", "model": OPUS, "focus": "Unconventional + data-aware: H-driven mixed "
     "precision, learned rotation / incoherence before quant, product+residual hybrids, entropy "
     "coding. Heavy compute OK. Lowest recon loss at lowest ratio."},
    {"name": "mutate", "model": SONNET, "focus": "Tune the best PARENT's knobs (codebook size, "
     "sub-vector dim, #codebooks, bits, group size, fine-tune steps) for lower ratio at low recon."},
]


def build_prompt(parents, spec, best):
    parts = [SPEC, "\nYOUR SPECIALIZATION: " + spec["focus"]]
    if best:
        parts.append(f"\nCurrent best: ratio {best[0]:.4f} at recon loss {best[1]:.3f} "
                     f"(victim loss {best[2]:.3f}). Beat it (lower ratio at low recon loss).")
    if parents:
        parts.append("\nParents (ratio | recon loss) to mutate/combine:")
        for p in parents:
            parts.append(f"\n# === {p['name']} ratio={p['ratio']:.4f} recon={p.get('recon',9):.3f} ===\n"
                         + p["code"].strip())
    parts.append("\nReturn ONLY one ```python block (KNOBS, compress_tensor(w,H=None,**knobs), "
                 "reconstruct_tensor(payload,shape)).")
    return "\n".join(parts)


def seed_codes():
    # only chunk-safe + fast-byte-count seeds at 1.4B scale: the H-weighted AQLM (compact payload,
    # chunked k-means) + 8-bit scalar quant (near-lossless high-bpp anchor). The other AWARE_SEEDS
    # (aqlm_aware/vq_plain) use non-chunked k-means -> OOM on the embeddings; the int8 numpy seeds
    # (quant4/kmeans4) make ~GB payloads -> slow zlib. Opus generates the rest.
    return {"seed_aqlm_finetune": AWARE_SEEDS["seed_aqlm_finetune"],
            "seed_quant8": SEEDS["seed_quant8"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim-dir", default="runs/pythia_victim")
    ap.add_argument("--revision", default="step143000")
    ap.add_argument("--cache-dir", default="data/pile")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--build-timeout", type=int, default=900)
    ap.add_argument("--proposals", type=int, default=100)
    ap.add_argument("--wave", type=int, default=5)
    ap.add_argument("--max-spend", type=float, default=30.0)
    ap.add_argument("--archive", default="runs/pythia_search/archive.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(os.path.dirname(args.archive), exist_ok=True)
    _, ev, meta = load_packed(args.cache_dir)
    eb = Batcher(ev, args.batch_size, device, seed=1)
    wnpz = os.path.join(args.victim_dir, "weights.npz")
    hess = os.path.join(args.victim_dir, "hessians.pt")

    # resident victim model: reused as the vessel — set_weights(recon) overwrites the 2D matrices,
    # the dense 1D params (LayerNorm, biases) stay at victim values (stolen verbatim).
    victim = load_model(args.revision, device=device, dtype=torch.float32)
    vloss = eval_loss(victim, eb, device, max_batches=args.eval_batches)
    vic_W = get_weights(victim)                    # to restore between candidates
    fp32 = fp32_bytes(vic_W)
    print(f"[pythia-search] victim {args.revision} recon-eval loss={vloss:.4f} fp32={fp32/1e9:.2f}GB "
          f"proposals<={args.proposals} max_spend=${args.max_spend}", flush=True)

    arch = Archive(args.archive, budget_fraction=1.0).load()

    def score_add(name, code, fam, gen, parents):
        r = run_candidate(code, wnpz, hess, timeout=args.build_timeout)
        if not r["ok"]:
            arch.add({"name": name, "code": code, "family": fam, "ratio": 1.0, "recovered": False,
                      "recovery_fraction": 9.99, "recon": 9.99, "gen": gen, "parents": parents,
                      "valid": False, "error": r["error"][:200]})
            return f"INVALID ({r['error'][:70]})"
        recon = load_recon(r["recon_path"]); os.remove(r["recon_path"])
        set_weights(victim, recon)
        rl = eval_loss(victim, eb, device, max_batches=args.eval_batches)
        set_weights(victim, vic_W)                 # restore for the next candidate
        # store recon loss as the minimised cost (recovered=valid build)
        arch.add({"name": name, "code": code, "family": fam, "ratio": r["ratio"], "recovered": True,
                  "recovery_fraction": rl, "recon": rl, "gen": gen, "parents": parents, "valid": True})
        return f"ratio={r['ratio']:.4f} ({r['ratio']*32:.2f}bpp) recon_loss={rl:.4f}"

    if not arch.entries:
        for name, (fam, code) in seed_codes().items():
            print(f"[seed] {name:22} {score_add(name, code, fam, 0, [])}", flush=True)
        arch.save()

    rng = random.Random(0)
    made = 0; spend = 0.0; si = 0
    while made < args.proposals and spend < args.max_spend:
        wave = []
        while len(wave) < args.wave and made < args.proposals and spend < args.max_spend:
            spec = SPECS[si % len(SPECS)]; si += 1
            parents = arch.sample_parents(k=2, rng=rng)
            rec = [e for e in arch.entries if e.get("valid")]
            best = None
            if rec:
                b = min(rec, key=lambda e: (e["ratio"], e.get("recon", 9)))
                best = (b["ratio"], b.get("recon", 9), vloss)
            code, err = propose(build_prompt(parents, spec, best), backend="api", model=spec["model"])
            made += 1; spend += COST.get(spec["model"], 0.05)
            if code:
                wave.append((f"cand_{made}", code, spec["name"], [p["name"] for p in parents]))
            else:
                print(f"[propose {made}/{spec['name']}] failed: {err}", flush=True)
        for name, code, fam, par in wave:
            print(f"[{made} {fam:8}] {name:9} {score_add(name, code, fam, 1, par)}  (~${spend:.2f})", flush=True)
        arch.save()

    valid = [e for e in arch.entries if e.get("valid")]
    print(f"\n[pythia-search] DONE. {len(arch.entries)} evaluated, ~${spend:.2f}. "
          f"victim loss {vloss:.3f}. Best recon loss per ratio band:", flush=True)
    for e in sorted(valid, key=lambda e: e["ratio"])[:12]:
        print(f"  {e['name']:14}[{e.get('family','?'):8}] ratio={e['ratio']:.4f} ({e['ratio']*32:.2f}bpp) recon={e['recon']:.3f}", flush=True)


if __name__ == "__main__":
    main()

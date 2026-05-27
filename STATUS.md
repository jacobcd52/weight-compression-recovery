# STATUS

updated: 2026-05-27T03:00:00Z
state: AUTORESEARCH_PILOT_DONE
phase: Phase B / auto-research (LLM evolves compression schemes) — pilot complete
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: decide scale-up (more proposals / richer primitives) or move to SimpleStories LLM

## Headline result (pilot, 150 proposals, ~$5, ~75 min)
An LLM (Sonnet, Anthropic API) evolved per-tensor compression schemes against the vmap-ensemble
recovery metric (epoch-30 target loss 0.510, 30-epoch denominator, 5% budget, LR-swept). It
**beat the hand-written seeds**: recovering frontier pushed from quant4 @ ratio 0.098 (free
recovery) down to **cand_101 @ ratio 0.0528 (≈19× smaller), recovering in 2.13%**.

Final Pareto front (4 points): seed_quant4 0.098 @0.00% · cand_86 0.072 @1.49% ·
cand_134 0.061 @1.92% · cand_101 0.053 @2.13%.

Every evolved frontier point is **group-wise low-bit (2-bit) quantization with a NormalFloat
codebook** — the GPTQ/AWQ/QLoRA-NF4/AQLM recipe, rediscovered. The frontier is the `group_size`
trade-off; the winner also delta-encodes + 8-bit-quantizes the per-group scales.

156 schemes evaluated, 102 recovered, ~47 invalid/build-fail (rejected by the safety screen).

## Report (clickable URL)
- Main: https://jacobcd52.github.io/weight-compression-recovery/
- Auto-research: https://jacobcd52.github.io/weight-compression-recovery/autoresearch.html
  (Pareto frontier + search-progress curve + frontier table + winning program source)

## Setup recap
- Eval engine: GroupNorm ResNet-20, best-LR baseline (loss@ep30=0.510), vmap ensemble (wide-batch),
  honest bytes = zlib(serialized payload), candidate sandbox (subprocess+timeout+static screen).
- API key in /workspace/.env (gitignored; ROTATE after project — shared in plaintext). $100 credit.
- Backends: api (default for pilot) or claude_p (subscription login).

## Recent
- 2026-05-27 — pilot complete; built docs/autoresearch.html + figures; linked from index.html
- 2026-05-27 00:00Z — launched 150-proposal API pilot
- earlier: epoch-30 metric + wide-batch ensemble + curves-by-default

## Backlog
- Scale-up auto-research: more proposals, richer primitives (per-channel, residual/2-level VQ,
  optimizer-state exfiltration as a method), and a prompt that pushes harder into the 0.01-0.05 band.
- SimpleStories ~10-50M LLM as the next scale (middle ground before 1.5B Qwen).
- Stale text in the *sweep* report (_findings still says "10% budget" / "~7%") — refresh to 5%/ep30.
- RESULTS.md rewrite to current methodology; multi-seed error bars; rotate API key when done.

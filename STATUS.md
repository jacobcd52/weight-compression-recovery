# STATUS

updated: 2026-05-27T00:00:00Z
state: AUTORESEARCH_PILOT_RUNNING
phase: Phase B / auto-research (LLM evolves compression schemes)
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: pilot ~150 proposals (~$5-6, ~60-75 min), then inspect frontier

## Now
Auto-research PILOT running (exp job `autoresearch`): an LLM (Sonnet, Anthropic API) proposes
per-tensor compression schemes; each is sandbox-built, scored by the vmap ensemble (epoch-30
target 0.510, 30-epoch denominator, 5% budget, LR-swept), and added to a Pareto-grid archive.
Compression-only search; 150 proposals, waves of 8.

## Setup recap
- Eval engine: GroupNorm ResNet-20, best-LR baseline (loss@ep30=0.510), vmap ensemble (wide-batch),
  honest bytes = zlib(serialized payload) (hack-resistant), candidate sandbox (subprocess+timeout+static screen).
- Seeds (quant/kmeans/lowrank/sparse) frontier = quant4 @ ratio 0.098 (free recovery). Search target:
  beat 0.098 (lower ratio, still recover) — quant2 @ 0.011 is DNR, so the action is the 0.01-0.1 band.
- API key in /workspace/.env (gitignored; ROTATE after project — shared in plaintext). $100 credit added.
- Backends: api (default for pilot) or claude_p (subscription login).

## Recent
- 2026-05-27 00:00Z — launched 150-proposal API pilot
- API smoke 6/6 valid candidates (139s); seed validation 6 seeds (217s)
- earlier: epoch-30 metric + wide-batch ensemble + curves-by-default

## Issues / flags
none. (API account needed funding; resolved with $100 credit.)

# STATUS

updated: 2026-05-26T03:25:00Z
state: DONE
phase: v2 sweep complete (test-loss, undertrained baseline, full AQLM, amortized plot)
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: awaiting human

## Now
DONE. v2 sweep complete (29 runs). Two Pareto plots (honest + amortized) + baseline curve + table
all embedded in the clickable report. Pushed; live page refreshed.
- Report: https://jacobcd52.github.io/weight-compression-recovery/

## Result summary (v2)
- Undertrained 50-epoch baseline: test loss 0.401, acc 88.1%. Recovery = test loss < 0.401 (strict).
- Recovery cost ~constant ~7% for all recovering methods (schedule artifact) → the real axis is
  max compression that recovers.
- Recoverable frontier ~32x (amortized ~0.031): quantize_1, additive_vq_1_8_8, aqlm_2_4_8.
- AMORTIZED plot (codebook/scale overhead removed): AQLM jumps from full 0.218 -> amortized 0.0625
  and recovers @7% — competitive with kmeans/quantize/additive_vq. Honest bytes penalize AQLM's
  per-layer codebooks on a 270k model; amortized shows it's fine at scale. VQ family ~tied here.

## Recent (most recent first)
- 2026-05-26T03:25Z — v2 sweep done (29 runs); amortized + honest plots + baseline curve in report
- 2026-05-26T02:35Z — launched v2 sweep
- 2026-05-26T02:20Z — 50-epoch baseline + full AQLM

## Open runs / Issues
- none

## Backlog / future
- ~50M-param LLM middle ground (where AQLM codebook tax amortizes); then maybe 1.5B (see SCALING.md).
- Decouple the recovery-schedule confound (recovery cost is ~constant ~7% under the current recipe).
- Full-Hessian + beam-search AQLM (current uses diagonal Hessian + greedy codes).

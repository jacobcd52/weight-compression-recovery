# STATUS

updated: 2026-05-26T01:35:00Z
state: DONE
phase: AQLM reproduction + fine-grained recovery sweep complete
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: awaiting human

## Now
DONE. Reproduced additive multi-codebook VQ (AQLM-style) on ResNet-20 and swept the full
compression-vs-recovery-cost curve with fine (0.1%) resolution. Plots + report + RESULTS + SCALING
updated and pushed; live report refreshed.
- Report: https://jacobcd52.github.io/weight-compression-recovery/  (now 63 runs)

## Result summary
- additive_vq recovers up to 14x within 10% budget (~7.5-8.7% cost); DNR at 28x+ (89.3 vs 89.58).
  Competitive with — not better than — kmeans/quantize at this scale (paper's AQLM wins are LLM-scale).
- Fine cadence reveals BIMODAL recovery: near-lossless 4-5x methods recover <0.1%; aggressive ones
  need ~7-9%; the 0.1-1% region is essentially empty.
- Beyond ~16x, plain-CE recovery needs >10% budget (distillation previously reached 32x).
- SCALING.md: 1.5B sweep ~$0.5-6k at <=0.1% cap, ~$56k at 1% cap; recovery-retraining dominates.

## Recent (most recent first)
- 2026-05-26T01:35Z — AQLM sweep done (13 runs); plots/report/RESULTS/SCALING updated, state DONE
- 2026-05-26T01:05Z — launched AQLM sweep (additive_vq + fine cadence)
- 2026-05-26T00:20Z — prior sweep DONE (56 runs, 9 recovered)
- 2026-05-25T22:46Z — baseline 90.08%

## Open runs
- none

## Completed runs
- baseline + 56 prior + 7 additive_vq + 6 fine re-runs = full curve. See results/summary.csv.

## Issues / flags
none. (drivers show "CRASHED" in exp only because drain exits rc3 = queue empty; logs confirm clean.)

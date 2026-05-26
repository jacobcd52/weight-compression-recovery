# STATUS

updated: 2026-05-26T00:20:00Z
state: DONE
phase: Phase 6 complete
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: awaiting human

## Now
DONE. All 56 sweep runs complete (40 plain + 16 distill). Pareto figure, results CSV,
HTML report, and RESULTS.md all generated and pushed.

- Clickable report: https://jacobcd52.github.io/weight-compression-recovery/
- Figure: figures/pareto.png · Data: results/summary.csv · Write-up: RESULTS.md

## Result summary
Baseline 90.08%; threshold 89.58%; 10% budget (20 epochs). **9 of 56 recovered.**
Pareto frontier (most compression first):
- quantize_1_distill: ratio 0.031 (32×), recovered at 8.5% of training cost
- kmeans_2_distill:   ratio 0.063 (16×), 6.0%
- kmeans_4_distill:   ratio 0.126 (8×),  0.5%
- kmeans_6:           ratio 0.192 (5×),  0.5%
- quantize_8:         ratio 0.250 (4×),  0.5%
Quantization & k-means weight-sharing recover cheaply; all pruning + low-rank are DNR in budget;
distillation extends recovery to 1-bit weights.

## Recent (most recent first)
- 2026-05-26T00:20Z — Phase 6 done: figure + report + RESULTS.md pushed; state DONE
- 2026-05-26T00:10Z — sweep finished 56/56, 9 recovered
- 2026-05-25T23:51Z — 38/56
- 2026-05-25T22:46Z — baseline 90.08%
- 2026-05-25T22:05Z — Phase 0 complete

## Open runs
- none

## Completed runs
- baseline 90.08% + 56 sweep runs (9 recovered). See results/summary.csv / RESULTS.md.

## Issues / flags
none. (sweep_driver shows "CRASHED" in exp only because the drain loop exits with code 3 =
"queue empty"; the log confirms clean completion.)

# STATUS

updated: 2026-05-25T23:51:00Z
state: SWEEP_RUNNING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Sweep draining via exp job **sweep_driver**. **38/56 complete**, 0 crashes, **6 recovered**.
~25 min remaining. Currently: magprune_quant_0.01_4. Remaining: rest of magprune_quant + 16 distill variants.

## NOTE FOR AUTOPILOT / FRESH SESSIONS
**Do NOT call `sweep launch-next` manually** while `sweep_driver` is alive (`exp status sweep_driver`).
If it died with the queue not empty:
`exp sweep_driver -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain`.
Queue empty → `python -m src.plot` then Phase 6 (RESULTS.md, DONE issue).

## Headline result so far
**Quantization & k-means weight-sharing dominate the Pareto frontier** — they preserve the weight
distribution, so the reconstructed init is near-lossless and recovery is cheap:
- quantize_8 (ratio 0.250): REC @ recfrac 0.005 (~1 epoch)
- kmeans_6  (ratio 0.192): REC @ recfrac 0.005
- kmeans_4  (ratio 0.126): REC @ 0.080      quantize_4 (0.125): REC @ 0.080
- kmeans_2  (ratio 0.063): REC @ 0.080   <- best compression-with-recovery so far
- magnitude_prune_0.5 (ratio 0.531): REC @ 0.085
Pruning (random/snip/fisher/most magnitude) & low-rank: DNR within the 10% budget.

## Recent (most recent first)
- 2026-05-25T23:51Z — 38/56, 6 recovered. Quant/kmeans recover cheaply; magprune_quant + distill pending
- 2026-05-25T23:32Z — 25/56, 1 recovered
- 2026-05-25T22:52Z — launched sweep_driver
- 2026-05-25T22:46Z — baseline 90.08%

## Open runs
- sweep_driver: draining queue (38/56)
- magprune_quant_0.01_4: running

## Completed (6 recovered; see headline)
- quantize_8/4, kmeans_6/4/2, magnitude_prune_0.5 recovered; all pruning+low_rank knobs DNR

## Issues / flags
none.

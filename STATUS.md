# STATUS

updated: 2026-05-25T23:32:00Z
state: SWEEP_RUNNING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Sweep draining via exp job **sweep_driver**. **25/56 complete**, 0 crashes, 1 recovered so far.
~45 min remaining. Currently: low_rank_0.05. Still pending: quantize, kmeans, magprune_quant,
and all 16 distill variants (these are expected to recover more — 8-bit quant init is near-lossless,
distillation tends to push borderline cases over the line).

## NOTE FOR AUTOPILOT / FRESH SESSIONS
**Do NOT call `sweep launch-next` manually** while `sweep_driver` is alive (`exp status sweep_driver`).
If it died with the queue not empty:
`exp sweep_driver -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain`.
Queue empty → `python -m src.plot` then Phase 6 (RESULTS.md, DONE issue).

## Key pattern so far (plain runs)
The 10% budget (20 epochs) is a HARD bar vs a 200-epoch baseline. Pruning/low-rank mostly DNR,
clustering 85–89.6% final. Only magnitude_prune_0.5 recovered (REC@0.085). Method quality order
visible: magnitude ≈ snip > fisher > random for a given ratio; many land just under 89.58.

## Recent (most recent first)
- 2026-05-25T23:32Z — 25/56 done, 1 recovered. Through pruning+low_rank; quant/kmeans/distill pending
- 2026-05-25T23:13Z — 13/56
- 2026-05-25T22:52Z — launched sweep_driver
- 2026-05-25T22:46Z — baseline DONE 90.08%

## Open runs
- sweep_driver: draining queue (25/56)
- low_rank_0.05: running

## Completed (recovered marked)
- magnitude_prune_0.5: REC@0.085 (only recovery so far)
- pruning (mag/snip/fisher/random) + low_rank: all other knobs DNR, finals 75–89.5

## Issues / flags
none.

# STATUS

updated: 2026-05-25T23:13:00Z
state: SWEEP_RUNNING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Sweep draining via exp job **sweep_driver** (one job at a time). **13/56 complete**, no crashes.
~1.5 min/run → ~1 h remaining. Currently: snip_0.25.

## NOTE FOR AUTOPILOT / FRESH SESSIONS
**Do NOT call `sweep launch-next` manually** while `sweep_driver` is alive (`exp status sweep_driver`).
If it died with the queue not empty, resume:
`exp sweep_driver -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain`.
Queue empty → `python -m src.plot` then Phase 6.

## Recent (most recent first)
- 2026-05-25T23:13Z — 13/56 done, 1 recovered. magnitude_prune degrades cleanly; many DNRs land
  JUST under threshold (89.53/89.54 vs 89.58) — 10% budget is right at the recovery edge.
- 2026-05-25T23:02Z — 7/56; magnitude_prune_0.5 recovers, random_sparse all DNR
- 2026-05-25T22:52Z — launched sweep_driver
- 2026-05-25T22:46Z — baseline DONE 90.08%

## Open runs
- sweep_driver: draining queue (13/56)
- snip_0.25: running

## Completed runs (recovered = Y; most recent technique groups)
- snip_0.5:               ratio=0.531, DNR, final=89.54
- magnitude_prune_0.5:    ratio=0.531, REC@0.085, final=89.60
- magnitude_prune_0.25:   ratio=0.281, DNR, final=89.53
- magnitude_prune_0.1:    ratio=0.131, DNR, final=88.76
- magnitude_prune_0.05:   ratio=0.081, DNR, final=86.47
- magnitude_prune_0.01:   ratio=0.020, DNR, final=87.32
- magnitude_prune_0.001:  ratio=0.002, DNR, final=74.66
- random_sparse_*:        all DNR (final 82–89), random zeroing destroys structure

## Issues / flags
none. Observation: with a 10% budget many techniques plateau just below the 0.5pp threshold;
distill variants + near-lossless quantize (8-bit) still pending and expected to recover.

# STATUS

updated: 2026-05-25T23:02:00Z
state: SWEEP_RUNNING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Sweep draining autonomously via exp job **sweep_driver** (one retrain job at a time).
Progress: **7/56 runs complete** (~1.5 min/run → ~1 h remaining). Currently: magnitude_prune_0.25.

## NOTE FOR AUTOPILOT / FRESH SESSIONS
**Do NOT call `sweep launch-next` manually** while `sweep_driver` is alive (`exp status sweep_driver`).
It drives the queue. If it has died with the queue not empty, resume by relaunching it as an exp job:
`exp sweep_driver -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain`.
When the queue is empty, run `python -m src.plot` then do Phase 6. Progress: `python -m src.sweep status`.

## Recent (most recent first)
- 2026-05-25T23:02Z — 7/56 done. magnitude_prune_0.5 RECOVERED (recfrac 0.085); all random_sparse DNR
- 2026-05-25T22:52Z — launched sweep_driver
- 2026-05-25T22:46Z — baseline DONE 90.08%; built 56 compressed reps
- 2026-05-25T22:05Z — Phase 0 complete

## Open runs
- sweep_driver: draining queue (7/56 done)
- magnitude_prune_0.25: running

## Completed runs (most recent first)
- magnitude_prune_0.5: ratio=0.531, recovery_fraction=0.085, final_acc=89.60 (RECOVERED)
- random_sparse_0.5:   ratio=0.531, recovery=DNR, final_acc=89.01
- random_sparse_0.25:  ratio=0.281, recovery=DNR, final_acc=85.91
- random_sparse_0.1:   ratio=0.131, recovery=DNR, final_acc=87.42
- random_sparse_0.05:  ratio=0.081, recovery=DNR, final_acc=87.64
- random_sparse_0.01:  ratio=0.020, recovery=DNR, final_acc=88.46
- random_sparse_0.001: ratio=0.002, recovery=DNR, final_acc=82.01

## Issues / flags
none. Note: random_sparse contributes only DNR points (random zeroing destroys structure,
no recovery within 10% budget) — a legitimate finding, not an error.

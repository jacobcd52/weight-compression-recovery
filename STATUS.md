# STATUS

updated: 2026-05-25T22:52:00Z
state: SWEEP_RUNNING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Sweep is draining autonomously. An exp job named **sweep_driver** runs `src.sweep drain`,
which launches retrain jobs ONE AT A TIME (waits while one is RUNNING, launches the next
when the GPU frees up). 56 runs total (40 plain + 16 distill). Each capped at 7,820 steps
(20 epochs); recovered runs stop early at 89.58% test acc.

## NOTE FOR AUTOPILOT / FRESH SESSIONS
**Do NOT call `sweep launch-next` manually** while `sweep_driver` is alive — it already
drives the queue and a manual launch could double-launch on one GPU. Check first:
`exp status sweep_driver`. Only if sweep_driver is CRASHED/DONE-but-queue-not-empty should
you resume it: `/workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain`
(run it as an exp job, or `launch-next` once). When the queue is empty, run
`python -m src.plot` and do Phase 6. Progress: `python -m src.sweep status`.

## Recent (most recent first)
- 2026-05-25T22:52Z — launched sweep_driver (autonomous queue drain)
- 2026-05-25T22:51Z — run #1 random_sparse_0.5: ratio 0.531, final 89.01% → DNR (just under threshold)
- 2026-05-25T22:46Z — baseline DONE 90.08%; built 56 compressed reps
- 2026-05-25T22:05Z — Phase 0 complete

## Open runs
- sweep_driver: draining the 56-run queue

## Completed runs (most recent first)
- random_sparse_0.5: ratio=0.531, recovery=DNR, final_acc=89.01

## Issues / flags
none

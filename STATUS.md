# STATUS

updated: 2026-05-26T01:05:00Z
state: SWEEP_RUNNING
phase: AQLM high-compression sweep (reproducing "Aggressive Compression Enables LLM Weight Theft")
gpu: NVIDIA GeForce RTX 4090, ~24 GB
next: sweep running (~15-20 min)

## Now
New direction: reproduce the paper's headline method — **additive multi-codebook VQ (AQLM-style)** —
on ResNet-20, and sweep the FULL compression-ratio vs recovery-compute curve with fine resolution
in the 0.1-1% budget region (kept the 10% cap too). exp job **sweep_driver_aqlm** is draining a
7-knob queue (additive_vq, 7x-126x compression), plain-CE recovery, eval every 78 steps (~0.1%).

Built reps + init accuracy: 7x→init 86%, 14x→28%, 28x/63x/126x→~random. The sweep measures how
much retraining compute each needs to recover (threshold 89.58%).

## NOTE FOR AUTOPILOT / FRESH SESSIONS
Two sweeps use separate queues via --tag. The AQLM one: driver `sweep_driver_aqlm`, queue
runs/queue_aqlm.txt. Resume if it dies with queue non-empty:
`exp sweep_driver_aqlm -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain --config configs/aqlm.yaml --sweep configs/sweep_aqlm.yaml --tag aqlm`
When done: `python -m src.plot` then `python -m src.report`; both auto-include the new points.

## Recent (most recent first)
- 2026-05-26T01:05Z — implemented additive_vq + fine-cadence eval; launched 7-knob AQLM sweep
- 2026-05-26T00:20Z — Phase 1-6 DONE (8 techniques, 56 runs); report live
- 2026-05-25T22:46Z — baseline 90.08%

## Open runs
- sweep_driver_aqlm: draining 7-knob additive_vq queue

## Completed runs
- prior sweep: 56 runs, 9 recovered (report live at the Pages URL)

## Issues / flags
none

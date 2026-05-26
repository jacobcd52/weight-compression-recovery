# STATUS

updated: 2026-05-26T02:35:00Z
state: SWEEP_RUNNING
phase: v2 sweep — undertrained CIFAR baseline, test-loss recovery, full AQLM, promising methods only
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Major regime change this session:
- Baseline retrained to **50 epochs** (undertrained, less overfit): acc 88.12%, **test loss 0.4013**
  (lower than the 200-epoch's 0.62 — overfitting confirmed).
- Recovery metric switched to **test loss** (recover when full-test CE < baseline 0.4013, strict).
- Retrain **warmup cut** from 5 epochs to 100 steps; eval every 20 steps (fine).
- **Full AQLM implemented** (activation-aware per-layer additive quant) alongside simplified additive_vq.
- Sweep restricted to **promising methods** (quantize, kmeans, magnitude_prune, magprune_quant,
  additive_vq, aqlm); dropped random/snip/fisher/low_rank.

exp job **sweep_driver_v2** is draining a 29-run queue (queue_v2.txt). 1 done (quantize_8 recovered
@7.0%). ~30-50 min remaining.

## NOTE FOR AUTOPILOT / FRESH SESSIONS
Driver: sweep_driver_v2, queue runs/queue_v2.txt. Resume if dead with queue non-empty:
`exp sweep_driver_v2 -- /workspace/envs/weight-compression-recovery/bin/python -m src.sweep drain --config configs/retrain.yaml --sweep configs/sweep_promising.yaml --tag v2`
When empty: `python -m src.plot && python -m src.report`. Plot only shows test-loss-regime runs.

## Recent (most recent first)
- 2026-05-26T02:35Z — launched v2 sweep (test-loss, undertrained, full AQLM, promising only)
- 2026-05-26T02:20Z — 50-epoch baseline done (88.12%, loss 0.4013); full AQLM implemented
- 2026-05-26T01:35Z — prior AQLM/additive_vq sweep done (now superseded by v2 regime)

## Open runs
- sweep_driver_v2: draining 29-run queue (1/29 done)

## Completed runs
- quantize_8: ratio 0.250, recovered @ recovery_fraction 0.070 (test_loss 0.399 < 0.401)

## Issues / flags
none

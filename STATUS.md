# STATUS

updated: 2026-05-25T22:20:00Z
state: BASELINE_RUNNING
phase: Phase 2 (baseline) + Phase 3/4 (writing compress/retrain) in parallel
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Baseline ResNet-20 training is running (200 epochs, 391 steps/epoch, 78,200 total steps).
While it trains I'm writing + testing the compression library and retraining harness.

## Recent (most recent first)
- 2026-05-25T22:20Z — baseline healthy: epoch 6 test_acc 68.5% (1:36→6:68.5), trending to ~91%
- 2026-05-25T22:16Z — first baseline launch CRASHED (exp used system python, no tensorboard);
  fixed by launching with the explicit venv python path. Noted in GOTCHAS.
- 2026-05-25T22:14Z — Phase 1 core pipeline committed; smoke test passed (roundtrip OK, 0.78s eval)
- 2026-05-25T22:05Z — Phase 0 complete (scaffolding, deps, setup-complete issue)

## Open runs
- baseline: ~epoch 6/200, last test_acc 68.5%, budget-irrelevant (this is the baseline)

## Completed runs (most recent first)
- (none)

## Issues / flags
none — first baseline launch crashed on venv/python mismatch, now fixed.

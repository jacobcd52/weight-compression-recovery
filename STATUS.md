# STATUS

updated: 2026-05-25T22:46:00Z
state: SWEEP_LAUNCHING
phase: Phase 5 / sweep
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
Baseline DONE at 90.08% test acc. Preparing the compression sweep: building all ~35
compressed representations from runs/baseline/best.pt, writing runs/queue.txt, then
launching retrain jobs one at a time via exp.

## SWEEP RECOVERY PROCEDURE (for a fresh autopilot session)
Jobs run one at a time. When a retrain job finishes/crashes, from the project dir run:
  `/workspace/envs/weight-compression-recovery/bin/python -m src.sweep launch-next`
It is stateless (driven by runs/<name>/summary.json existence): it waits if one is still
RUNNING, else launches the next incomplete queue entry. When it prints "QUEUE EMPTY", run
`python -m src.plot` and proceed to Phase 6. A crashed job (no summary, not running) will be
relaunched up to 3 attempts; investigate if a run hits that cap.

## Recent (most recent first)
- 2026-05-25T22:46Z — baseline DONE: best 90.08% / final 89.88%; threshold=89.58%, budget=7820 steps
- 2026-05-25T22:40Z — all code written & smoke-tested
- 2026-05-25T22:20Z — baseline relaunched on venv python (healthy)
- 2026-05-25T22:05Z — Phase 0 complete

## Open runs
- (about to launch sweep run #1)

## Completed runs (most recent first)
- baseline: test_acc=90.08% (not a sweep point — the reference)

## Issues / flags
none

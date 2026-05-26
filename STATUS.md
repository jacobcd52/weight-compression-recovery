# STATUS

updated: 2026-05-26T05:20:00Z
state: BUILDING (vmap ensemble) + SWEEP_RUNNING (v4 BN reference)
phase: speedups -> vectorized ensemble evaluator; LR-swept recovery metric
gpu: NVIDIA GeForce RTX 4090, ~24 GB

## Now
- v4 BatchNorm sweep (LR-swept recovery, concurrency 4) finishing: 18/29. This is the reference frontier.
- Built the **vmap ensemble evaluator** (src/ensemble.py) + GroupNorm ResNet-20 + orchestration
  (src/sweep_ensemble.py). CPU-validated. Waiting for v4 to free the GPU, then: train GN baseline,
  run the ensemble sweep, validate vs per-process, measure speedup, regenerate frontier.

## Key decisions this session
- Recovery metric = TEST LOSS below the (undertrained 50-epoch) baseline; cost = MIN over a small
  peak-LR sweep {3e-5,1e-4,3e-4,1e-3} (the retraining recipe is part of the method).
- Diagnostic proved the old ~7% was an LR-schedule artifact (easy inits recover ~33x cheaper at low LR;
  hard inits need the cosine anneal). Live report has the loss curves.
- Speedups: across-model concurrency (~2x measured; GPU was ~55% util, time-sliced) + exact LR pruning.
  Decided to go further with the **vmap ensemble** (GroupNorm => stateless => vmap-clean): trains
  (init x LR) configs vectorized in one batched pass; eval (the real bottleneck) done batched too.
- Auto-research orchestration plan: cheap concurrent LLM proposals -> single batched (vmap) GPU
  evaluator; recipe parameterized so candidates stay vmappable; MPS only as escape hatch.

## Next
- Plan 1 (SimpleStories-V2-11M LM) once the image-side eval engine is solid; ~free compute, ~half-day eng.

## Issues / flags
none.

# NOTES — running log of decisions & future work

## Decisions log (most recent first)

- 2026-05-25 — **BatchNorm / biases / 1-D tensors are dense & uncompressed** and are
  excluded from BOTH the numerator and denominator of the compression ratio. Only conv
  and linear *weight* tensors (≥2-D) are compressed and counted. This keeps the ratio
  honest and comparable across techniques.
- 2026-05-25 — **bf16 mixed precision** for training: `torch.amp.autocast('cuda',
  dtype=torch.bfloat16)` around the forward/loss, master weights kept fp32, **no GradScaler**
  (bf16 has fp32 exponent range so loss scaling is unnecessary).
- 2026-05-25 — **Compression-ratio denominator is fp32 (4 bytes/weight)** for the
  compressed keys, per brief. Even though we train in bf16, the *baseline storage* reference
  is fp32 weight bytes.
- 2026-05-25 — Single seed (42) for the first pass. Multi-seed error bars are future work.
- 2026-05-25 — Retraining cosine schedule is **rescaled to the 10% budget cap** so LR
  actually decays within the short run; 5-epoch linear warmup retained.
- 2026-05-25 — Added **src/engine.py** (not in the brief's file list) holding the shared
  AdamW+warmup-cosine+bf16 training loop, so train.py and retrain.py use an identical
  optimizer state machine. Keeps the modules small and avoids duplication.
- 2026-05-25 — **configs/sweep.yaml** is a mapping (`techniques:` list +
  `distill_extras_per_technique:`) rather than the brief's "flat list + trailing key",
  which is not a single valid YAML document. Same content.
- 2026-05-25 — Reconstructed init keeps **BN/bias/1-D tensors exactly** (they are the
  uncompressed part of the stolen bytes); only conv/linear weights are lossy. Distillation
  extras = the 2 lowest-ratio knobs per technique; they reuse the same compressed file as
  the plain run (compression is identical, only the retrain loss differs).

## Open questions / assumptions made autonomously
- 2026-05-25 — **torch CUDA build pin.** Default PyPI torch (2.12.0) ships a CUDA 13.0 wheel;
  this pod's driver is 565.57.01 (CUDA 12.7) and is too old → `cuda.is_available()==False`.
  Fixed by pinning `torch==2.5.1 torchvision==0.20.1` (CUDA 12.4 build). Documented in
  requirements.txt. (Also worth adding to the global GOTCHAS file.)

## Planned follow-ups (future work)
- Structure-preserving retraining (option-b): keep the sparsity mask / low-rank structure
  fixed during retraining instead of going dense-from-init.
- Heterogeneous bit allocation across layers (sensitivity-weighted bit budgets).
- Better importance methods: GraSP, OBS (optimal brain surgeon), Wanda.
- Multi-seed error bars on the Pareto frontier.

# BRIEF — weight-compression-recovery

> This is the canonical experiment specification. Any autopilot-woken Claude session should
> read this file (and NOTES.md + STATUS.md) before acting. Operate fully autonomously per
> the global CLAUDE.md autopilot rules — do not stall waiting for a human.

## The research question

**Threat model: weight exfiltration.** If an attacker steals a small number of bytes from a
model's weights and has access to the training data, how cheaply can they retrain to full
performance? We want a Pareto frontier of (compression ratio) vs. (fraction of original
training cost needed to recover within 0.5pp of baseline test accuracy), with one point per
(technique, knob) combination.

## Hard rules (do not deviate without writing why in NOTES.md)

- **Model + data:** ResNet-20 (CIFAR variant, ~270k params) on CIFAR-10. Standard.
- **Precision:** bf16 mixed precision (`torch.amp.autocast('cuda', dtype=torch.bfloat16)`
  around the forward pass; master weights fp32; no GradScaler needed for bf16).
- **Optimizer:** AdamW (lr 1e-3, weight decay 5e-4, betas (0.9, 0.999)) with cosine LR
  schedule and 5-epoch linear warmup. Batch size 128.
- **Epochs (baseline):** 200. Standard CIFAR-10 augmentation: random crop 32 padding 4,
  random horizontal flip, normalize with CIFAR-10 mean/std.
- **Recovery threshold:** test accuracy ≥ baseline_test_acc − 0.5pp on the full 10,000-image
  test split. No subsetting.
- **Retraining budget cap:** if a run does not hit the threshold within 10% of baseline
  steps, kill it and mark it DNR (did not recover).
- **Retraining mode:** dense-from-init (option a). Reconstruct best dense approximation from
  compressed bits, use as init, train with same optimizer/schedule. No sparsity-mask
  preservation.
- **Compression ratio:** compressed_bytes / baseline_fp32_bytes. Count honest bytes — include
  CSR/bitmask index overhead for sparse, U+V byte counts for low-rank, codebook + index bytes
  for k-means, etc. The baseline is fp32 weight storage (4 bytes/weight). BatchNorm stays
  dense and uncompressed; do not count it in either numerator or denominator.
- **Seeds:** single seed (42) for first pass. Multi-seed only after everything else is done.
- **No wandb:** use tensorboard + per-run CSV logs under runs/<run-name>/.
- **Everything important under /workspace.**

## Phases

- **Phase 0 — Setup:** `proj new`, scaffolding files, deps, STATUS.md, setup-complete issue.
- **Phase 1 — Codebase:** src/{models,data,utils,train,compress,retrain,sweep,plot}.py +
  configs/{baseline,sweep}.yaml. Smoke-test train + retrain (--smoke: 2 epochs, 1024 imgs).
- **Phase 2 — Baseline:** `exp baseline -- python -m src.train --config configs/baseline.yaml`.
  Expect ~91% test acc. Sanity-check 88%–94%.
- **Phase 3 — Compression library:** 8 techniques (below) with compress/reconstruct/bytes +
  `__main__` unit tests.
- **Phase 4 — Retraining harness:** plain + distill modes, rescaled cosine, full-test eval per
  epoch, stop at threshold OR 10% budget.
- **Phase 5 — Sweep:** ~35 plain + ~16 distill runs, launched sequentially via exp + a
  runs/queue.txt the autopilot drains one at a time.
- **Phase 6 — Plot & write-up:** figures/pareto.{png,pdf}, results/summary.csv, RESULTS.md,
  STATUS=DONE, final heartbeat issue.

## Compression techniques (Phase 3)

Targets: only conv and linear **weight** tensors. Leave BN params, biases, and any 1-D tensors
dense & uncompressed; exclude their bytes from numerator and denominator.

- **random_sparse** — keep p% of entries at uniformly random positions.
  Knob: keep_fraction ∈ {0.5, 0.25, 0.1, 0.05, 0.01, 0.001}. Bytes = values (fp32) + index
  overhead. Bitmask (1 bit/param) when keep_fraction > ~3%, else packed int32 indices; pick
  the smaller per tensor.
- **magnitude_prune** — keep top-k% by |w| per tensor. Same knob grid + bytes accounting.
- **snip** — keep top-k% by |w · ∂L/∂w| on a single random batch (size 128). Same grid.
- **fisher_prune** — keep top-k% by diagonal Fisher (g² averaged over 10 random batches). Same grid.
- **low_rank** — per tensor (reshape conv to (out, in·k·k)), rank-r SVD. Knob: rank_fraction ∈
  {0.5, 0.25, 0.1, 0.05, 0.01} of min(m,n), rounded up to ≥1. Bytes = r·(m+n)·4.
  Reconstruct: U_r · diag(S_r) · V_r^T.
- **quantize** — uniform scalar quant, per-tensor symmetric, k bits/weight + one fp32 scale
  per tensor. Knob: bits ∈ {8, 4, 2, 1}. Bytes = ceil(numel·bits/8) + 4. Round-to-nearest.
  (1-bit = sign quantization.)
- **kmeans** — per-tensor k-means weight sharing. 2^k centroids, fp32 codebook +
  ceil(log2(2^k)) bits/weight index. Knob: bits ∈ {6, 4, 2}. MiniBatchKMeans, deterministic
  with fixed seed.
- **magprune_quant** — stacked: magnitude-prune to keep_fraction ∈ {0.1, 0.01}, then
  uniform-quantize kept values to bits ∈ {8, 4, 2}. Bytes = kept-values quantized + index.

## API (compress.py)

```python
TECHNIQUES = { "random_sparse": ..., ..., "magprune_quant": ... }

def compress(state_dict, technique, knob, **kwargs) -> dict:
    """keys: technique, knob, payload, bytes, dense_keys, original_shapes."""

def reconstruct(compressed) -> dict:
    """state_dict with all keys; compressed ones reconstructed as dense fp32 tensors."""

def total_bytes(compressed, original_state_dict) -> tuple[int, int]:
    """(compressed_bytes, baseline_fp32_bytes_of_compressed_keys); ratio = first/second."""
```

## Sweep spec (configs/sweep.yaml)

```yaml
- {technique: random_sparse,  knobs: [0.5, 0.25, 0.1, 0.05, 0.01, 0.001]}
- {technique: magnitude_prune, knobs: [0.5, 0.25, 0.1, 0.05, 0.01, 0.001]}
- {technique: snip,           knobs: [0.5, 0.25, 0.1, 0.05, 0.01]}
- {technique: fisher_prune,   knobs: [0.5, 0.25, 0.1, 0.05, 0.01]}
- {technique: low_rank,       knobs: [0.5, 0.25, 0.1, 0.05, 0.01]}
- {technique: quantize,       knobs: [8, 4, 2, 1]}
- {technique: kmeans,         knobs: [6, 4, 2]}
- {technique: magprune_quant, knob_pairs: [[0.1,8],[0.1,4],[0.1,2],[0.01,8],[0.01,4],[0.01,2]]}
distill_extras_per_technique: 2   # 2 most-compressed knobs also run with distillation
```

~35 plain + ~16 distill = ~50 runs. Each capped at 10% of baseline cost (~3-6 min on a 4090).

## Retraining modes (Phase 4)

- `--mode plain` (default): vanilla retraining, identical loss to baseline.
- `--mode distill`: loss = α·KL(softmax(student/T), softmax(teacher/T))·T² + (1-α)·CE(student,labels),
  T=4.0, α=0.9. Teacher = runs/baseline/best.pt, frozen, eval mode.

Same eval cadence (full test set every epoch) and stopping rule for both. Cosine schedule
rescaled to the budget cap (0.10·baseline_steps); warmup unchanged.

## summary.json (per retrain run)

technique, knob, compressed_bytes, compression_ratio, recovered, recovery_steps,
recovery_fraction, final_test_acc, did_distill.

## Check-in protocol

Update STATUS.md (then commit+push) at: phase start, before launching any exp, when woken by
autopilot, when deciding next steps, on error, and at least every 30 min (heartbeat).

NEEDS_HUMAN (set state + open a `needs-human` GitHub issue, cc @jacobcd52) if: same job
crashes 3× and undiagnosable; baseline acc <85% or >95%; about to do something irreversible;
genuinely stuck. Phase-boundary heartbeats: open + immediately close an info issue labeled
`heartbeat`. Don't spam — one issue per genuine event.

## Plot (Phase 6)

x: compression ratio (log, lower=more compressed). y: recovery fraction (linear 0–0.10).
Color by technique, marker by plain/distill. DNR at y=0.10 with right-arrow. Pareto lower
envelope as a line. Readable fonts. -> figures/pareto.{png,pdf} + results/summary.csv.

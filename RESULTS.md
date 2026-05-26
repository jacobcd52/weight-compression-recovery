# RESULTS — weight-compression recovery

**Clickable report:** https://jacobcd52.github.io/weight-compression-recovery/
**Figure:** `figures/pareto.png` · **Data:** `results/summary.csv`

## Setup recap
- ResNet-20 / CIFAR-10. Baseline (200 epochs, AdamW + warmup-cosine, bf16): **90.08%** test acc.
- Recovery threshold = **89.58%** (baseline − 0.5 pp) on the full 10k test set.
- Budget cap = **10%** of baseline steps = 7,820 steps ≈ 20 epochs; otherwise **DNR**.
- Dense-from-init: reconstruct the best dense approximation from the compressed bits, retrain
  with the same recipe (cosine rescaled to the 20-epoch budget). BN/biases kept dense,
  excluded from the byte ratio. Compression ratio = compressed bytes / fp32 bytes of the
  conv+linear weights (honest index/codebook/U+V accounting).
- **56 runs**: 40 plain + 16 distillation (2 most-compressed knobs per technique). Seed 42.

## Headline
**Only 9 of 56 runs recovered within the 10% budget — and they are almost all
information-preserving compressors (scalar quantization and k-means weight-sharing).**
Structure-destroying compressors (random/magnitude/SNIP/Fisher pruning, low-rank SVD) mostly
fail to recover in budget, even at mild compression. Distillation extends the recoverable
frontier all the way to **1-bit weights (~32× compression)**.

### Pareto frontier (most compression first)
| run | ratio | ×smaller | recovery cost | final acc |
|---|---|---|---|---|
| quantize_1_distill | 0.031 | 31.9× | 8.5% of baseline | 89.63 |
| kmeans_2_distill   | 0.063 | 15.9× | 6.0% | 89.80 |
| kmeans_4_distill   | 0.126 | 7.9×  | 0.5% | 89.61 |
| kmeans_6           | 0.192 | 5.2×  | 0.5% | 89.68 |
| quantize_8         | 0.250 | 4.0×  | 0.5% | 89.67 |

All 9 recoveries: quantize_8/4, quantize_1_distill, kmeans_6/4/2, kmeans_2_distill,
kmeans_4_distill, and magnitude_prune_0.5 (the only pruning recovery, at ratio 0.53).

## What worked
- **Scalar quantization (`quantize`).** 8-bit (4×) and 4-bit (8×) recover; 8-bit is essentially
  lossless at init and recovers in **1 epoch** (0.5% of budget). Even **1-bit/sign** weights
  (32×) recover *with distillation*.
- **K-means weight-sharing (`kmeans`).** Strongest method per byte: 6-bit and 4-bit recover
  fast; **2-bit (16×) recovers even with plain retraining**, and faster with distillation.
- These keep the *value distribution* of each tensor, so the reconstructed network starts only
  slightly degraded and snaps back quickly.

## What didn't
- **All pruning (`random_sparse`, `snip`, `fisher_prune`) and `low_rank`: DNR at every knob.**
  Zeroing/low-ranking conv weights collapses the init to ~random (10% acc), and 20 epochs is
  not enough to climb the last fraction of a point back to 89.58% — they plateau at ~85–89.5%.
- **`magnitude_prune`: only 50%-keep (1.9×) recovers**, and only barely (8.5% of budget, 89.60%).
  More aggressive magnitude pruning is DNR.
- **`magprune_quant` (prune-then-quantize): all DNR.** The pruning step dominates the damage, so
  stacking quantization on top doesn't rescue it.

## Surprises
1. **The budget/threshold regime is stringent.** Many DNR runs land *just* under the bar
   (e.g. magnitude_prune_0.25 → 89.53, snip_0.5 → 89.54 vs 89.58). 10% of training is right at
   the edge of "full recovery" for many methods — recovery is closer to all-or-nothing than to
   a smooth curve.
2. **Distillation matters most at extreme compression.** It rescued 1-bit quantization (plain
   DNR → distill recovered) and sped up 2-bit k-means (0.080 → 0.060), but did nothing for the
   structure-destroying methods (a destroyed init can't be distilled back cheaply either).
3. **Method quality ordering for pruning** (by final acc at a given ratio): magnitude ≈ SNIP >
   Fisher > random — but the differences are washed out because none recover in budget.

## Threat-model takeaway
For the weight-exfiltration threat model: **stealing quantized or clustered weights is far more
"recoverable per byte" than stealing a sparse subset.** An attacker who exfiltrates a 4-bit or
2-bit codebook representation (8–16× smaller than fp32) can retrain to full accuracy for ~0.5–8%
of the original training cost; an attacker who steals a pruned/low-rank slice generally cannot,
within a 10% budget. Information that preserves the weight *distribution* is cheap to finish;
information that preserves only a *subset/subspace* is not.

## Suggested next experiments
(see also NOTES.md "Planned follow-ups")
- **Wider budgets.** Re-run with 20–40% budgets to convert the many near-miss DNRs into a smooth
  recovery-cost curve and find each method's true recovery point.
- **Structure-preserving retraining (option-b).** Keep the sparsity mask / low-rank factorization
  fixed during retraining; pruning may recover much better when not forced dense-from-init.
- **Heterogeneous bit allocation** across layers (sensitivity-weighted), and better importance
  methods (GraSP, OBS, Wanda) for the pruning families.
- **Multi-seed error bars** to quantify the noise around the threshold (several runs sit within
  ±0.1 pp of the bar, so a single seed can flip recover/DNR).

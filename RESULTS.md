# RESULTS — weight-compression recovery

**Clickable report:** https://jacobcd52.github.io/weight-compression-recovery/
**Figure:** `figures/pareto.png` · **Data:** `results/summary.csv`

## v2 (current, 2026-05-26): undertrained baseline · test-loss recovery · full AQLM · amortized ratio

**Regime changes** (supersede the v1 sections below): baseline retrained to **50 epochs**
(undertrained → not overfit: test loss **0.401**, acc 88.1%); recovery now = retrain until
full-test loss drops **below the baseline's 0.401** (strict); retrain warmup cut to 100 steps;
**full AQLM** implemented (activation-aware, per-layer codebooks — see below); sweep restricted
to the promising families (quantize, kmeans, magnitude_prune, magprune_quant, additive_vq, aqlm).
**29 runs.** Live report has two Pareto plots (honest bytes + amortized) and the baseline curve.

**Findings:**
- **Recovery cost is ~constant (~7% of baseline) for everything that recovers.** It is set by
  *when the cosine schedule lets the loss dip below the baseline* (≈70% through the 10% budget),
  not by the compression method. So the meaningful axis here is the **most compression that still
  recovers**, not the cost — the frontier is essentially horizontal.
- **Recoverable frontier reaches ~32×** (amortized ratio ≈0.031): quantize_1, additive_vq_1_8_8,
  and **aqlm_2_4_8** all recover (~8.5% cost). Below ~32× everything is DNR within the 10% budget.
- **Amortized ratio (codebook/scale overhead removed) is what makes AQLM competitive.** On honest
  bytes, faithful per-layer AQLM codebooks are a heavy fixed tax on a 270k-param model
  (aqlm_2_8_8 full ratio 0.218); **amortized, it drops to 0.0625 (16×) and recovers at ~7%** —
  on par with k-means/additive_vq/quantize at the same amortized ratio. i.e. AQLM is *not* worse;
  the tiny model just penalizes its codebooks. This is direct evidence its advantage should appear
  at the ~50M-LLM scale where the codebook tax amortizes away.
- **The VQ/quant family is essentially tied at this scale** — quantize, kmeans, additive_vq, and
  AQLM all recover down to ~16–32× (amortized), with AQLM's extra activation-aware machinery giving
  no clear edge on a 270k-param net (low redundancy). magprune_quant is all-DNR (the pruning step
  dominates); magnitude_prune only recovers at ≥0.25 keep.

**AQLM implementation note (transparency):** per-layer codebooks, per-output scales, residual-kmeans
init, alternating weighted-LS codebook updates, and **activation-aware weighting via the *diagonal*
of the layer Hessian** (im2col for conv). Two deviations from full AQLM remain: (1) diagonal — not
full off-diagonal — Hessian, and (2) weighted-greedy — not beam-search — code assignment. Given
AQLM shows no edge here even amortized, these refinements are unlikely to change the small-model
conclusion; they matter more at LLM scale.

---

## Update (2026-05-26): AQLM reproduction + fine-grained recovery sweep (v1.5, accuracy metric)

Reproduced the headline method of *"Aggressive Compression Enables LLM Weight Theft"* —
**additive multi-codebook vector quantization (AQLM-style)** — on ResNet-20, with shared global
codebooks (RVQ init + least-squares codebook refinement), and re-ran the recovery axis with a
**fine eval cadence (every ~0.1% of baseline steps)** to resolve cheap recoveries.

**Findings:**
- **Additive VQ recovers up to ~14× within the 10% budget** (additive_vq 7×/9×/14× recover at
  7.5–8.7% cost) but is **DNR at 28×+** — and crucially, on this small model it is *competitive
  with but not better than* our existing k-means / scalar quantization (kmeans_2 already recovers
  at 16×). The paper's large AQLM gains are at LLM scale; at 270k params the simpler distribution-
  preserving methods are just as good.
- **Recovery cost is bimodal.** The fine cadence shows near-lossless methods (quantize_8 at 4×,
  kmeans_6 at 5×) recover in **<0.1%** of baseline compute (their reconstructed init is already
  ~89–90%), while aggressive methods need **~7–9%**. The **0.1–1% region is essentially empty** —
  recovery is closer to all-or-nothing than a smooth curve. (This is what the original epoch-
  granularity sweep hid.)
- **Beyond ~16× compression, plain-CE recovery needs >10% budget.** additive_vq 28×/31× plateau at
  89.3% (just under the 89.58% bar) even at full budget; only distillation previously reached 32×
  (quantize_1_distill). Pushing the high-compression frontier further would mean raising the budget
  cap above 10%, not staying at the paper's ~0.01–1% levels.

**Scaling to 1.5B (e.g. Qwen2.5-1.5B):** recovery-retraining GPU compute dominates and scales as
6·N·(budget × pretrain_tokens). Rough per-sweep cost on H100: ~$0.5–6k at a ≤0.1% budget cap,
but ~$56k / ~940 GPU-days at a 1% cap across ~40 runs — recommend tiering (bulk at ≤0.1%, spot-
check 1% on the few most-compressed knobs). Needs 80 GB GPUs (optimizer states), and the LLM auto-
search becomes GPU-bound (evals cost hours–days), so it would need a small-model proxy.

---

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

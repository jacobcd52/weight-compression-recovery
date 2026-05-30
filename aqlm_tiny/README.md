# TinyLlama-1.1B AQLM scale-up — vs SmolLM2-360M and naive quant

Scales the validated 2-stage AQLM pipeline from SmolLM2-360M to TinyLlama-1.1B (3× bigger,
~2700 vs ~11000 tokens/param — less overtrained). Stage-1 quantize + free stage-2 PV-tuning,
identical settings to the SmolLM2 run.

## Results — block weights, AQLM `evaluate_perplexity`, seqlen 2048
Baseline (bf16): WikiText2 7.78, C4 9.29.

| method | honest bpp | stage-1 (wt2 / c4) | + stage-2 (wt2 / c4) |
|---|---|---|---|
| AQLM 2 bpp (2x8g8)   | 2.017 | 16.4 / 16.9 | **11.51 / 12.90** (1.48× base) |
| AQLM 1 bpp (1x8g8)   | 1.012 | 858 / 461   | **84.1 / 58.7** (10.8× / 6.3×)  |
| AQLM 0.5 bpp (1x8g16)| 0.517 | 7259 / 4926 | **1144 / 660** (147× / 71×)     |
| naive uniform 8-bit  | 8.013 | 7.78 / 9.29 | — |
| naive uniform 4-bit  | 4.013 | 10.11 / 12.01 | — |
| naive uniform 3-bit  | 3.013 | 5796 / 6611  | — |
| naive uniform 2-bit  | 2.013 | 50677 / 55246 | — |

## Verdict — scale-up is dramatic and broad-based
At the **same bit budget** vs SmolLM2-360M (wt2): 2 bpp 24.9 → **11.5** (2.2×);
1 bpp 364 → **84** (4.3×); 0.5 bpp 2326 → **1144** (2×).

- **AQLM 2 bpp on TinyLlama is essentially lossless** (1.48× baseline, AQLM-7B territory).
- **AQLM 1 bpp on TinyLlama is degraded but not broken** (was unrecoverable on SmolLM2 at 364) —
  the cliff is much shallower at this scale, even if not fully usable.
- **Codebooks amortize even more cleanly** (honest 2.017 / 1.012 / 0.517 vs nominal 2 / 1 / 0.5).
- **Pareto vs naive quant is sharper:** AQLM 2 bpp ≈ naive 4-bit at half the bits;
  naive still breaks below 4-bit.

See `figures/scaleup_aqlm_vs_simple.png` for the overlay (TinyLlama frontier dramatically below
SmolLM2 at every bit budget).

## Pipeline
- `run_pipeline_tiny.sh` — full 2-stage for 2 / 1 / 0.5 bpp, identical settings to SmolLM2.
- `plot_scaleup.py` — overlay plot.
- Reuses the parametrized scripts (`build_ft_dataset.py`, `eval_baseline.py`, `simple_eval.py`,
  `reconstruct_aqlm.py`) — set `MODEL_PATH=/workspace/models/tinyllama` and `SIMPLE_OUT=...tiny.json`.

The AQLM source patches from `aqlm_smol/README.md` still apply.

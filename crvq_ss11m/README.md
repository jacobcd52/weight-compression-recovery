# CRVQ on SimpleStories-11M — extreme 1-bit compression at the smallest viable scale

Applies CRVQ (Channel-Relaxed Vector Quantization, arXiv 2412.09282, [xuyuzhuang11/CRVQ](https://github.com/xuyuzhuang11/CRVQ),
TACL 2025) to the smallest Llama-architecture model we have. CRVQ is the published direct successor
to AQLM that claims a 38.9% perplexity reduction at 1-bit on Llama-2-7B vs plain AQLM.

## Result
- Stage-1 only (CRVQ quantization, no finetune): ppl = **15.64** on SimpleStories eval
- Stage-2 (KL-distillation finetune, 2 epochs): ppl = **13.85**
- Avg honest bits/param: **1.24** (codebook overhead higher than paper's 1.06 due to small-scale amortization, same story as 11M AQLM result)
- Baseline (uncompressed 11M): ppl ≈ 7.08 → CRVQ stage-2 is **1.96× baseline**
- vs plain AQLM 1 bpp (16.7) and AQLM 2 bpp (8.14) on the same 11M model: CRVQ falls between

The paper's 39% improvement (Llama-2-7B: AQLM 15.25 → CRVQ 9.81 at ~1 bpp) doesn't fully transfer
to 11M scale because the multibook critical-channel codebooks have to be shrunk from 256 to 16
centroids — the data floor for kmeans is set by `num_out_groups × group_bound`, which is too small
for `k_proj`/`v_proj` (out=128 due to GQA) on a 384-dim model. See "deviations" below.

## Why the 11M is below the algorithm's design floor (and how we adapted)

Paper recipe needs `num_out_groups × group_bound ≥ ~10 × 2^nbits` data vectors per kmeans, so for
nbits=8 (256 centroids) needs ~2560 vectors. 11M's k_proj/v_proj has only 128 out groups (GQA, 2 KV
heads × 64 head_dim). Even with `multibook_ratio` bumped to 0.025 to get `group_bound=1`, that's
only 128 data vectors — way under 256 centroids. So we reduced critical-codebook nbits to 4 (16
centroids each), keeping the method spirit (4 codebooks, channel relaxation) but with less granular
critical-channel fitting.

For paper-default `multibook_ratio=0.02` + `nbits=8 8 8 8`, the smallest model where CRVQ literally
runs is **TinyLlama-1.1B**. At 360M / 135M / 11M, you need either nbits reduction or a different
quant scheme.

## Deviations from paper
1. Model: SimpleStories-11M (vs Llama-2-7B)
2. Calibration data: SimpleStories train (vs RedPajama-1T-Sample) — tokenizer vocab 4096 can't handle RedPajama
3. Stage-2 finetune data: SimpleStories train (different slice; vs RedPajama)
4. Eval ppl: SimpleStories eval (vs WikiText-2)
5. `model_seqlen=512` (vs 4096)
6. `multibook_ratio=0.025` (vs 0.02) — `group_bound=0` on tiny layers otherwise
7. `nbits_per_codebook=8 4 4 4` (vs 8 8 8 8) — k_proj/v_proj kmeans data floor
8. Source patches in `/workspace/projects/CRVQ/`:
   - `src/datautils.py` `get_c4` train branch: `>= seqlen` → `> seqlen` (randrange(0,-1) crash on exact-length docs)
   - `src/finetune.py:260` + `main.py:396`: guard `attention_mask.to(...)` against None (small model + transformers 4.37)
   - `main.py:376`: guard `Avg_bits` divide-by-zero when resume loads all pre-quantized layers
   - `main.py:932`: env-var `AQLM_EVAL_PATH` override so eval uses our pre-tokenized SS file (the 11M tokenizer class can't be loaded by AutoTokenizer)

## Files
- `run_ss11m_crvq.sh` — full pipeline (stage-1 + stage-2)
- `build_ss_data.py` — builds the SimpleStories `.pth` calibration/finetune files in CRVQ format
- `results.txt` — final ppls + avg bits

## Pipeline ran
`exp ss11m_crvq6` job (after fixes); stage-1 took ~10 min on A5000, stage-2 took ~2 min.

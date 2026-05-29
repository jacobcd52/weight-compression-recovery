# AQLM on SmolLM2-360M — honest compression frontier vs naive quant

Tests whether AQLM (additive-quantization VQ, Vahe1994/AQLM) is a Pareto improvement over naive
per-row uniform quantization at a real ~360M scale, where codebook overhead amortizes (unlike the
11M model, where fp16 embeddings + codebooks floored the honest ratio at ~0.1).

## Threat-model framing
AQLM is a **two-stage** method:
- **stage-1** (`main.py`): per-layer quantization + per-block codebook tuning.
- **stage-2** (`finetune.py`, PV-tuning): global end-to-end CE-vs-teacher tuning of the codebooks.

Stage-2 is **attacker-side, in-datacenter prep done *before* exfiltration** — it only tunes the
continuous codebooks (the bit budget is unchanged), so it is FREE and does **not** count toward
recovery cost. The exfiltrated artifact is the post-stage-2 codebooks.

## Results (WikiText2 ppl, AQLM `evaluate_perplexity`, seqlen 2048; block weights only)
Baseline (bf16) = 11.47.

| method | honest block bpp | stage-1 | + free stage-2 |
|---|---|---|---|
| AQLM 2 bpp (2x8g8)   | 2.06 | 99.0   | **24.9** (2.2× base) |
| AQLM 1 bpp (1x8g8)   | 1.04 | 31,698 | 364 |
| AQLM 0.5 bpp (1x8g16)| 0.56 | 40,943 | 2,326 |
| naive uniform 8-bit  | 8.03 | 11.5   | — |
| naive uniform 4-bit  | 4.03 | 21.8   | — |
| naive uniform 3-bit  | 3.03 | 26,849 | — |
| naive uniform 2-bit  | 2.03 | 2.9M   | — |

**Verdict:** AQLM 2 bpp matches naive 4-bit quality at HALF the bits; naive quant breaks below 4
bit. AQLM extends the usable frontier to ~2 bpp. Below 2 bpp AQLM also collapses on this (heavily
overtrained) model. Codebooks amortize: honest bpp ≈ nominal. See `figures/smol_aqlm_vs_simple.png`.

## AQLM source patches required (in the /workspace/projects/AQLM clone)
These are NOT in upstream Vahe1994/AQLM and must be re-applied if the clone is recreated:
1. `src/datautils.py` get_c4 train branch: `if trainenc.input_ids.shape[1] >= seqlen:` → `> seqlen`
   (lines ~33/91/157) — avoids `randrange(0,0,0)` when a doc tokenizes to exactly seqlen.
2. `main.py` after ~line 898: `if os.environ.get("AQLM_EVAL_PATH"): datasets=[os.environ["AQLM_EVAL_PATH"]]`.
3. `src/modelutils.py` end of `get_model`: untie input/output embeddings when
   `config.tie_word_embeddings` (clone lm_head.weight, set tie=False). FSDP's gradient writeback
   crashes on a tied embed/lm_head shared flat param ("Cannot writeback when the gradient shape
   changes") — e.g. SmolLM2. Values unchanged.

## Pipeline
- `build_ft_dataset.py` — pre-tokenize one cached c4 shard into 2048-blocks for finetune.py.
- `run_ft_2bpp.sh` — stage-2 PV-tuning of the 2 bpp model (single-GPU via torch.distributed.run, eager attn).
- `run_pipeline_remaining.sh` — stage-1 + stage-2 for 1 bpp and 0.5 bpp.
- `eval_baseline.py` / `simple_eval.py` — baseline + naive-quant ppl with AQLM's exact eval.
- `plot_smol_compare.py` — the comparison figure.

Models live at /workspace/models/smollm2-360m; quantized runs at /workspace/projects/AQLM/runs/aqlm_smol_*.

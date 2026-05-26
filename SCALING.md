# SCALING — cost of running this on a 1.5B model (e.g. Qwen2.5-1.5B)

At ResNet-20 scale the whole 56-run sweep cost ~1.4 GPU-hr (~$0.50) and tokens would have been
the only real cost. **At 1.5B the opposite holds: recovery-retraining GPU compute dominates**,
scaling as `C_recover = 6 · N · D_budget`, where `D_budget = budget% × pretrain_tokens`.

Assumptions: Qwen2.5-1.5B pretrained on ~18T tokens → `C_pretrain ≈ 6·1.5e9·18e12 ≈ 1.6e23` FLOPs.
Realistic ~45% MFU. Community/spot rates: H100-80G ~$2.5/hr, A100-80G ~$1.5/hr. (Qwen2-1.5B was
~7T tokens → halve everything. On-demand H100 ~$3–4/hr → +20–60%.) Order-of-magnitude only.

## Per recovery run (one knob, consuming the full budget)
| budget | tokens | H100 | A100 | $ (H100) |
|---|---|---|---|---|
| 0.01% (paper's level) | 1.8B | 11 hr | 32 hr | ~$28 |
| 0.1% | 18B | 4.7 days | 13 days | ~$280 |
| 1% (high-compression target) | 180B | 47 GPU-days | 134 days | ~$2,800 |

## Full sweep (~40 technique×knob runs, ~50% of cap consumed on average)
| budget cap | total GPU | cost (H100) |
|---|---|---|
| 0.01% | ~225 H100-hr (~9 GPU-days) | ~$560 |
| 0.1% | ~2,250 H100-hr (~94 GPU-days) | ~$5,600 |
| 1% | ~22,500 H100-hr (~940 GPU-days) | ~$56,000 |

## Other cost lines (minor next to retraining)
- **Compression step:** additive-VQ codebook fitting over ~1.5B weights needs GPU k-means
  (faiss/torch, not sklearn-CPU); one-time ~minutes–1 hr per knob. Negligible.
- **Data:** need a real corpus for the retraining tokens. 0.01% = 1.8B tokens (~7 GB, trivial);
  1% = 180B tokens (~700 GB) — a genuine streaming data pipeline (FineWeb-style).
- **LLM auto-search tokens:** unchanged by model size (~$50–300/run for proposals), BUT each eval
  is now hours–days of GPU instead of seconds, so the cheap-eval assumption breaks — you'd have to
  search on a small proxy and validate the best candidates on 1.5B.

## Caveats
- **Memory:** full AdamW recovery of 1.5B needs ~21 GB (weights + fp32 master + optimizer states),
  so a 24 GB 4090 won't do it without 8-bit Adam / offload / FSDP — plan on 80 GB cards.
- **Wall-clock:** a single 1%-budget run is ~47 days on one H100 → needs data-parallel
  (8× H100 ≈ 6 days/run). This is a small-cluster job, not a single pod.

## Recommendation
Tier the budget axis: run the bulk of the sweep at **≤0.1% (~$0.5–6k)** — already 10× the paper's
recovery budget and enough to map most of the frontier — and **spot-check 1% only on the few most-
compressed knobs** that are DNR-but-close at 0.1%. A full 1%-cap sweep (~$56k, ~940 GPU-days) wastes
most compute on runs that recover far below 1% or never recover.

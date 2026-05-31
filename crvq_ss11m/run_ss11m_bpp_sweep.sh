#!/bin/bash
# Phase 1: sweep nbits_per_codebook[0] in {2,3,4,6,8} with structure [X 4 4 4], all else fixed.
# Each point = full pipeline (stage-1 + stage-2 with lr=1e-3). 5 runs in parallel on one A5000.
set -e
cd /workspace/projects/CRVQ
export VIRTUAL_ENV=/workspace/envs/crvq
export AQLM_EVAL_PATH=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
PY=/workspace/envs/crvq/bin/python
MODEL=/workspace/models/simplestories11m
CALIB=/workspace/projects/CRVQ/data/ss11m_crvq_calib.pth
FT_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_ft.pth
EVAL_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
LOG_DIR=/workspace/projects/CRVQ/runs/bpp_sweep_logs
mkdir -p "$LOG_DIR"

run_one() {
  local X=$1
  local TAG="X${X}_X444"
  local SAVE_S1=/workspace/projects/CRVQ/runs/bpp_${TAG}
  local SAVE_S2=/workspace/projects/CRVQ/runs/bpp_${TAG}_ft
  mkdir -p "$SAVE_S1" "$SAVE_S2"
  echo "[start X=$X tag=$TAG] $(date '+%H:%M:%S')"
  {
    $PY main.py "$MODEL" "$CALIB" \
      --nsamples=2048 --val_size=256 --model_seqlen=512 \
      --num_codebooks=4 --nbits_per_codebook $X 4 4 4 \
      --multibook_ratio=0.025 --shuffle_rule=3 \
      --out_group_size=1 --in_group_size=8 \
      --beam_size=1 --relative_mse_tolerance=0.01 \
      --max_epochs=50 --finetune_lr=3e-5 \
      --finetune_adam_beta1=0.90 --finetune_adam_beta2=0.95 \
      --finetune_keep_best --finetune_batch_size=128 --local_batch_size=4 \
      --finetune_max_epochs=10 --finetune_early_stop=3 \
      --offload_activations --save "$SAVE_S1" --resume
    $PY finetune.py \
      --base_model "$MODEL" --quant_model "$SAVE_S1" --dataset "$FT_DATA" \
      --model_seqlen=512 --eval_model_seqlen=512 \
      --eval_datasets "$EVAL_DATA" --nsamples=1024 --val_size=64 \
      --lr=1e-3 --adam_beta1=0.90 --adam_beta2=0.999 \
      --epochs=10 --early_stop=5 --batch_size=16 --microbatch_size=1 \
      --save "$SAVE_S2" --gradient_checkpointing --amp --device_map auto
  } > "$LOG_DIR/${TAG}.log" 2>&1
  echo "[done  X=$X tag=$TAG] $(date '+%H:%M:%S')"
}

run_one 2 & run_one 3 & run_one 4 & run_one 6 & run_one 8 &
wait
echo "ALL_BPP_DONE"

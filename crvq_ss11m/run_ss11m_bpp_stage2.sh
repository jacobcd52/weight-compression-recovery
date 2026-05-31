#!/bin/bash
# Phase 1 stage-2 only: 5 parallel finetunes on the already-saved stage-1 models.
# lr=1e-3 (best from LR sweep), 10 epochs.
cd /workspace/projects/CRVQ
export VIRTUAL_ENV=/workspace/envs/crvq
export AQLM_EVAL_PATH=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
PY=/workspace/envs/crvq/bin/python
MODEL=/workspace/models/simplestories11m
FT_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_ft.pth
EVAL_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
LOG_DIR=/workspace/projects/CRVQ/runs/bpp_stage2_logs
mkdir -p "$LOG_DIR"

run_ft() {
  local X=$1
  local TAG="X${X}_X444"
  local QUANT=/workspace/projects/CRVQ/runs/bpp_${TAG}
  local SAVE=/workspace/projects/CRVQ/runs/bpp_${TAG}_ft
  mkdir -p "$SAVE"
  echo "[start X=$X] $(date '+%H:%M:%S')"
  $PY finetune.py \
    --base_model "$MODEL" --quant_model "$QUANT" --dataset "$FT_DATA" \
    --model_seqlen=512 --eval_model_seqlen=512 \
    --eval_datasets "$EVAL_DATA" --nsamples=1024 --val_size=64 \
    --lr=1e-3 --adam_beta1=0.90 --adam_beta2=0.999 \
    --epochs=10 --early_stop=5 --batch_size=16 --microbatch_size=1 \
    --save "$SAVE" --gradient_checkpointing --amp --device_map auto \
    > "$LOG_DIR/${TAG}.log" 2>&1
  echo "[done  X=$X] $(date '+%H:%M:%S')"
}

run_ft 2 & run_ft 3 & run_ft 4 & run_ft 6 & run_ft 8 &
wait
echo "ALL_STAGE2_DONE"

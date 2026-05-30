#!/bin/bash
# Extension of the LR sweep: higher LRs (1e-3, 3e-3, 1e-2) to find where it breaks.
# Same params as run_ss11m_lr_sweep.sh otherwise.
set -e
cd /workspace/projects/CRVQ
export VIRTUAL_ENV=/workspace/envs/crvq
export AQLM_EVAL_PATH=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
PY=/workspace/envs/crvq/bin/python
MODEL=/workspace/models/simplestories11m
QUANT_MODEL=/workspace/projects/CRVQ/runs/ss11m_8444_025
FT_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_ft.pth
EVAL_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
LOG_DIR=/workspace/projects/CRVQ/runs/lr_sweep_logs
mkdir -p "$LOG_DIR"

run_lr() {
  local LR=$1
  local TAG=$(echo "$LR" | tr '.' 'p' | tr '-' 'm')
  local SAVE=/workspace/projects/CRVQ/runs/ss11m_ft_lr_${TAG}
  echo "[start lr=$LR tag=$TAG] $(date '+%H:%M:%S')"
  $PY finetune.py \
    --base_model "$MODEL" \
    --quant_model "$QUANT_MODEL" \
    --dataset "$FT_DATA" \
    --model_seqlen=512 \
    --eval_model_seqlen=512 \
    --eval_datasets "$EVAL_DATA" \
    --nsamples=1024 \
    --val_size=64 \
    --lr=$LR \
    --adam_beta1=0.90 \
    --adam_beta2=0.999 \
    --epochs=10 \
    --early_stop=5 \
    --batch_size=16 \
    --microbatch_size=1 \
    --save "$SAVE" \
    --gradient_checkpointing \
    --amp \
    --device_map auto \
    > "$LOG_DIR/${TAG}.log" 2>&1
  echo "[done  lr=$LR tag=$TAG] $(date '+%H:%M:%S')"
}

run_lr 1e-3 &
run_lr 3e-3 &
run_lr 1e-2 &
wait
echo "ALL_LR_DONE"

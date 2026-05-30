#!/bin/bash
# Parallel LR sweep of CRVQ stage-2 (KL distillation) on SimpleStories-11M.
# Reuses the already-quantized stage-1 model at runs/ss11m_8444_025/.
# 4 LRs × 10 epochs each, all 4 running concurrently on the single GPU.
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
  local TAG=$(echo "$LR" | tr '.' 'p' | tr '-' 'm')   # e.g. 3e-5 -> 3em05
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

# Launch 4 lr's in parallel
run_lr 1e-5 &
run_lr 3e-5 &
run_lr 1e-4 &
run_lr 3e-4 &
wait
echo "ALL_LR_DONE"

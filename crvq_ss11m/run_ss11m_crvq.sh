#!/bin/bash
# CRVQ paper-exact recipe applied to SimpleStories-11M.
# Diff from run_CRVQ.sh (paper's Llama-2-7B): model path, dataset path (pre-tokenized
# SimpleStories instead of RedPajama .pth — forced by vocab=4096 tokenizer mismatch),
# model_seqlen 512 (model's max ctx). All other hparams identical.
set -e
cd /workspace/projects/CRVQ
export VIRTUAL_ENV=/workspace/envs/crvq
export OMP_NUM_THREADS=8
# patch: tell main.py + finetune.py to eval on our pre-tokenized SimpleStories eval file
# (the 11M tokenizer can't be loaded by AutoTokenizer so we can't use wikitext2/c4)
export AQLM_EVAL_PATH=/workspace/projects/CRVQ/data/ss11m_crvq_eval.pt
PY=/workspace/envs/crvq/bin/python
MODEL=/workspace/models/simplestories11m
CALIB=/workspace/projects/CRVQ/data/ss11m_crvq_calib.pth
FT_DATA=/workspace/projects/CRVQ/data/ss11m_crvq_ft.pth
STAGE1_SAVE=/workspace/projects/CRVQ/runs/ss11m_8444_025
STAGE2_SAVE=/workspace/projects/CRVQ/runs/ss11m_8444_025_ft
mkdir -p "$STAGE1_SAVE" "$STAGE2_SAVE"

echo "============================== STAGE 1 (CRVQ quantization) =============================="
$PY main.py \
    "$MODEL" \
    "$CALIB" \
    --nsamples=2048 \
    --val_size=256 \
    --model_seqlen=512 \
    --num_codebooks=4 \
    --nbits_per_codebook 8 4 4 4 \
    --multibook_ratio=0.025 \
    --shuffle_rule=3 \
    --out_group_size=1 \
    --in_group_size=8 \
    --beam_size=1 \
    --relative_mse_tolerance=0.01 \
    --max_epochs=50 \
    --finetune_lr=3e-5 \
    --finetune_adam_beta1=0.90 \
    --finetune_adam_beta2=0.95 \
    --finetune_keep_best \
    --finetune_batch_size=128 \
    --local_batch_size=4 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --offload_activations \
    --save "$STAGE1_SAVE" \
    --resume

echo "============================== STAGE 2 (KL-distillation finetune) =============================="
$PY finetune.py \
    --base_model "$MODEL" \
    --quant_model "$STAGE1_SAVE" \
    --dataset "$FT_DATA" \
    --model_seqlen=512 \
    --eval_model_seqlen=512 \
    --eval_datasets /workspace/projects/CRVQ/data/ss11m_crvq_eval.pt \
    --nsamples=1024 \
    --val_size=64 \
    --lr=1e-5 \
    --adam_beta1=0.90 \
    --adam_beta2=0.999 \
    --epochs=2 \
    --early_stop=3 \
    --batch_size=16 \
    --microbatch_size=1 \
    --save "$STAGE2_SAVE" \
    --gradient_checkpointing \
    --amp \
    --device_map auto

echo "PIPELINE_ALLDONE"

#!/bin/bash
# Full 2-stage AQLM pipeline on TinyLlama-1.1B for 2 / 1 / 0.5 bpp.
# Settings identical to the SmolLM2-360M run for direct comparison.
#   stage-1 = quantize, stage-2 = global PV-tuning (free attacker prep).
set -e
cd /workspace/projects/AQLM
export VIRTUAL_ENV=/workspace/envs/aqlm
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=8
PY=/workspace/envs/aqlm/bin/python
MODEL=/workspace/models/tinyllama
DS=/workspace/projects/AQLM/c4_tiny_ft_ds

stage1() {  # $1=num_codebooks $2=nbits $3=in_group_size $4=name
  echo "===== STAGE1 $4 ====="
  $PY main.py $MODEL c4 --nsamples=512 --val_size=64 \
    --num_codebooks=$1 --nbits_per_codebook=$2 --in_group_size=$3 \
    --relative_mse_tolerance=0.01 --finetune_batch_size=32 --finetune_max_epochs=3 \
    --finetune_early_stop=2 --finetune_keep_best --local_batch_size=4 \
    --model_seqlen=2048 --dtype=bfloat16 --attn_implementation=eager \
    --save runs/aqlm_tiny_$4
}

stage2() {  # $1=name
  echo "===== STAGE2 $1 ====="
  $PY -m torch.distributed.run --nproc-per-node=1 --master-port=29557 finetune.py \
    --base_model $MODEL \
    --quantized_model runs/aqlm_tiny_$1 \
    --model_seqlen 2048 --block_type LlamaDecoderLayer \
    --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype uint16 \
    --dataset_name $DS --split none --seed 42 \
    --trust_remote_code --update_codes --update_codebooks_and_scales \
    --lamb --debias --lr 3e-4 --max_code_change_per_step 1e-2 --code_lr 1e-2 \
    --beam_size 5 --delta_decay 0 --batch_size 32 --microbatch_size 1 --max_epochs 1 \
    --gradient_checkpointing --print_every_steps 1 --eval_every_steps 5 \
    --keep_best_model --eval_datasets wikitext2 c4 --attn_implementation eager \
    --save runs/aqlm_tiny_${1}_ft
}

echo "########## 2 bpp ##########"; stage1 2 8 8  tiny_2x8g8;  stage2 tiny_2x8g8
echo "########## 1 bpp ##########"; stage1 1 8 8  tiny_1x8g8;  stage2 tiny_1x8g8
echo "########## 0.5 bpp ##########"; stage1 1 8 16 tiny_1x8g16; stage2 tiny_1x8g16
echo "PIPELINE_ALLDONE"

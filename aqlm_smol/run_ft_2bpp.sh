#!/bin/bash
# AQLM stage-2: global end-to-end PV-tuning of the 2bpp SmolLM2-360M quantized model.
# This is attacker-side, in-datacenter prep (does NOT count as recovery cost).
set -e
cd /workspace/projects/AQLM
export VIRTUAL_ENV=/workspace/envs/aqlm
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=8
/workspace/envs/aqlm/bin/python -m torch.distributed.run --nproc-per-node=1 --master-port=29555 finetune.py \
  --base_model /workspace/models/smollm2-360m \
  --quantized_model runs/aqlm_smol_smol_2x8g8 \
  --model_seqlen 2048 \
  --block_type LlamaDecoderLayer \
  --load_dtype bfloat16 \
  --amp_dtype bfloat16 \
  --code_dtype uint16 \
  --dataset_name /workspace/projects/AQLM/c4_smol_ft_ds \
  --split none \
  --seed 42 \
  --trust_remote_code \
  --update_codes \
  --update_codebooks_and_scales \
  --lamb --debias \
  --lr 3e-4 \
  --max_code_change_per_step 1e-2 \
  --code_lr 1e-2 \
  --beam_size 5 \
  --delta_decay 0 \
  --batch_size 32 \
  --microbatch_size 1 \
  --max_epochs 1 \
  --gradient_checkpointing \
  --print_every_steps 1 \
  --eval_every_steps 5 \
  --keep_best_model \
  --eval_datasets wikitext2 c4 \
  --attn_implementation eager \
  --save runs/aqlm_smol_smol_2x8g8_ft

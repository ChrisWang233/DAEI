#!/bin/bash
# SURE_CE Baseline: train DAEI on noisy embeddings (no DAE)
# The first run creates this reusable cache: data/dataset_cache/nq_noisy_-1_gtr_base_train.arrow

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# Adjust the batch size for available GPU memory; 64 fits on a 24GB GPU.
BATCH_SIZE=${BATCH_SIZE:-256}

nohup python -m DAEI.run \
  --experiment inversion \
  --model_name_or_path t5-base \
  --embedder_model_name gtr_base \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --use_frozen_embeddings_as_input True \
  --dataset_name nq,msmarco,yahoo \
  --dataset_mode noisy \
  --embedder_gaussian_noise_level 0.01 \
  --learning_rate 0.0005 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length 128 \
  --num_train_epochs 30 \
  --warmup_steps 20000 \
  --eval_steps 50000 \
  --save_steps 4000 \
  --output_dir saves/t1 \
  --max_eval_samples 500 \
  --bf16=1 \
  --use_less_data 7000000 > logs/t1.log &

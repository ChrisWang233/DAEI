#!/bin/bash


set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# Adjust the batch size for available GPU memory; 64 fits on a 24GB GPU.
BATCH_SIZE=${BATCH_SIZE:-64}

nohup python -m DAEI.run \
  --experiment inversion \
  --model_name_or_path t5-base \
  --embedder_model_name gtr_base \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --use_frozen_embeddings_as_input True \
  --dataset_name nq,msmarco,yahoo \
  --learning_rate 0.001 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length 128 \
  --num_train_epochs 50 \
  --warmup_steps 20000 \
  --eval_steps 80000 \
  --save_steps 5000 \
  --output_dir saves/test \
  --max_eval_samples 1000 \
  --bf16=1 \
  --use_less_data 7000000 > logs/test.log 2>&1 &

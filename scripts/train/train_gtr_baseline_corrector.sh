#!/bin/bash
# Train the original Corrector from the baseline inversion model for comparison with the DAE Corrector.
# Requires a trained baseline at saves/baseline or saves/baseline/checkpoint-xxx.
# The first run precomputes hypotheses and stores them under DAEI_CACHE.
#
# Memory notes:
#   - Reduce the original batch size of 128 on GPUs smaller than 80GB.
#   - Gradient accumulation preserves the effective batch size.
#   - Gradient checkpointing reduces activation memory.
#   - Use --max_steps when resuming to avoid epoch-count mismatches.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# Use the final baseline or a specific checkpoint.
BASELINE_CKPT="${BASELINE_CHECKPOINT:-saves/baseline}"
# Cache directory for precomputed Corrector hypotheses.
export DAEI_CACHE="${DAEI_CACHE:-$PROJECT_ROOT/data/corrector_cache}"
mkdir -p "$DAEI_CACHE"

# Use gradient accumulation to compensate for a smaller per-device batch.
BATCH_SIZE=${BATCH_SIZE:-16}


nohup python -m DAEI.run \
  --experiment corrector_encoder \
  --corrector_model_from_pretrained "$BASELINE_CKPT" \
  --model_name_or_path t5-base \
  --embedder_model_name gtr_base \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --use_frozen_embeddings_as_input True \
  --dataset_name nq,msmarco \
  --use_less_data 5000000 \
  --learning_rate 0.001 \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --max_seq_length 128 \
  --max_steps $MAX_STEPS \
  --warmup_steps 20000 \
  --eval_steps 100000 \
  --save_steps 4000 \
  --output_dir saves/corrector_baseline111 \
  --max_eval_samples 500 \
  --bf16=1 \
  > logs/corrector_baseline111.log 2>&1 &

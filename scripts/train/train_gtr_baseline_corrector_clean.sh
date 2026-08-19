#!/bin/bash
# ==========================================================================
# Baseline-Clean Pipeline - Corrector Training
#
# Train a vanilla Corrector on top of the clean-embedding baseline inverter.
# This is the non-DAE baseline counterpart to the DAEI corrector runs.
#
# Environment overrides:
#   BASELINE_CLEAN_CHECKPOINT       default: saves/baseline_clean if it has
#                                   weights, otherwise latest checkpoint
#   OUTPUT_DIR                      default: saves/baseline_clean_corrector
#   LOG_FILE                        default: logs/baseline_clean_corrector.log
#   DAEI_CACHE                  default: data/corrector_cache
#   BATCH_SIZE                      default: 32
#   GRADIENT_ACCUMULATION_STEPS     default: 2
#   NPROC                           default: auto-detect visible GPUs
# ==========================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

BASELINE_ROOT="${BASELINE_ROOT:-saves/baseline_clean}"
if [[ -n "${BASELINE_CLEAN_CHECKPOINT:-}" ]]; then
  BASELINE_CKPT="$BASELINE_CLEAN_CHECKPOINT"
elif [[ -f "$BASELINE_ROOT/model.safetensors" || -f "$BASELINE_ROOT/pytorch_model.bin" ]]; then
  BASELINE_CKPT="$BASELINE_ROOT"
else
  BASELINE_CKPT="$(find "$BASELINE_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
fi

if [[ -z "$BASELINE_CKPT" || ! -d "$BASELINE_CKPT" ]]; then
  echo "ERROR: baseline clean checkpoint not found. Set BASELINE_CLEAN_CHECKPOINT." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-saves/baseline_clean_corrector}"
LOG_FILE="${LOG_FILE:-logs/baseline_clean_corrector.log}"

export DAEI_CACHE="${DAEI_CACHE:-$PROJECT_ROOT/data/corrector_cache}"
mkdir -p "$DAEI_CACHE"

BATCH_SIZE=${BATCH_SIZE:-96}
GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS:-2}
DATASET_NAME="${DATASET_NAME:-nq,msmarco,yahoo}"
USE_LESS_DATA="${USE_LESS_DATA:-7000000}"

if [[ -z "${NPROC+x}" || -z "$NPROC" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -cve '^$' || true)
    [[ -z "$NPROC" || "$NPROC" -lt 1 ]] && NPROC=1
  else
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    NPROC=$((NPROC + 0))
    [[ "$NPROC" -lt 1 ]] && NPROC=1
  fi
fi

echo "[baseline_clean_corrector] NPROC=$NPROC"
echo "[baseline_clean_corrector] Baseline inverter: $BASELINE_CKPT"
echo "[baseline_clean_corrector] Output: $OUTPUT_DIR"
echo "[baseline_clean_corrector] Cache: $DAEI_CACHE"

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment corrector \
  --corrector_model_from_pretrained "$BASELINE_CKPT" \
  --model_name_or_path t5-base \
  --embedder_model_name gtr_base \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --use_frozen_embeddings_as_input True \
  --dataset_name "$DATASET_NAME" \
  --dataset_mode standard \
  --use_less_data "$USE_LESS_DATA" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_seq_length 128 \
  --warmup_steps 50000 \
  --eval_steps 10000 \
  --save_steps 10000 \
  --metric_for_best_model eval_yahoo_bleu_score \
  --greater_is_better True \
  --output_dir "$OUTPUT_DIR" \
  --max_eval_samples 500 \
  --bf16 True \
  --num_train_epochs 100 \
  --eval_log_text_metrics_only True \
  --weight_decay 0.01 \
  --save_total_limit 3 \
  > "$LOG_FILE" 2>&1 &

echo "Baseline-clean Corrector training launched. PID: $!"
echo "Log: $LOG_FILE"

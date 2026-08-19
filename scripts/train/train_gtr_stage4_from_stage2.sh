#!/bin/bash
# ==========================================================================
# DAEI Pipeline — Stage 4 Ablation: Corrector from Stage 2
#
# Train the Corrector directly from:
#   - Stage 2 inverter: initialized as the frozen inversion model
#   - Stage 1 DAE: used as the frozen DAE Shield
#
# This intentionally skips Stage 3 joint fine-tuning, so it can run in parallel
# with Stage 3 as a direct comparison:
#   Stage1 DAE + Stage2 inverter -> Corrector
#
# Environment overrides:
#   STAGE1_DAE_CHECKPOINT       default: saves/dae_first_stage1/checkpoint-27400
#   STAGE2_INVERTER_CHECKPOINT  default: saves/dae_first_stage2_inverter
#   OUTPUT_DIR                  default: saves/dae_first_stage4_corrector_from_stage2
#   LOG_FILE                    default: logs/dae_first_stage4_corrector_from_stage2.log
#   BATCH_SIZE                  default: 80
#   GRADIENT_ACCUMULATION_STEPS default: 2
#   NPROC                       default: auto-detect visible GPUs
# ==========================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

# --- Checkpoint paths ---
# Stage 1 DAE checkpoint (contains dae.* keys in model.safetensors)
STAGE1_DAE_CKPT="${STAGE1_DAE_CHECKPOINT:-saves/dae_first_stage1/checkpoint-27400}"

# Stage 2 inverter checkpoint/root (complete v2t model trained on denoised embeddings)
STAGE2_INV_CKPT="${STAGE2_INVERTER_CHECKPOINT:-saves/dae_first_stage2_inverter}"

OUTPUT_DIR="${OUTPUT_DIR:-saves/dae_first_stage4_corrector_from_stage2}"
LOG_FILE="${LOG_FILE:-logs/dae_first_stage4_corrector_from_stage2.log}"

# Corrector hypothesis cache files include source/DAE checkpoint identity, so
# this ablation can share the common corrector cache root safely.
export DAEI_CACHE="${DAEI_CACHE:-$PROJECT_ROOT/data/corrector_cache}"
mkdir -p "$DAEI_CACHE"

# Corrector training overrides the inner loaded inverter trainer to use noisy
# target embeddings for this ablation via --dataset_mode noisy.

BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS:-2}
DATASET_NAME="${DATASET_NAME:-nq,msmarco,yahoo}"
USE_LESS_DATA="${USE_LESS_DATA:-7000000}"
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD:-1e-3}"

if [[ ! -d "$STAGE1_DAE_CKPT" ]]; then
  echo "ERROR: STAGE1_DAE_CHECKPOINT does not exist: $STAGE1_DAE_CKPT" >&2
  exit 1
fi

if [[ ! -d "$STAGE2_INV_CKPT" ]]; then
  echo "ERROR: STAGE2_INVERTER_CHECKPOINT does not exist: $STAGE2_INV_CKPT" >&2
  exit 1
fi

# Auto-detect GPU count
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

echo "[dae_first_stage4_corrector_from_stage2] NPROC=$NPROC"
echo "[dae_first_stage4_corrector_from_stage2] Stage 1 DAE: $STAGE1_DAE_CKPT"
echo "[dae_first_stage4_corrector_from_stage2] Stage 2 inverter: $STAGE2_INV_CKPT"
echo "[dae_first_stage4_corrector_from_stage2] Dataset: $DATASET_NAME/$USE_LESS_DATA (dataset_mode=noisy)"
echo "[dae_first_stage4_corrector_from_stage2] Output: $OUTPUT_DIR"
echo "[dae_first_stage4_corrector_from_stage2] Cache: $DAEI_CACHE"

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment dae_shield_corrector \
  --corrector_model_from_pretrained "$STAGE2_INV_CKPT" \
  --dae_checkpoint "$STAGE1_DAE_CKPT" \
  --use_dae_shield True \
  --dae_depth 3 \
  --dae_use_sigma_cond True \
  --dae_use_spectral_norm True \
  --early_stop_threshold "$EARLY_STOP_THRESHOLD" \
  --model_name_or_path t5-base \
  --embedder_model_name gtr_base \
  --num_repeat_tokens 16 \
  --embedder_no_grad True \
  --use_frozen_embeddings_as_input True \
  --dataset_name "$DATASET_NAME" \
  --dataset_mode noisy \
  --use_less_data "$USE_LESS_DATA" \
  --learning_rate 5e-4 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_seq_length 128 \
  --warmup_steps 50000 \
  --eval_steps 10000 \
  --save_steps 4000 \
  --output_dir "$OUTPUT_DIR" \
  --max_eval_samples 500 \
  --bf16 True \
  --num_train_epochs 100 \
  --eval_log_text_metrics_only True \
  --weight_decay 0.01 \
  --save_total_limit 3 \
  > "$LOG_FILE" 2>&1 &

echo "Stage 4 Corrector-from-Stage2 training launched. PID: $!"
echo "Log: $LOG_FILE"

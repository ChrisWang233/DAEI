#!/bin/bash
# ==========================================================================
# DAEI Pipeline — Stage 4: Corrector Training
#
# Train the Corrector using the jointly fine-tuned DAE (Stage 3) as a frozen
# denoiser. The Corrector's projection layers are initialized in the "denoised"
# distribution from the start — no distribution mismatch.
#
# Key advantages over the original DAE Shield Corrector:
#   - projection layers (embedding_transform_1/2/3) see denoised-distribution
#     inputs from the beginning
#   - The DAE was fine-tuned with text signal (Stage 3), so its denoising
#     direction is optimized for text reconstruction
#   - Initial hypothesis quality is higher (Stage 2-3 Inverter is stronger)
#
# Prerequisites:
#   - saves/dae_first_stage3_joint/checkpoint-XXXXX (Stage 3 joint model)
#     - Contains both v2t weights (used as inverter_checkpoint)
#     - Contains dae.* keys (extracted as frozen DAE)
#
# Cache root is set in DAEI.experiments.CORRECTOR_CACHE_PATH.
# ==========================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ ! -d saves/dae_first_stage3_joint/checkpoint-51500 ]]; then
  echo "ERROR: Stage 3 checkpoint does not exist: saves/dae_first_stage3_joint/checkpoint-51500" >&2
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

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment dae_shield_corrector \
  --inverter_checkpoint saves/dae_first_stage3_joint/checkpoint-51500 \
  --dae_checkpoint saves/dae_first_stage3_joint/checkpoint-51500 \
  --use_dae_shield True \
  --use_frozen_embeddings_as_input True \
  --dataset_mode noisy \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 200 \
  --per_device_eval_batch_size 200 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 50000 \
  --metric_for_best_model eval_indomain_bleu_score \
  --output_dir saves/dae_first_stage4_corrector \
  --max_eval_samples 500 \
  --bf16 True \
  --num_train_epochs 100 \
  --weight_decay 0.01 \
  > logs/dae_first_stage4_corrector.log 2>&1 &

echo "Stage 4 Corrector training launched. PID: $!"
echo "Log: logs/dae_first_stage4_corrector.log"

#!/bin/bash
# ==========================================================================
# DAEI Pipeline — Stage 2: Train Inverter on Denoised Embeddings
#
# Train the InversionModel (t5-base + gtr_base) using frozen_embeddings that
# have been pre-denoised by the Stage 1 DAE. This ensures the Inverter's
# projection layers adapt to the "denoised" distribution from the start.
#
# Prerequisites:
#   - Stage 1 DAE checkpoint.
#   - If denoised caches are missing, DAEI will create them from matching
#     noisy caches using the Stage 1 DAE.
#
# Key design:
#   - dataset_mode=denoised
#   - default train cache: data/dataset_cache/nq_msmarco_yahoo_denoised_7000000_gtr_base_train.arrow
#   - default val cache:   data/dataset_cache/nq_msmarco_yahoo_denoised_7000000_gtr_base_val.arrow
#   - T5 from HuggingFace pretrained (no baseline needed)
# ==========================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

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

echo "[dae_first_stage2_inverter] NPROC=$NPROC"

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment inversion \
  --use_frozen_embeddings_as_input True \
  --dataset_mode denoised \
  --stage1_checkpoint saves/dae_first_stage1/checkpoint-27400 \
  --learning_rate 1e-3 \
  --per_device_train_batch_size 384 \
  --per_device_eval_batch_size 384 \
  --num_train_epochs 100 \
  --warmup_steps 20000 \
  --output_dir saves/dae_first_stage2_inverter \
  --max_eval_samples 500 \
  --bf16 True \
  --mock_embedder True \
  > logs/dae_first_stage2_inverter.log 2>&1 &

echo "Stage 2 Inverter training launched. PID: $!"
echo "Log: logs/dae_first_stage2_inverter.log"

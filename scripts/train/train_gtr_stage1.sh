#!/bin/bash
# ==========================================================================
# DAEI Pipeline — Stage 1: Independent DAE Training
#
# Train a ResidualDAE using MC-SURE loss (no clean embeddings required).
# The DAEI model is frozen; only DAE parameters update.
#
# Prerequisites:
#   - Noisy embedding dataset (nq,msmarco with --dataset_mode noisy)
#   - No baseline / stage0 checkpoint needed
#
# Enhancements over vanilla Stage 1:
#   - Progressive noise curriculum with per-batch random sigma sampling:
#     Each batch samples σ ~ LogUniform[σ_min, upper(t)] where upper(t) decays
#     from σ_max=0.02 → σ_min=0.01 over the first 20% of training steps.
#     This ensures the SigmaConditioner sees ALL noise levels from step 1,
#     preventing the train-eval distribution mismatch that occurs with a
#     deterministic schedule (where the conditioner only sees one σ at a time).
#     mc_sure_loss uses the sampled σ as its formula coefficient (= actual noise).
#     Does NOT require clean embeddings.
#   - Sigma-conditioned DAE (SigmaConditioner enabled)
#   - Deeper DAE (depth=3 vs default 2)
#   - Spectral normalization on DAE Linear layers (stabilizes SURE Jacobian)
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

echo "[dae_first_stage1] NPROC=$NPROC"

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment joint_inversion \
  --training_stage 1 \
  --loss_mode sure \
  --embedder_model_name gtr_base \
  --use_frozen_embeddings_as_input True \
  --dataset_name nq,msmarco,yahoo \
  --dataset_mode noisy \
  --embedder_gaussian_noise_level 0.01 \
  --sure_n_probes 5 \
  --dae_lr 1e-3 \
  --dae_depth 3 \
  --dae_use_sigma_cond True \
  --dae_sigma_schedule log_uniform \
  --dae_use_spectral_norm True \
  --per_device_train_batch_size 160 \
  --per_device_eval_batch_size 160 \
  --max_seq_length 128 \
  --num_train_epochs 1000 \
  --warmup_steps 5000 \
  --output_dir saves/dae_first_stage1 \
  --max_eval_samples 500 \
  --eval_log_text_metrics_only False \
  --bf16 True \
  --use_less_data 7000000 \
  > logs/dae_first_stage1.log 2>&1 &

echo "Stage 1 DAE training launched. PID: $!"
echo "Log: logs/dae_first_stage1.log"

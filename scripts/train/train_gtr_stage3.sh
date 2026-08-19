#!/bin/bash
# ==========================================================================
# DAEI Pipeline — Stage 3: Joint Fine-tuning (DAE + Inverter)
#
# Jointly fine-tune the Stage 1 DAE and Stage 2 Inverter with:
#   L_total = L_SURE + lambda_ce * L_CE
#
# CE gradients flow through DAE → the DAE learns to denoise in a direction
# that benefits text reconstruction, not just embedding-space MSE.
#
# Key hyperparameter choices:
#   - DAE lr (1e-4) << Inverter lr (5e-4): prevent DAE collapse
#   - Lambda warmup: CE weight ramps from 0 to 1.0 over 10k steps
#   - DAE gradient clipping: 0.5 (stricter than default 1.0)
#   - Spectral normalization on DAE (consistent with Stage 1)
#   - PCGrad: projects CE gradient when it conflicts with SURE gradient
#   - No contrastive loss (requires clean embeddings which we don't have)
#
# Prerequisites:
#   - saves/dae_first_stage1/checkpoint-XXXXX  (Stage 1 DAE)
#   - saves/dae_first_stage2_inverter/checkpoint-XXXXX  (Stage 2 Inverter)
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

echo "[dae_first_stage3_joint] NPROC=$NPROC"

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC" -m DAEI.run)
else
  LAUNCH=(python -m DAEI.run)
fi

nohup "${LAUNCH[@]}" \
  --experiment joint_inversion \
  --training_stage 3 \
  --loss_mode sure \
  --stage1_checkpoint saves/dae_first_stage1/checkpoint-27400 \
  --stage2_checkpoint saves/dae_first_stage2_inverter \
  --use_frozen_embeddings_as_input True \
  --dataset_mode noisy \
  --embedder_gaussian_noise_level 0.01 \
  --sure_n_probes 5 \
  --learning_rate 1e-3 \
  --dae_lr 2e-4 \
  --dae_grad_max_norm 0.5 \
  --dae_pcgrad True \
  --lambda_warmup_steps 10000 \
  --per_device_train_batch_size 160 \
  --per_device_eval_batch_size 160 \
  --num_train_epochs 100 \
  --warmup_steps 5000 \
  --metric_for_best_model eval_indomain_bleu_score \
  --output_dir saves/dae_first_stage3_joint \
  --max_eval_samples 500 \
  --bf16 True \
  --weight_decay 0.01 \
  > logs/dae_first_stage3_joint.log 2>&1 &

echo "Stage 3 Joint fine-tuning launched. PID: $!"
echo "Log: logs/dae_first_stage3_joint.log"

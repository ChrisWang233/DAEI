# DAEI

DAEI (Denoising-Aware Embedding Inversion) reconstructs text from noisy dense embeddings. It combines a residual denoising autoencoder, an embedding inverter, joint denoising-aware fine-tuning, and a Corrector with a frozen DAE Shield. The main setting assumes the real embedder output is noisy, so clean embeddings are used only for evaluation and diagnostics, not as a required training signal.
Our pretrained GTR model is available at https://huggingface.co/ChrisWang233/daei_gtr_stage4.

## Training

### Environment

Create the conda environment from the project environment file:

```bash
conda env create -f environment.yml
conda activate daei
```

The training scripts auto-detect the number of visible GPUs. To choose GPUs manually:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Run the GTR pipeline

Run the four stages in order:

```bash
bash scripts/train/train_gtr_stage1.sh
bash scripts/train/train_gtr_stage2.sh
bash scripts/train/train_gtr_stage3.sh
bash scripts/train/train_gtr_stage4.sh
```

Stage summary:

- Stage 1 trains the DAE on noisy embeddings with SURE.
- Stage 2 trains the inverter on DAE-denoised cached embeddings.
- Stage 3 jointly fine-tunes the inverter and DAE.
- Stage 4 trains the Corrector with a frozen DAE Shield.

Logs are written under `logs/`, and checkpoints are written under `saves/`.


## Important Parameters

DAE parameters:

- `--dae_depth`: number of residual DAE blocks.
- `--dae_lr`: DAE optimizer learning rate.
- `--dae_use_sigma_cond`: conditions the DAE on the assumed noise level.
- `--dae_sigma_schedule`: noise schedule for DAE training, usually `fixed` or `log_uniform`.
- `--dae_use_spectral_norm`: applies spectral normalization to DAE linear layers for more stable SURE training.
- `--sure_n_probes`: number of Monte Carlo probes used in the SURE trace estimate.
- `--noise_sigma`: assumed Gaussian embedding noise level, typically `0.01`.

Inverter parameters:

- `--embedder_model_name`: embedding model family, for example `gtr_base`.
- `--max_seq_length`: maximum text sequence length.
- `--num_repeat_tokens`: number of repeated embedding tokens fed into the encoder-decoder inverter.
- `--dataset_name`: training datasets, for example `nq,msmarco,yahoo`.
- `--dataset_mode`: embedding cache mode. `noisy` is used for Stage 1 and Stage 3, while `denoised` is used for Stage 2.
- `--use_less_data`: cap on training examples.

Fine-tuning and Corrector parameters:

- `--stage1_checkpoint`: Stage 1 DAE checkpoint used by Stage 2 and Stage 3.
- `--stage2_checkpoint`: Stage 2 inverter checkpoint used by Stage 3.
- `--dae_checkpoint`: frozen DAE checkpoint used by Stage 4.
- `--inverter_checkpoint`: inverter checkpoint used by Stage 4 to generate hypotheses.
- `--learning_rate`: inverter or Corrector learning rate.
- `--lambda_warmup_steps`: warmup steps for the joint CE objective in Stage 3.
- `--dae_pcgrad`: enables PCGrad for DAE and CE gradient conflicts in joint training.
- `--max_correction_steps`: maximum recursive correction steps in Corrector evaluation/training.
- `--early_stop_threshold`: threshold for stopping correction when embedding change is small.

Many architecture and dataset parameters are inherited from upstream checkpoints, so later stages should not manually override them unless you intentionally want to check or change compatibility.

## Evaluation

### End-to-end text reconstruction

Evaluate a trained DAEI Corrector:

```bash
python scripts/test/daei_eval.py \
  --mode corrector \
  --skip_baseline \
  --dae_shield_ckpt saves/<stage4_corrector_checkpoint> \
  --eval_all_checkpoints true \
  --val_dir data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_val.arrow \
  --per_device_eval_batch_size 96 \
  --max_correction_steps 10 \
  --sequence_beam_width 3 \
  --max_sequence_length 32 \
  --cache_dir data/eval_cache/corrector_eval_stage4_cache \
  --out_json scripts/test/results/daei_eval/eval_daei_all.json
```

### DAE-only embedding evaluation

Evaluate DAE checkpoints directly against cached clean embeddings:

```bash
python scripts/test/eval_dae.py \
  --dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_val.arrow \
  --dae_ckpt stage1=saves/<stage1_dae_checkpoint> \
  --dae_ckpt stage3=saves/<stage3_joint_checkpoint> \
  --splits indomain yahoo python_code_alpaca \
  --batch_size 2048 \
  --sigma 0.01 \
  --out_json scripts/test/results/dae_eval/eval_gtr_dae_stage1_vs_stage3.json
```

### Metrics

- `bleu_score`: n-gram overlap between generated text and reference text. Higher is better.
- `rouge_score`: recall-oriented text overlap, useful for longer generations. Higher is better.
- `token_set_f1`: token-level set F1 between prediction and reference, less sensitive to word order than BLEU.
- `exact_match`: fraction of predictions that exactly match the reference string.
- `emb_cos_sim`: cosine similarity between embeddings of generated and reference text. Higher means semantic embedding recovery is closer.
- `sure_noisy_cos`: cosine similarity between noisy and clean embeddings during DAE diagnostics.
- `sure_dae_cos`: cosine similarity between DAE-denoised and clean embeddings.
- `sure_delta_cos`: improvement from denoising, computed as `sure_dae_cos - sure_noisy_cos`.
- `noisy_mse`: MSE between noisy and clean embeddings in DAE-only evaluation.
- `dae_mse`: MSE between DAE-denoised and clean embeddings.
- `delta_mse`: reduction in MSE from denoising, computed as `noisy_mse - dae_mse`.
- `residual_l2`: average L2 size of the DAE correction vector.


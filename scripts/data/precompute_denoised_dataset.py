#!/usr/bin/env python3
"""Precompute denoised embeddings for the DAEI Pipeline.

Loads the Stage 1 DAE checkpoint and runs it over cached rows so
``frozen_embeddings`` becomes DAE output; originals are copied to
``noisy_embeddings``.

Split keys:

- **Train** caches (``..._train.arrow``) are usually a ``DatasetDict`` with
  ``train`` and ``validation``. Default (--splits omitted) processes **all**
  keys, which matches both splits.
- **Val** caches (``..._val.arrow``) use **dataset names** as keys
  (see ``dataset_dict.json``), e.g. ``ag_news``, ``yahoo``, not
  ``train``/``validation``. Omit ``--splits`` to denoise every key.

Example::

python scripts/data/archive_precompute_denoised_dataset.py \
  --dae_checkpoint saves/gte_stage1/checkpoint-6150 \
  --input_dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gte_base_train.arrow \
  --output_dir data/dataset_cache/nq_msmarco_yahoo_denoised_7000000_gte_base_train.arrow \
  --batch_size 2048 \
  --dae_depth 3 \
  --dae_use_sigma_cond \
  --dae_use_spectral_norm \
  --sigma 0.01


python scripts/data/archive_precompute_denoised_dataset.py \
  --dae_checkpoint saves/gte_stage1/checkpoint-6150 \
  --input_dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gte_base_val.arrow \
  --output_dir data/dataset_cache/nq_msmarco_yahoo_denoised_7000000_gte_base_val.arrow \
  --batch_size 2048 \
  --dae_depth 3 \
  --dae_use_sigma_cond \
  --dae_use_spectral_norm \
  --sigma 0.01

"""

import argparse
import os
import sys
from pathlib import Path

import datasets
import torch
import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from DAEI.models.dae import ResidualDAE  # noqa: E402


def load_dae(args) -> ResidualDAE:
    """Instantiate and load DAE from a JointTrainer checkpoint."""
    dae = ResidualDAE(
        emb_dim=args.emb_dim,
        hidden_dim=args.dae_hidden_dim,
        depth=args.dae_depth,
        use_sigma_cond=args.dae_use_sigma_cond,
        use_spectral_norm=args.dae_use_spectral_norm,
    )

    ckpt_dir = args.dae_checkpoint

    # Try legacy dae.pt
    legacy_path = os.path.join(ckpt_dir, "dae.pt")
    if os.path.isfile(legacy_path):
        print(f"Loading DAE from legacy dae.pt: {legacy_path}")
        state = torch.load(legacy_path, map_location="cpu", weights_only=True)
        dae.load_state_dict(state)
    else:
        # Extract from full model checkpoint
        for fname in ("model.safetensors", "pytorch_model.bin"):
            weights_path = os.path.join(ckpt_dir, fname)
            if os.path.isfile(weights_path):
                break
        else:
            raise FileNotFoundError(
                f"No dae.pt / model.safetensors / pytorch_model.bin in {ckpt_dir}"
            )

        print(f"Extracting DAE weights from {weights_path}")
        if weights_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            full_state = load_file(weights_path)
        else:
            full_state = torch.load(weights_path, map_location="cpu", weights_only=True)

        dae_state = {
            k.replace("dae.", "", 1): v
            for k, v in full_state.items()
            if k.startswith("dae.")
        }
        if not dae_state:
            raise ValueError(f"No 'dae.' prefixed keys found in {weights_path}")
        dae.load_state_dict(dae_state)
        print(f"Loaded {len(dae_state)} DAE keys")
        del full_state

    dae.eval()
    for p in dae.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in dae.parameters())
    print(f"DAE ready: {n_params / 1e6:.2f}M params")
    return dae


def denoise_batch(batch, dae, device, sigma_t=None):
    """Map function: denoise frozen_embeddings."""
    emb = torch.tensor(batch["frozen_embeddings"], dtype=torch.float32, device=device)
    with torch.no_grad():
        denoised = dae(emb, sigma=sigma_t)
    batch["noisy_embeddings"] = batch["frozen_embeddings"]
    batch["frozen_embeddings"] = denoised.cpu().numpy().tolist()
    return batch


def main():
    parser = argparse.ArgumentParser(description="Precompute denoised embeddings")
    parser.add_argument("--dae_checkpoint", type=str, required=True)
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="Path to HuggingFace datasets dir (DatasetDict or Dataset)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--dae_hidden_dim", type=int, default=1024)
    parser.add_argument("--dae_depth", type=int, default=3)
    parser.add_argument("--dae_use_sigma_cond", action="store_true")
    parser.add_argument(
        "--dae_use_spectral_norm",
        action="store_true",
        help="Must match Stage 1 training (train_dae_first_stage1.sh uses True)",
    )
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Sigma to pass to sigma-conditioned DAE")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        metavar="NAMES",
        help="Comma-separated split or dataset keys (must match dataset_dict.json). "
        "Default: for DatasetDict, process all keys; for a single Dataset, the whole table. "
        "Train caches often use train,validation; val caches use names like ag_news,yahoo.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dae = load_dae(args).to(device)

    sigma_t = None
    if args.dae_use_sigma_cond:
        sigma_t = torch.tensor(args.sigma, device=device).unsqueeze(0)

    # Load input dataset
    print(f"Loading dataset from: {args.input_dataset}")
    try:
        ds = datasets.load_from_disk(args.input_dataset)
    except Exception:
        ds = datasets.Dataset.load_from_disk(args.input_dataset)

    is_dict = isinstance(ds, datasets.DatasetDict)
    if args.splits is not None:
        requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        requested = None

    if is_dict:
        available = list(ds.keys())
        if requested is not None:
            found = [s for s in requested if s in ds]
            if found:
                splits_to_process = found
                for m in [s for s in requested if s not in ds]:
                    print(f"  Split / dataset key '{m}' not in cache, skipping.")
            elif available:
                print(
                    "Requested splits %s not found in DatasetDict (available keys: %s). "
                    "Processing all keys — typical for *val*.arrow caches keyed by dataset name."
                    % (requested, available)
                )
                splits_to_process = available
            else:
                raise ValueError(
                    f"DatasetDict at {args.input_dataset} is empty (no splits)."
                )
        else:
            splits_to_process = available
            if not splits_to_process:
                raise ValueError(
                    f"DatasetDict at {args.input_dataset} is empty (no splits)."
                )
            print(
                "Processing all DatasetDict keys (default --splits): %s" % splits_to_process
            )

        result = {}
        for split_name in splits_to_process:
            split_ds = ds[split_name]
            print(f"\nProcessing split '{split_name}': {len(split_ds)} samples")

            split_ds = split_ds.map(
                lambda batch: denoise_batch(batch, dae, device, sigma_t),
                batched=True,
                batch_size=args.batch_size,
                desc=f"Denoising {split_name}",
                writer_batch_size=5000,
            )
            result[split_name] = split_ds

        result_ds = datasets.DatasetDict(result)
    else:
        print(f"\nProcessing single dataset: {len(ds)} samples")
        ds = ds.map(
            lambda batch: denoise_batch(batch, dae, device, sigma_t),
            batched=True,
            batch_size=args.batch_size,
            desc="Denoising",
            writer_batch_size=5000,
        )
        result_ds = ds

    if isinstance(result_ds, datasets.DatasetDict) and len(result_ds) == 0:
        raise RuntimeError(
            "No data was denoised (empty DatasetDict). Check --splits vs. dataset_dict.json keys."
        )

    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    print(f"\nSaving denoised dataset to: {args.output_dir}")
    result_ds.save_to_disk(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate DAE checkpoints directly in embedding space.

This compares noisy ``frozen_embeddings`` and ``DAE(frozen_embeddings)`` against
``clean_embeddings`` from a cached validation dataset.

/scratch2/hj82/yubow/miniconda3/envs/a100/bin/python scripts/test/eval_dae_embeddings.py \
  --dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_val.arrow \
  --dae_ckpt stage1=saves/dae_first_stage1/checkpoint-27400 \
  --splits indomain \
  --batch_size 2048 \
  --sigma 0.01


/scratch2/hj82/yubow/miniconda3/envs/a100/bin/python scripts/test/eval_dae_embeddings.py \
  --dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_val.arrow \
  --dae_ckpt stage1=saves/dae_first_stage1/checkpoint-27400 \
  --dae_ckpt stage3=saves/dae_first_stage3_joint/checkpoint-51500 \
  --splits indomain yahoo python_code_alpaca \
  --batch_size 2048 \
  --sigma 0.01 \
  --out_json scripts/test/results/eval_dae_stage1_vs_stage3.json

  /scratch2/hj82/yubow/miniconda3/envs/a100/bin/python scripts/test/eval_dae_embeddings.py \
  --dataset data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_train.arrow/validation \
  --dae_ckpt stage1=saves/dae_first_stage1/checkpoint-27400 \
  --dae_ckpt stage3=saves/dae_first_stage3_joint/checkpoint-51500 \
  --batch_size 2048 \
  --sigma 0.01 \
  --out_json scripts/test/results/eval_dae_stage1_vs_stage3.json

  /scratch2/hj82/yubow/miniconda3/envs/a100/bin/python scripts/test/eval_dae_embeddings.py \
  --dataset data/dataset_cache/nq_msmarco_yahoo_noisy_5000_gte_base_mean_norm_train.arrow/validation \
  --dae_ckpt stage1=saves/gte_stage1/checkpoint-6150 \
  --batch_size 2048 \
  --sigma 0.01 \
  --out_json scripts/test/results/eval_gte_dae_stage1.json
  
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import datasets
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_PRECOMPUTE_PATH = ROOT / "scripts" / "data" / "archive_precompute_denoised_dataset.py"
_PRECOMPUTE_SPEC = importlib.util.spec_from_file_location(
    "archive_precompute_denoised_dataset", _PRECOMPUTE_PATH
)
if _PRECOMPUTE_SPEC is None or _PRECOMPUTE_SPEC.loader is None:
    raise ImportError(f"Cannot load {_PRECOMPUTE_PATH}")
_PRECOMPUTE_MODULE = importlib.util.module_from_spec(_PRECOMPUTE_SPEC)
_PRECOMPUTE_SPEC.loader.exec_module(_PRECOMPUTE_MODULE)
load_dae = _PRECOMPUTE_MODULE.load_dae


def _parse_labeled_path(value: str) -> Tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad checkpoint spec: {value}")
        return label, path
    path = value.strip()
    label = Path(path).name or Path(path).parent.name
    return label, path


def _load_dataset(path: str):
    try:
        return datasets.load_from_disk(path)
    except Exception:
        return datasets.Dataset.load_from_disk(path)


def _iter_splits(ds, requested: List[str] | None) -> Iterable[Tuple[str, datasets.Dataset]]:
    if isinstance(ds, datasets.DatasetDict):
        names = list(ds.keys()) if requested is None else requested
        for name in names:
            if name not in ds:
                print(f"[eval_dae] split '{name}' not found, skipping")
                continue
            yield name, ds[name]
    else:
        if requested not in (None, ["dataset"]):
            print("[eval_dae] single Dataset input; ignoring --splits")
        yield "dataset", ds


def _as_float_tensor(value, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32, device=device)


def _evaluate_split(
    split_ds: datasets.Dataset,
    dae: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    sigma: float,
    use_sigma_cond: bool,
    max_samples: int | None,
) -> Dict[str, float]:
    for col in ("frozen_embeddings", "clean_embeddings"):
        if col not in split_ds.column_names:
            raise ValueError(f"Dataset split is missing required column: {col}")

    n = len(split_ds) if max_samples is None else min(len(split_ds), max_samples)
    totals = {
        "noisy_mse": 0.0,
        "dae_mse": 0.0,
        "noisy_cos": 0.0,
        "dae_cos": 0.0,
        "residual_l2": 0.0,
    }

    sigma_t = None
    if use_sigma_cond:
        sigma_t = torch.tensor(float(sigma), device=device)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = split_ds[start:end]
            noisy = _as_float_tensor(batch["frozen_embeddings"], device)
            clean = _as_float_tensor(batch["clean_embeddings"], device)
            denoised = dae(noisy, sigma=sigma_t)

            bsz = noisy.shape[0]
            noisy_flat = noisy.view(bsz, -1)
            clean_flat = clean.view(bsz, -1)
            denoised_flat = denoised.view(bsz, -1)

            totals["noisy_mse"] += (
                (noisy_flat - clean_flat).pow(2).sum(dim=-1).sum().item()
            )
            totals["dae_mse"] += (
                (denoised_flat - clean_flat).pow(2).sum(dim=-1).sum().item()
            )
            totals["noisy_cos"] += (
                F.cosine_similarity(noisy_flat, clean_flat, dim=-1).sum().item()
            )
            totals["dae_cos"] += (
                F.cosine_similarity(denoised_flat, clean_flat, dim=-1).sum().item()
            )
            totals["residual_l2"] += (
                (denoised_flat - noisy_flat).norm(dim=-1).sum().item()
            )

    if n == 0:
        raise ValueError("Cannot evaluate an empty split")

    result = {key: value / n for key, value in totals.items()}
    result["delta_mse"] = result["noisy_mse"] - result["dae_mse"]
    result["delta_cos"] = result["dae_cos"] - result["noisy_cos"]
    result["count"] = n
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DAE checkpoints on cached embeddings")
    parser.add_argument("--dataset", required=True, help="HF Dataset/DatasetDict cache path")
    parser.add_argument(
        "--dae_ckpt",
        action="append",
        required=True,
        help="Checkpoint path, optionally label=path. Repeat for stage1/stage3.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="DatasetDict keys to evaluate. Default: all keys.",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--dae_hidden_dim", type=int, default=1024)
    parser.add_argument("--dae_depth", type=int, default=3)
    parser.add_argument("--dae_use_sigma_cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dae_use_spectral_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = _load_dataset(args.dataset)
    ckpts = [_parse_labeled_path(item) for item in args.dae_ckpt]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, ckpt_path in ckpts:
        print(f"\n[eval_dae] loading {label}: {ckpt_path}")
        args.dae_checkpoint = ckpt_path
        dae = load_dae(args).to(device)
        dae.eval()

        label_results: Dict[str, Dict[str, float]] = {}
        for split_name, split_ds in _iter_splits(ds, args.splits):
            print(f"[eval_dae] evaluating {label} on {split_name} ({len(split_ds)} rows)")
            label_results[split_name] = _evaluate_split(
                split_ds=split_ds,
                dae=dae,
                device=device,
                batch_size=args.batch_size,
                sigma=args.sigma,
                use_sigma_cond=args.dae_use_sigma_cond,
                max_samples=args.max_samples,
            )
            metrics = label_results[split_name]
            print(
                "  count={count:.0f} noisy_mse={noisy_mse:.6f} dae_mse={dae_mse:.6f} "
                "delta_mse={delta_mse:.6f} noisy_cos={noisy_cos:.6f} "
                "dae_cos={dae_cos:.6f} delta_cos={delta_cos:.6f} residual_l2={residual_l2:.6f}".format(
                    **metrics
                )
            )

        results[label] = label_results
        del dae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"\n[eval_dae] wrote {args.out_json}")


if __name__ == "__main__":
    main()

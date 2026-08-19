#!/usr/bin/env python3
"""Build indomain validation caches with Gaussian noise injected into embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import datasets
import numpy as np


DEFAULT_INPUT = (
    "data/dataset_cache/"
    "nq_msmarco_yahoo_standard_7000000_gtr_base_val.arrow/indomain"
)
DEFAULT_SIGMAS = [0.005, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.05]


def _sigma_tag(sigma: float) -> str:
    return f"{sigma:g}"


def _add_noise_batch(
    batch: Dict[str, Any],
    *,
    rng: np.random.Generator,
    sigma: float,
) -> Dict[str, Any]:
    clean = np.asarray(batch["frozen_embeddings"], dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=clean.shape).astype(np.float32)
    noisy = clean + noise
    batch["clean_embeddings"] = clean.tolist()
    batch["frozen_embeddings"] = noisy.tolist()
    batch["noise_sigma"] = [float(sigma)] * clean.shape[0]
    return batch


def _embedding_summary(ds: datasets.Dataset, sigma: float) -> Dict[str, float]:
    sample_n = min(len(ds), 1000)
    sample = ds.select(range(sample_n))
    clean = np.asarray(sample["clean_embeddings"], dtype=np.float32)
    noisy = np.asarray(sample["frozen_embeddings"], dtype=np.float32)
    diff = noisy - clean
    return {
        "sigma": float(sigma),
        "rows": int(len(ds)),
        "sample_rows": int(sample_n),
        "empirical_noise_mean": float(diff.mean()),
        "empirical_noise_std": float(diff.std()),
        "mean_l2_noisy_minus_clean": float(np.linalg.norm(diff, axis=1).mean()),
    }


def build_noise_sweep(
    input_dataset: Path,
    output_dir: Path,
    sigmas: List[float],
    seed: int,
    batch_size: int,
) -> Dict[str, Any]:
    source = datasets.Dataset.load_from_disk(str(input_dataset))
    if "frozen_embeddings" not in source.column_names:
        raise ValueError(f"{input_dataset} missing required column frozen_embeddings")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("sigmas", {})
    else:
        summary: Dict[str, Any] = {
            "input_dataset": str(input_dataset),
            "output_dir": str(output_dir),
            "seed": seed,
            "sigmas": {},
        }
    summary["input_dataset"] = str(input_dataset)
    summary["output_dir"] = str(output_dir)
    summary["seed"] = seed

    for offset, sigma in enumerate(sigmas):
        tag = _sigma_tag(sigma)
        out_path = output_dir / f"sigma_{tag}.arrow"
        if out_path.exists():
            existing = datasets.DatasetDict.load_from_disk(str(out_path))["indomain"]
            summary["sigmas"][tag] = {
                "path": str(out_path),
                **_embedding_summary(existing, sigma),
            }
            print(f"[noise_sweep] sigma={tag} already exists, refreshed summary")
            continue

        rng = np.random.default_rng(seed + offset)
        noisy_ds = source.map(
            lambda batch: _add_noise_batch(batch, rng=rng, sigma=sigma),
            batched=True,
            batch_size=batch_size,
            desc=f"inject noise sigma={tag}",
        )
        result = datasets.DatasetDict({"indomain": noisy_ds})
        result.save_to_disk(str(out_path))
        summary["sigmas"][tag] = {
            "path": str(out_path),
            **_embedding_summary(noisy_ds, sigma),
        }
        print(
            f"[noise_sweep] sigma={tag} rows={len(noisy_ds)} "
            f"-> {out_path}"
        )

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[noise_sweep] wrote summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/noise_sweep_indomain_gtr"),
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    build_noise_sweep(
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        sigmas=args.sigmas,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Estimate local intrinsic dimensionality before/after DAE projection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import matplotlib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    }
)


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = (
    "data/dataset_cache/"
    "nq_msmarco_yahoo_noisy_7000000_gtr_base_train.arrow/validation"
)
DEFAULT_DAE_CKPTS = [
    ("stage1", "saves/dae_first_stage1/checkpoint-27400"),
    ("stage3", "saves/dae_first_stage3_joint/checkpoint-51500"),
]


def _load_dae_loader():
    precompute_path = ROOT / "scripts" / "data" / "archive_precompute_denoised_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "archive_precompute_denoised_dataset", precompute_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {precompute_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_dae


def _parse_labeled_path(value: str) -> Tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), path.strip()
    path = value.strip()
    return Path(path).name or Path(path).parent.name, path


def _load_dataset(path: str) -> datasets.Dataset:
    obj = datasets.load_from_disk(path)
    if isinstance(obj, datasets.DatasetDict):
        raise ValueError(f"{path} is a DatasetDict; pass a concrete split path")
    return obj


def _stack_column(ds: datasets.Dataset, indices: List[int], column: str) -> np.ndarray:
    values = ds.select(indices)[column]
    return np.asarray(values, dtype=np.float32)


def _maybe_l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


def _denoise(
    noisy: np.ndarray,
    dae: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    sigma: float,
    use_sigma_cond: bool,
) -> np.ndarray:
    outputs = []
    sigma_t = torch.tensor(float(sigma), device=device) if use_sigma_cond else None
    with torch.no_grad():
        for start in range(0, len(noisy), batch_size):
            batch = torch.tensor(
                noisy[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(dae(batch, sigma=sigma_t).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _lid_mle(
    nn: NearestNeighbors,
    query: np.ndarray,
    n_neighbors: int,
    batch_size: int,
) -> np.ndarray:
    lids = []
    eps = 1e-12
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        distances, _ = nn.kneighbors(batch, n_neighbors=n_neighbors)
        distances = np.maximum(distances, eps)
        r_k = distances[:, -1:]
        logs = np.log(distances[:, :-1] / r_k)
        log_mean = np.mean(logs, axis=1)
        lid = -1.0 / np.minimum(log_mean, -eps)
        lids.append(lid.astype(np.float64))
    return np.concatenate(lids, axis=0)


def _summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "count": int(len(values)),
    }


def _knn_summary(
    nn: NearestNeighbors,
    query: np.ndarray,
    n_neighbors: int,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    d1 = []
    dk = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        distances, _ = nn.kneighbors(batch, n_neighbors=n_neighbors)
        d1.append(distances[:, 0])
        dk.append(distances[:, -1])
    return {
        "nn_dist_1": _summary(np.concatenate(d1, axis=0)),
        "nn_dist_k": _summary(np.concatenate(dk, axis=0)),
    }


def _embedding_summary(x: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {"norm": _summary(np.linalg.norm(x, axis=1))}


def _relative_to_clean(x: np.ndarray, clean: np.ndarray) -> Dict[str, Dict[str, float]]:
    diff = x - clean
    x_norm = np.linalg.norm(x, axis=1)
    clean_norm = np.linalg.norm(clean, axis=1)
    cos = np.sum(x * clean, axis=1) / np.maximum(x_norm * clean_norm, 1e-12)
    return {
        "mse_to_clean": _summary(np.mean(diff * diff, axis=1)),
        "cos_to_clean": _summary(cos),
        "l2_to_clean": _summary(np.linalg.norm(diff, axis=1)),
    }


def _tight_ylim(values: List[float], errors: List[float] | None = None) -> Tuple[float, float]:
    lower_values = np.asarray(values, dtype=np.float64)
    upper_values = np.asarray(values, dtype=np.float64)
    if errors is not None:
        err = np.asarray(errors, dtype=np.float64)
        lower_values = lower_values - err
        upper_values = upper_values + err
    lo = float(np.min(lower_values))
    hi = float(np.max(upper_values))
    span = max(hi - lo, 1.0)
    return max(0.0, lo - 0.18 * span), hi + 0.70 * span


def _bar_values(
    results: Dict[str, Dict[str, Dict[str, float]]],
    labels: List[str],
    group: str,
    statistic: str,
) -> List[float]:
    return [results[label][group]["lid"][statistic] for label in labels]


def _bar_sem(
    results: Dict[str, Dict[str, Dict[str, float]]],
    labels: List[str],
    group: str,
) -> List[float]:
    sem = []
    for label in labels:
        lid = results[label][group]["lid"]
        sem.append(lid["std"] / np.sqrt(max(lid["count"], 1)))
    return sem


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )


def _plot_one(
    results: Dict[str, Dict[str, Dict[str, float]]],
    out_plot: Path,
    statistic: str,
    title: str,
    with_errorbar: bool = False,
    y_lim: Tuple[float, float] | None = None,
) -> None:
    labels = list(results)
    groups = ["clean", "noisy", "denoised"]
    colors = {"clean": "#666666", "noisy": "#CC6677", "denoised": "#4477AA"}

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    all_values: List[float] = []
    all_errors: List[float] = []
    for offset, group in zip([-width, 0.0, width], groups):
        values = _bar_values(results, labels, group, statistic)
        errors = _bar_sem(results, labels, group) if with_errorbar else None
        all_values.extend(values)
        if errors is not None:
            all_errors.extend(errors)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=group,
            color=colors[group],
            edgecolor="black",
            linewidth=0.4,
            yerr=errors,
            capsize=4 if with_errorbar else 0,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        _annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LID MLE (kNN)")
    ax.set_ylim(*(y_lim or _tight_ylim(all_values, all_errors if with_errorbar else None)))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=1,
        frameon=True,
    )
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200)
    plt.close(fig)


def _plot(results: Dict[str, Dict[str, Dict[str, float]]], out_plot: Path) -> List[Path]:
    stem = out_plot.with_suffix("")
    suffix = out_plot.suffix or ".png"
    outputs = [
        (stem.with_name(f"{stem.name}_mean").with_suffix(suffix), "mean", "Mean LID vs. clean reference", False),
        (stem.with_name(f"{stem.name}_median").with_suffix(suffix), "median", "Median LID vs. clean reference", False),
        (
            stem.with_name(f"{stem.name}_mean_errorbar").with_suffix(suffix),
            "mean",
            "Mean LID vs. clean reference",
            True,
        ),
    ]
    labels = list(results)
    groups = ["clean", "noisy", "denoised"]
    y_values: List[float] = []
    y_errors: List[float] = []
    for _, statistic, _, with_errorbar in outputs:
        for group in groups:
            values = _bar_values(results, labels, group, statistic)
            y_values.extend(values)
            if with_errorbar:
                y_errors.extend(_bar_sem(results, labels, group))
            else:
                y_errors.extend([0.0] * len(values))
    shared_y_lim = _tight_ylim(y_values, y_errors)
    written = []
    for path, statistic, title, with_errorbar in outputs:
        _plot_one(results, path, statistic, title, with_errorbar, shared_y_lim)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot_json",
        type=Path,
        default=None,
        help="Read an existing result JSON and only regenerate plots.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--dae_ckpt",
        action="append",
        default=None,
        help="Checkpoint path, optionally label=path. Repeat for stage1/stage3.",
    )
    parser.add_argument("--ref_samples", type=int, default=400)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--n_neighbors", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--dae_hidden_dim", type=int, default=1024)
    parser.add_argument("--dae_depth", type=int, default=3)
    parser.add_argument("--dae_use_sigma_cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dae_use_spectral_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out_json",
        type=Path,
        default=Path("scripts/test/results/lid_projection/lid_projection_gtr_stage1_stage3.json"),
    )
    parser.add_argument(
        "--out_plot",
        type=Path,
        default=Path("scripts/test/results/lid_projection/lid_projection_gtr_stage1_stage3.png"),
    )
    args = parser.parse_args()

    if args.plot_json is not None:
        with args.plot_json.open("r", encoding="utf-8") as f:
            results = json.load(f)
        plot_paths = _plot(results, args.out_plot)
        for path in plot_paths:
            print(f"[lid] wrote {path}")
        return

    ckpts = (
        [_parse_labeled_path(item) for item in args.dae_ckpt]
        if args.dae_ckpt
        else DEFAULT_DAE_CKPTS
    )
    ds = _load_dataset(args.dataset)
    for col in ("frozen_embeddings", "clean_embeddings"):
        if col not in ds.column_names:
            raise ValueError(f"Dataset is missing required column: {col}")

    if args.ref_samples <= args.n_neighbors:
        raise ValueError("--ref_samples must be larger than --n_neighbors")
    total_needed = args.ref_samples + args.eval_samples
    if len(ds) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} rows, but dataset has {len(ds)}"
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=total_needed, replace=False).tolist()
    ref_idx = indices[: args.ref_samples]
    eval_idx = indices[args.ref_samples :]

    print(f"[lid] loading reference/eval embeddings from {args.dataset}")
    clean_ref = _stack_column(ds, ref_idx, "clean_embeddings")
    clean_eval = _stack_column(ds, eval_idx, "clean_embeddings")
    noisy_eval = _stack_column(ds, eval_idx, "frozen_embeddings")
    if args.normalize:
        clean_ref = _maybe_l2_normalize(clean_ref)
        clean_eval = _maybe_l2_normalize(clean_eval)
        noisy_eval = _maybe_l2_normalize(noisy_eval)

    print(
        f"[lid] fitting kNN on clean reference: n={len(clean_ref)} "
        f"dim={clean_ref.shape[1]} k={args.n_neighbors}"
    )
    nn = NearestNeighbors(n_neighbors=args.n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(clean_ref)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dae = _load_dae_loader()
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    lid_clean = _lid_mle(nn, clean_eval, args.n_neighbors, args.batch_size)
    lid_noisy = _lid_mle(nn, noisy_eval, args.n_neighbors, args.batch_size)
    clean_diagnostics = {
        **_knn_summary(nn, clean_eval, args.n_neighbors, args.batch_size),
        **_embedding_summary(clean_eval),
    }
    noisy_diagnostics = {
        **_knn_summary(nn, noisy_eval, args.n_neighbors, args.batch_size),
        **_embedding_summary(noisy_eval),
        **_relative_to_clean(noisy_eval, clean_eval),
    }

    for label, ckpt_path in ckpts:
        print(f"[lid] loading DAE {label}: {ckpt_path}")
        args.dae_checkpoint = ckpt_path
        dae = load_dae(args).to(device)
        dae.eval()
        denoised = _denoise(
            noisy_eval,
            dae=dae,
            device=device,
            batch_size=args.batch_size,
            sigma=args.sigma,
            use_sigma_cond=args.dae_use_sigma_cond,
        )
        if args.normalize:
            denoised = _maybe_l2_normalize(denoised)
        lid_denoised = _lid_mle(nn, denoised, args.n_neighbors, args.batch_size)
        denoised_diagnostics = {
            **_knn_summary(nn, denoised, args.n_neighbors, args.batch_size),
            **_embedding_summary(denoised),
            **_relative_to_clean(denoised, clean_eval),
        }

        results[label] = {
            "clean": {"lid": _summary(lid_clean), **clean_diagnostics},
            "noisy": {"lid": _summary(lid_noisy), **noisy_diagnostics},
            "denoised": {"lid": _summary(lid_denoised), **denoised_diagnostics},
            "metadata": {
                "checkpoint": ckpt_path,
                "dataset": args.dataset,
                "ref_samples": args.ref_samples,
                "eval_samples": args.eval_samples,
                "n_neighbors": args.n_neighbors,
                "sigma": args.sigma,
                "normalize": args.normalize,
                "seed": args.seed,
            },
        }
        print(
            "[lid] {label}: clean={clean:.3f} noisy={noisy:.3f} "
            "denoised={denoised:.3f}".format(
                label=label,
                clean=results[label]["clean"]["lid"]["mean"],
                noisy=results[label]["noisy"]["lid"]["mean"],
                denoised=results[label]["denoised"]["lid"]["mean"],
            )
        )
        del dae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    plot_paths = _plot(results, args.out_plot)
    print(f"[lid] wrote {args.out_json}")
    for path in plot_paths:
        print(f"[lid] wrote {path}")


if __name__ == "__main__":
    main()

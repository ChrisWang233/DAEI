#!/usr/bin/env python3
"""
Compare the baseline Corrector and the DAE Shield Corrector on the same noisy
NQ and MS MARCO validation data.

The summary reports BLEU, token-set F1, and ROUGE metrics. Set
``--max_correction_steps`` to match training. For the combined ``nq,msmarco``
dataset, the primary metric prefix is ``eval_nq,msmarco`` and matches the
single-step metrics in the training log (``num_gen_recursive_steps=1``).


  ------------------------------------------------------------------------------------------

python scripts/test/daei_eval.py \
  --mode corrector \
  --skip_baseline \
  --dae_shield_ckpt saves/dae_first_stage4_corrector \
  --eval_all_checkpoints true \
  --val_dir data/dataset_cache/nq_msmarco_yahoo_noisy_7000000_gtr_base_val.arrow \
  --per_device_eval_batch_size 96 \
  --max_correction_steps 10 \
  --sequence_beam_width 3 \
  --max_sequence_length 32 \
  --cache_dir data/eval_cache/corrector_eval_stage4_cont_cache \
  --out_json scripts/test/results/daei_eval/eval_daei_all.json

-------------------------------------------------------------

for s in 0.005 0.008 0.01 0.012 0.015 0.018 0.02; do
  python scripts/test/daei_eval.py \
    --mode corrector \
    --skip_baseline \
    --dae_shield_ckpt saves/dae_first_stage4_corrector \
    --eval_all_checkpoints true \
    --val_dir data/noise_sweep_indomain_gtr/sigma_${s}.arrow \
    --per_device_eval_batch_size 64 \
    --max_correction_steps 10 \
    --sequence_beam_width 3 \
    --max_sequence_length 32 \
    --cache_dir data/eval_cache/corrector_eval_noise_sweep_cache/sigma_${s} \
    --out_json scripts/test/results/noise_sweep/eval_daei_noise_sweep_sigma_${s}.json
done

"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# PyTorch 2.6+ checkpoint args .bin
import torch

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _permissive_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _permissive_load  # type: ignore[assignment]

DEFAULT_VAL_DIR = (
    ROOT
    / "data/dataset_cache/nq_msmarco_noisy_5000000_gtr_base_train.arrow/validation"
)

# DataCollatorForCorrection sends non-hypothesis columns to tokenizer.pad;
# remove raw string columns such as text and category before tensorization.
_CORRECTOR_BASE_COLS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "length",
        "embedder_input_ids",
        "embedder_attention_mask",
        "frozen_embeddings",
        "idx",
    }
)

_INVERTER_BASE_COLS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "length",
        "embedder_input_ids",
        "embedder_attention_mask",
        "frozen_embeddings",
        "clean_embeddings",
        "noisy_embeddings",
        "idx",
    }
)


def _strip_cols_for_corrector_collator(val):
    drop = [
        c
        for c in val.column_names
        if c not in _CORRECTOR_BASE_COLS and not c.startswith("hypothesis_")
    ]
    if drop:
        val = val.remove_columns(drop)
    return val


def _strip_cols_for_inverter_collator(val):
    drop = [c for c in val.column_names if c not in _INVERTER_BASE_COLS]
    if drop:
        val = val.remove_columns(drop)
    return val


def _str2bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


_TOKENIZED_SEQUENCE_COLUMNS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "embedder_input_ids",
        "embedder_attention_mask",
        "hypothesis_input_ids",
        "hypothesis_attention_mask",
    }
)


def _truncate_tokenized_columns(val, max_sequence_length: Optional[int]):
    if max_sequence_length is None:
        return val
    if max_sequence_length < 1:
        raise ValueError("--max_sequence_length must be >= 1")
    cols = [c for c in val.column_names if c in _TOKENIZED_SEQUENCE_COLUMNS]
    if not cols:
        return val

    def truncate_sequence(seq):
        if isinstance(seq, (list, tuple)):
            return list(seq)[:max_sequence_length]
        if hasattr(seq, "tolist"):
            return seq.tolist()[:max_sequence_length]
        return seq

    def truncate_batch(batch):
        for col in cols:
            batch[col] = [truncate_sequence(seq) for seq in batch[col]]
        if "length" in batch:
            batch["length"] = [
                min(int(length), max_sequence_length) for length in batch["length"]
            ]
        return batch

    return val.map(
        truncate_batch,
        batched=True,
        batch_size=1024,
        desc=f"[nice_eval] truncate tokenized columns to {max_sequence_length}",
    )


def _prepare_eval_dataset(
    val,
    *,
    name: str,
    tokenizer,
    max_seq_length: int,
    max_sequence_length: Optional[int],
    max_samples: Optional[int],
    strip_for_corrector: bool,
):
    val.set_format(None)
    if max_samples is not None and len(val) > max_samples:
        val = val.select(range(max_samples))
    if "idx" not in val.column_names:
        val = val.add_column("idx", list(range(len(val))))
    if "labels" not in val.column_names:
        if "text" not in val.column_names:
            raise ValueError(
                f"Validation split '{name}' has no tokenized columns or text column; "
                "cannot construct labels: "
                f"columns={val.column_names}"
            )
        from DAEI.tokenize_data import tokenize_function_noisy_frozen

        tok_fn = tokenize_function_noisy_frozen(
            tokenizer=tokenizer,
            text_column_name="text",
            max_seq_length=max_seq_length,
            padding=False,
        )
        val = val.map(
            tok_fn,
            batched=True,
            batch_size=128,
            num_proc=1,
            desc=f"[nice_eval] tokenize {name}",
        )
    val = _truncate_tokenized_columns(val, max_sequence_length)
    if strip_for_corrector:
        val = _strip_cols_for_corrector_collator(val)
    else:
        val = _strip_cols_for_inverter_collator(val)
    if len(val) == 0:
        raise ValueError(f"Validation split is empty: {name}")
    val.set_format("pt")
    return val


def _load_eval_datasets(
    val_dir: Path,
    *,
    tokenizer,
    max_seq_length: int,
    max_sequence_length: Optional[int],
    max_samples: Optional[int],
    strip_for_corrector: bool,
):
    import datasets

    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory does not exist: {val_dir}")
    loaded = datasets.load_from_disk(str(val_dir))
    if isinstance(loaded, datasets.DatasetDict):
        eval_datasets = {
            name: _prepare_eval_dataset(
                val,
                name=name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_sequence_length=max_sequence_length,
                max_samples=max_samples,
                strip_for_corrector=strip_for_corrector,
            )
            for name, val in loaded.items()
        }
    else:
        eval_name = val_dir.name
        eval_datasets = {
            eval_name: _prepare_eval_dataset(
                loaded,
                name=eval_name,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_sequence_length=max_sequence_length,
                max_samples=max_samples,
                strip_for_corrector=strip_for_corrector,
            )
        }
    return datasets.DatasetDict(eval_datasets)


def _patch_val_only(
    val_dir: Path,
    max_sequence_length: Optional[int],
    max_samples: Optional[int],
    strip_for_corrector: bool,
) -> Tuple[Callable, Callable]:
    import datasets

    from DAEI.experiments import InversionExperiment

    _orig = InversionExperiment.load_train_and_val_datasets

    def patched(self, model, tokenizer, embedder_tokenizer):
        effective_max_seq_length = max_sequence_length or self.model_args.max_seq_length
        if max_sequence_length is not None:
            self.model_args.max_seq_length = max_sequence_length
            if hasattr(model, "config"):
                model.config.max_seq_length = max_sequence_length
        eval_dict = _load_eval_datasets(
            val_dir,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
            max_sequence_length=max_sequence_length,
            max_samples=max_samples,
            strip_for_corrector=strip_for_corrector,
        )
        first_key = next(iter(eval_dict))
        first_val = eval_dict[first_key]
        n_dummy = min(8, len(first_val)) if len(first_val) > 0 else 0
        if n_dummy == 0:
            raise ValueError(f"Validation dataset is empty: {val_dir}")
        dummy_train = first_val.select(range(n_dummy))
        dummy_train.set_format("pt")
        print(f"[nice_eval] loaded eval datasets from {val_dir}: {list(eval_dict.keys())}")
        return dummy_train, eval_dict

    def restore():
        InversionExperiment.load_train_and_val_datasets = _orig  # type: ignore[assignment]

    InversionExperiment.load_train_and_val_datasets = patched  # type: ignore[assignment]
    return patched, restore


def _exact_checkpoint_if_available(checkpoint_folder: str) -> Optional[str]:
    path = Path(checkpoint_folder)
    model_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    if path.is_dir() and any((path / name).exists() for name in model_files):
        return str(path)
    if path.name.startswith("checkpoint-"):
        return str(path)
    return None


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("checkpoint-")


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("checkpoint-")), path.name
    except ValueError:
        return sys.maxsize, path.name


def _eval_targets(ckpt: str, eval_all_checkpoints: bool) -> list[tuple[str, str]]:
    root = Path(ckpt)
    if not eval_all_checkpoints:
        return [(root.name or "model", ckpt)]
    if _is_checkpoint_dir(root):
        raise ValueError(
            "--eval_all_checkpoints true requires a parent directory, not checkpoint-*"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint parent dir not found: {root}")
    checkpoints = sorted(
        (p for p in root.iterdir() if _is_checkpoint_dir(p)),
        key=_checkpoint_sort_key,
    )
    if not checkpoints:
        print(
            f"[nice_eval] No checkpoint-* directories found under {root}; "
            "evaluating it as a single model directory"
        )
        return [(root.name or "model", str(root))]
    return [(p.name, str(p)) for p in checkpoints]


def _safe_path_name(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in {"-", "_"} else "_")
    return "".join(keep).strip("_") or "model"


def _resolve_out_json(args) -> Path:
    if args.out_json is not None:
        return args.out_json
    if args.mode == "inverter":
        ckpt_name = _safe_path_name(Path(args.ckpt).name if args.ckpt else "model")
        suffix = "_all_ckpts" if args.eval_all_checkpoints else ""
        return args.results_dir / f"nice_eval_inverter_{ckpt_name}{suffix}.json"
    return args.results_dir / "nice_eval_corrector_compare.json"


def _write_results(results: Dict[str, Any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote complete metrics to: {out_json}")


def _reinit_trainer_accelerator(trainer) -> None:
    from accelerate.state import AcceleratorState, PartialState

    try:
        _ = PartialState().distributed_type
    except AttributeError:
        AcceleratorState._reset_state(reset_partial_state=True)

    trainer.args._n_gpu = 1 if torch.cuda.is_available() else 0
    trainer.args.deepspeed_plugin = None
    if "__cached__setup_devices" in trainer.args.__dict__:
        del trainer.args.__dict__["__cached__setup_devices"]
    if "_setup_devices" in dir(trainer.args):
        _ = trainer.args._setup_devices
    trainer.args.distributed_state = getattr(
        trainer.args, "distributed_state", None
    ) or PartialState()
    trainer.args.local_rank = -1
    trainer.create_accelerator_and_postprocess()


def _print_text_metrics(label: str, metrics: Dict[str, Any]) -> None:
    from DAEI.trainers.base import filter_eval_metrics_for_log

    slim = filter_eval_metrics_for_log(metrics)
    print(f"\n========== {label} ==========")
    print(json.dumps(slim, indent=2, ensure_ascii=False, default=str))


def _evaluate_with_fresh_accelerator_per_split(trainer) -> Dict[str, Any]:
    eval_dataset = trainer.eval_dataset
    if isinstance(eval_dataset, dict):
        metrics: Dict[str, Any] = {}
        for split_name, split_dataset in eval_dataset.items():
            print(f"[nice_eval] running split: {split_name}")
            _reinit_trainer_accelerator(trainer)
            split_metrics = trainer.evaluate(
                eval_dataset=split_dataset,
                metric_key_prefix=f"eval_{split_name}",
            )
            metrics.update(split_metrics)
        return metrics
    _reinit_trainer_accelerator(trainer)
    return trainer.evaluate()


def _apply_corrector_eval_settings(
    trainer,
    *,
    max_correction_steps: int,
    sequence_beam_width: int,
) -> None:
    if hasattr(trainer, "max_correction_steps"):
        trainer.max_correction_steps = max_correction_steps
        trainer.args.max_correction_steps = max_correction_steps
    if hasattr(trainer, "sequence_beam_width"):
        trainer.sequence_beam_width = sequence_beam_width
        trainer.return_best_hypothesis = sequence_beam_width > 1
    if hasattr(trainer, "num_gen_recursive_steps"):
        trainer.num_gen_recursive_steps = max_correction_steps


def _local_pretrained_corrector_overrides(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    pretrained = config.get("corrector_model_from_pretrained")
    if not isinstance(pretrained, str) or not pretrained:
        return {}

    sibling_name = pretrained.split("/")[-1]
    sibling = model_dir.parent / sibling_name
    if sibling.is_dir():
        return {"corrector_model_from_pretrained": str(sibling)}
    return {}


def _load_experiment_and_trainer_for_eval(
    *,
    checkpoint_folder: str,
    checkpoint: Optional[str],
    max_samples: Optional[int],
    max_sequence_length: Optional[int],
):
    from DAEI.analyze_utils import (
        load_experiment_and_trainer,
        load_experiment_and_trainer_from_pretrained,
    )

    try:
        return load_experiment_and_trainer(
            checkpoint_folder=checkpoint_folder,
            checkpoint=checkpoint,
            do_eval=False,
            sanity_decode=False,
            max_seq_length=max_sequence_length,
        )
    except FileNotFoundError:
        model_dir = Path(checkpoint or checkpoint_folder)
        if not (model_dir / "config.json").is_file():
            raise
        print(
            f"[nice_eval] {model_dir} has no data/model/training args .bin; "
            "loading from the pretrained config.json"
        )
        return load_experiment_and_trainer_from_pretrained(
            name=str(model_dir),
            use_less_data=max_samples or 1000,
            model_args_overrides=(
                {"max_seq_length": max_sequence_length}
                if max_sequence_length is not None
                else None
            ),
            training_args_overrides=_local_pretrained_corrector_overrides(model_dir),
        )


def _run_one(
    checkpoint_folder: str,
    label: str,
    val_dir: Path,
    max_sequence_length: Optional[int],
    max_samples: Optional[int],
    per_device_eval_batch_size: int,
    max_correction_steps: int,
    sequence_beam_width: int,
    cache_dir: Path,
    strip_for_corrector: bool,
) -> Dict[str, Any]:
    os.environ["DAEI_CACHE"] = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    trainer = None
    _, restore = _patch_val_only(
        val_dir, max_sequence_length, max_samples, strip_for_corrector
    )
    try:
        checkpoint = _exact_checkpoint_if_available(checkpoint_folder)
        if checkpoint is not None:
            print(f"[nice_eval] loading exact checkpoint directory: {checkpoint}")
        _, trainer = _load_experiment_and_trainer_for_eval(
            checkpoint_folder=checkpoint_folder,
            checkpoint=checkpoint,
            max_samples=max_samples,
            max_sequence_length=max_sequence_length,
        )
        if max_sequence_length is not None:
            if hasattr(trainer.model, "config"):
                trainer.model.config.max_seq_length = max_sequence_length
            if hasattr(trainer, "inversion_trainer") and hasattr(
                trainer.inversion_trainer.model, "config"
            ):
                trainer.inversion_trainer.model.config.max_seq_length = (
                    max_sequence_length
                )
        trainer.args.per_device_eval_batch_size = per_device_eval_batch_size
        _apply_corrector_eval_settings(
            trainer,
            max_correction_steps=max_correction_steps,
            sequence_beam_width=sequence_beam_width,
        )
        metrics = _evaluate_with_fresh_accelerator_per_split(trainer)
    finally:
        restore()
        if trainer is not None:
            del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_text_metrics(label, metrics)
    return metrics


def _run_corrector_group(
    *,
    results: Dict[str, Any],
    group_key: str,
    checkpoint_arg: str,
    args: argparse.Namespace,
) -> None:
    for target_label, ckpt in _eval_targets(
        checkpoint_arg, args.eval_all_checkpoints
    ):
        if args.eval_all_checkpoints:
            result_key = f"{group_key}_{target_label}"
            cache_dir = args.cache_dir / group_key / target_label
        else:
            result_key = group_key
            cache_dir = args.cache_dir / group_key

        display_label = (
            result_key
            if args.eval_all_checkpoints
            else f"corrector_{group_key}"
        )
        results[result_key] = _run_one(
            ckpt,
            display_label,
            args.val_dir,
            args.max_sequence_length,
            args.max_samples,
            args.per_device_eval_batch_size,
            args.max_correction_steps,
            args.sequence_beam_width,
            cache_dir,
            strip_for_corrector=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate inverter, joint, and Corrector models on the same validation "
            "data using the primary text metrics"
        )
    )
    p.add_argument(
        "--mode",
        choices=["corrector", "inverter"],
        default="corrector",
        help=(
            "Use the original comparison for Corrector models, or evaluate a Stage 2/3 "
            "joint checkpoint in inverter mode"
        ),
    )
    p.add_argument(
        "--val_dir",
        type=Path,
        default=DEFAULT_VAL_DIR,
        help="Validation subdirectory from save_to_disk, or a DatasetDict root directory",
    )
    p.add_argument("--ckpt", "--dae_first_ckpt", type=str, default=None)
    p.add_argument(
        "--baseline_ckpt",
        type=str,
        default=str(ROOT / "saves/corrector_baseline"),
        help="Baseline Corrector output directory or checkpoint-* directory",
    )
    p.add_argument(
        "--dae_shield_ckpt",
        type=str,
        default=str(ROOT / "saves/corrector_dae_shield/checkpoint-378400"),
        help="DAE Shield Corrector output directory or checkpoint-* directory",
    )
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_dae_shield", action="store_true")
    p.add_argument("--eval_all_checkpoints", type=_str2bool, default=False)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument(
        "--max_sequence_length",
        "--max_seq_length",
        type=int,
        default=None,
        help="Optionally truncate tokenized evaluation sequences to this length, e.g. 64",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument(
        "--max_correction_steps",
        type=int,
        default=20,
        help=(
            "Match the Corrector training setting; DAE Shield uses this value for the "
            "multi-step evaluation subset when a DAE is available"
        ),
    )
    p.add_argument(
        "--sequence_beam_width",
        type=int,
        default=1,
        help=(
            "Sequence-level beam width for recursive correction; 1 is the default, while "
            "values above 1 retain multiple candidates and select the closest embedding"
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=Path,
        default=ROOT / "data/eval_cache/corrector_eval_val_compare_cache",
        help="Precomputed hypothesis cache, with separate subdirectories for each model",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional path for complete metrics JSON; defaults to --results_dir",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT / "scripts/test/results",
        help="Default results directory when --out_json is not specified",
    )
    args = p.parse_args()
    if args.sequence_beam_width < 1:
        raise ValueError("--sequence_beam_width must be >= 1")
    if args.max_sequence_length is not None and args.max_sequence_length < 1:
        raise ValueError("--max_sequence_length must be >= 1")
    out_json = _resolve_out_json(args)

    results: Dict[str, Any] = {}
    if args.mode == "inverter":
        if not args.ckpt:
            raise ValueError("--mode inverter requires --ckpt/--dae_first_ckpt")
        for label, ckpt in _eval_targets(args.ckpt, args.eval_all_checkpoints):
            results[label] = _run_one(
                ckpt,
                label,
                args.val_dir,
                args.max_sequence_length,
                args.max_samples,
                args.per_device_eval_batch_size,
                args.max_correction_steps,
                args.sequence_beam_width,
                args.cache_dir / label,
                strip_for_corrector=False,
            )
        _write_results(results, out_json)
        return

    if not args.skip_baseline:
        _run_corrector_group(
            results=results,
            group_key="baseline",
            checkpoint_arg=args.baseline_ckpt,
            args=args,
        )
    if not args.skip_dae_shield:
        _run_corrector_group(
            results=results,
            group_key="dae_shield",
            checkpoint_arg=args.dae_shield_ckpt,
            args=args,
        )

    _write_results(results, out_json)


if __name__ == "__main__":
    main()

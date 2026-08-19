import abc
import functools
import hashlib
import json
import logging
import os
import shutil

import resource
import sys
from typing import Any, Dict, Optional

import datasets
import torch
import transformers

import DAEI
from DAEI.collator import DataCollatorForCorrection
from DAEI.data_helpers import dataset_from_args, load_n2n_dataset, load_standard_val_datasets
from DAEI.models import (
    CorrectorEncoderModel,
    InversionModel,
)
from DAEI.models.config import InversionConfig
from DAEI.run_args import DataArguments, ModelArguments, TrainingArguments
from DAEI.tokenize_data import (
    embed_dataset_batch,
    embed_dataset_batch_n2n,
    tokenize_function,
    tokenize_function_llama_chat,
)
from DAEI.utils import MockEmbedder, dataset_map_multi_worker, get_num_proc

# Allow W&B to start slowly.
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

# Don't send telemetry to HF every time we train.
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "False"
# os.environ["TOKENIZERS_PARALLELISM"] = "True"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger = logging.getLogger(__name__)

DATASET_CACHE_PATH = "data/dataset_cache/"
CORRECTOR_CACHE_PATH = "data/corrector_cache/"

# Noisy compilation from torch.compile
try:
    torch._logging.set_logs(dynamo=logging.INFO)
except AttributeError:
    # torch version too low
    pass


def md5_hash_kwargs(**kwargs) -> str:
    # We ignore special hf args that start with _ like '__cached__setup_devices'.
    safe_kwargs = {k: str(v) for k, v in kwargs.items() if not k.startswith("_")}
    s = json.dumps(safe_kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


def make_hf_fingerprint(*parts: object) -> str:
    """Build a deterministic datasets fingerprint under the 64-char HF limit."""
    s = "||".join(str(p) for p in parts)
    return hashlib.md5(s.encode()).hexdigest()


def _load_args_bin(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_checkpoint_arg_file(checkpoint_dir: str, filename: str) -> Optional[str]:
    candidates = [
        os.path.join(checkpoint_dir, filename),
        os.path.join(checkpoint_dir, os.pardir, filename),
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.isfile(norm):
            return norm
    return None


def _load_checkpoint_args(checkpoint_dir: Optional[str]) -> Dict[str, Any]:
    if not checkpoint_dir:
        return {}
    args = {}
    for key, filename in (
        ("model", "model_args.bin"),
        ("data", "data_args.bin"),
        ("training", "training_args.bin"),
    ):
        path = _find_checkpoint_arg_file(checkpoint_dir, filename)
        if path:
            args[key] = _load_args_bin(path)
            logger.info("Loaded %s args from %s", key, path)
    missing = [k for k in ("model", "data", "training") if k not in args]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir!r} is missing args files for {missing}. "
            "Expected model_args.bin, data_args.bin, and training_args.bin in the "
            "checkpoint directory or its parent. These are required for safe "
            "pipeline config inheritance."
        )
    return args


def _stage2_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    return getattr(training_args, "stage2_checkpoint", None) or getattr(
        training_args, "stage2_inverter_checkpoint", None
    )


def _inherit_arg_field(
    target: Any,
    source: Any,
    field_name: str,
    source_label: str,
) -> None:
    if source is None or not hasattr(source, field_name):
        return
    old_value = getattr(target, field_name, None)
    new_value = getattr(source, field_name)
    if old_value != new_value:
        logger.info(
            "Inheriting %s from %s: %r -> %r",
            field_name,
            source_label,
            old_value,
            new_value,
        )
        setattr(target, field_name, new_value)


def _assert_arg_field_matches(
    left: Any,
    right: Any,
    field_name: str,
    left_label: str,
    right_label: str,
) -> None:
    if (
        left is None
        or right is None
        or not hasattr(left, field_name)
        or not hasattr(right, field_name)
    ):
        return
    left_value = getattr(left, field_name)
    right_value = getattr(right, field_name)
    if left_value != right_value:
        raise ValueError(
            f"Pipeline checkpoint config mismatch for {field_name}: "
            f"{left_label} has {left_value!r}, {right_label} has {right_value!r}. "
            "Use checkpoints from the same pipeline or rebuild the downstream cache."
        )


def _inherit_config_from_pipeline_checkpoints(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    stage1 = _load_checkpoint_args(getattr(training_args, "stage1_checkpoint", None))
    stage2 = _load_checkpoint_args(_stage2_checkpoint(training_args))
    inverter_checkpoint = getattr(training_args, "inverter_checkpoint", None) or getattr(
        training_args, "corrector_model_from_pretrained", None
    )
    inverter = _load_checkpoint_args(inverter_checkpoint)
    dae_checkpoint = getattr(training_args, "dae_checkpoint", None)
    dae = _load_checkpoint_args(dae_checkpoint)

    stage1_model = stage1.get("model")
    stage1_data = stage1.get("data")
    stage1_training = stage1.get("training")
    inverter_model = inverter.get("model") or stage2.get("model")
    inverter_data = inverter.get("data") or stage2.get("data")
    dae_model = dae.get("model") or stage1_model
    dae_data = dae.get("data") or stage1_data
    dae_training = dae.get("training") or stage1_training

    for field_name in ("embedder_model_name", "max_seq_length", "num_repeat_tokens"):
        source = inverter_model or stage1_model
        source_label = (
            "inverter_checkpoint"
            if inverter.get("model") is not None
            else "stage2_checkpoint"
            if stage2.get("model") is not None
            else "stage1_checkpoint"
        )
        _inherit_arg_field(model_args, source, field_name, source_label)

    for field_name in ("dataset_name", "use_less_data"):
        source = inverter_data or stage1_data
        source_label = (
            "inverter_checkpoint"
            if inverter.get("data") is not None
            else "stage2_checkpoint"
            if stage2.get("data") is not None
            else "stage1_checkpoint"
        )
        _inherit_arg_field(data_args, source, field_name, source_label)

    for field_name in ("dae_depth", "dae_use_sigma_cond", "dae_use_spectral_norm"):
        _inherit_arg_field(
            training_args,
            dae_training,
            field_name,
            "dae_checkpoint" if dae.get("training") is not None else "stage1_checkpoint",
        )

    for field_name in ("embedder_model_name", "max_seq_length", "num_repeat_tokens"):
        _assert_arg_field_matches(
            stage1_model,
            stage2.get("model"),
            field_name,
            "stage1_checkpoint",
            "stage2_checkpoint",
        )
        _assert_arg_field_matches(
            dae_model,
            inverter_model,
            field_name,
            "dae_checkpoint",
            "inverter_checkpoint",
        )
    for field_name in ("dataset_name", "use_less_data"):
        _assert_arg_field_matches(
            stage1_data,
            stage2.get("data"),
            field_name,
            "stage1_checkpoint",
            "stage2_checkpoint",
        )
        _assert_arg_field_matches(
            dae_data,
            inverter_data,
            field_name,
            "dae_checkpoint",
            "inverter_checkpoint",
        )


class Experiment(abc.ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        _inherit_config_from_pipeline_checkpoints(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )
        # Interactions between args handled here.  Default to validation loss,
        # but respect an explicitly passed best-checkpoint metric such as
        # eval_yahoo_bleu_score.
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = f"{data_args.dataset_name}_loss"
            training_args.greater_is_better = False
        elif training_args.greater_is_better is None:
            metric_name = str(training_args.metric_for_best_model).lower()
            training_args.greater_is_better = not (
                metric_name == "loss" or metric_name.endswith("_loss")
            )

        logger.info(
            "Save checkpoints according to metric_for_best_model %s:",
            training_args.metric_for_best_model,
        )

        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        # Set random seed, add hash to output path.
        transformers.set_seed(training_args.seed)

        if training_args.output_dir is None:
            training_args.output_dir = os.path.join("saves", self.kwargs_hash)
        print(f"Experiment output_dir = {training_args.output_dir}")
        # Set up output_dir and wandb.
        self._setup_logging()
        self._consider_init_wandb()

    @property
    def config(self) -> InversionConfig:
        return InversionConfig(
            **vars(self.data_args),
            **vars(self.model_args),
            **vars(self.training_args),
        )

    @property
    def is_llama_chat(self) -> bool:
        return self.model_args.embedder_model_name in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ]

    @property
    def dataset_kwargs(self) -> Dict[str, str]:
        return {
            "model_name": self.model_args.model_name_or_path,
            "embedder_name": self.model_args.embedder_model_name,
            "max_seq_length": str(self.model_args.max_seq_length),
            "use_less_data": str(self.data_args.use_less_data),
            "use_full_data_datasets": str(
                self.data_args.use_full_data_datasets or ""
            ),
            "embedder_model_api": str(self.model_args.embedder_model_api),
        }

    def _get_readable_embedding_cache_name(
        self, split: str = "train", dataset_mode: Optional[str] = None
    ) -> str:
        """Cache name: dataset_name+dataset_mode+use_less_data[+_fd_<names>]+embedder_name+split.

        E.g. nq_standard_-1_gtr_base_train.arrow or nq_noisy_-1_gtr_base_train.arrow.
        With ``--use_full_data_datasets yahoo``: ``..._-1_fd_yahoo_gtr_base_train``.
        """
        dataset_name = self.data_args.dataset_name_str
        dataset_mode = dataset_mode or getattr(self.data_args, "dataset_mode", "standard")
        use_less_data = str(self.data_args.use_less_data)
        embedder_name = self.model_args.embedder_model_name
        if embedder_name == "gte_base":
            embedder_name = "gte_base_mean_norm"

        def sanitize(s: str) -> str:
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for c in invalid_chars:
                s = s.replace(c, '_')
            return s

        fd_names = getattr(self.data_args, "use_full_data_dataset_names", None) or []
        fd_suffix = (
            "_fd_" + sanitize("_".join(sorted(fd_names)))
            if fd_names
            else ""
        )

        cache_name = (
            f"{sanitize(dataset_name)}_{sanitize(dataset_mode)}_{sanitize(use_less_data)}"
            f"{fd_suffix}_{sanitize(embedder_name)}_{split}"
        )
        return cache_name

    def _get_embedding_cache_path(
        self, split: str = "train", dataset_mode: Optional[str] = None
    ) -> str:
        readable_cache_name = self._get_readable_embedding_cache_name(
            split=split, dataset_mode=dataset_mode
        )
        return os.path.join(DATASET_CACHE_PATH, (readable_cache_name + ".arrow"))

    def _infer_noisy_cache_path(self, denoised_path: str, split: str) -> str:
        if "_denoised_" in denoised_path:
            return denoised_path.replace("_denoised_", "_noisy_")
        return self._get_embedding_cache_path(split=split, dataset_mode="noisy")

    def _barrier_if_distributed(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _is_main_process(self) -> bool:
        return int(getattr(self.training_args, "process_index", 0)) == 0

    def _load_stage1_dae_for_denoising(
        self, model: transformers.PreTrainedModel
    ) -> torch.nn.Module:
        ckpt_dir = getattr(self.training_args, "stage1_checkpoint", None)
        if not ckpt_dir:
            raise ValueError(
                "dataset_mode=denoised requires --stage1_checkpoint when the "
                "denoised cache is missing."
            )

        from DAEI.models.dae import ResidualDAE

        dae = ResidualDAE(
            emb_dim=getattr(model, "embedder_dim", 768),
            hidden_dim=getattr(self.training_args, "dae_hidden_dim", 1024),
            depth=getattr(self.training_args, "dae_depth", 2),
            use_sigma_cond=getattr(self.training_args, "dae_use_sigma_cond", False),
            use_spectral_norm=getattr(self.training_args, "dae_use_spectral_norm", False),
        )

        legacy_path = os.path.join(ckpt_dir, "dae.pt")
        if os.path.isfile(legacy_path):
            logger.info("Loading Stage 1 DAE from legacy dae.pt: %s", legacy_path)
            try:
                state = torch.load(legacy_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(legacy_path, map_location="cpu")
            dae.load_state_dict(state)
        else:
            weights_path = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"No dae.pt / model.safetensors / pytorch_model.bin in {ckpt_dir}"
                )

            logger.info("Extracting Stage 1 DAE weights from %s", weights_path)
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                full_state = load_file(weights_path)
            else:
                try:
                    full_state = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    full_state = torch.load(weights_path, map_location="cpu")
            dae_state = {
                k.replace("dae.", "", 1): v
                for k, v in full_state.items()
                if k.startswith("dae.")
            }
            if not dae_state:
                raise ValueError(f"No 'dae.' prefixed keys found in {weights_path}")
            dae.load_state_dict(dae_state)
            logger.info("Loaded %d Stage 1 DAE keys", len(dae_state))
            del full_state

        dae.eval()
        for p in dae.parameters():
            p.requires_grad = False
        return dae

    def _denoise_batch_for_cache(self, batch, dae, denoise_device, sigma_t=None):
        emb = torch.tensor(
            batch["frozen_embeddings"], dtype=torch.float32, device=denoise_device
        )
        with torch.no_grad():
            denoised = dae(emb, sigma=sigma_t)
        batch["noisy_embeddings"] = batch["frozen_embeddings"]
        batch["frozen_embeddings"] = denoised.cpu().numpy().tolist()
        return batch

    def _save_denoised_cache_from_noisy(
        self,
        noisy_path: str,
        denoised_path: str,
        model: transformers.PreTrainedModel,
    ) -> None:
        if os.path.exists(denoised_path):
            return

        if not self._is_main_process():
            self._barrier_if_distributed()
            if not os.path.exists(denoised_path):
                raise FileNotFoundError(
                    f"Denoised cache was not created by main process: {denoised_path}"
                )
            return

        if not os.path.exists(noisy_path):
            raise FileNotFoundError(
                "dataset_mode=denoised could not find the denoised cache or the "
                f"matching noisy cache.\n  denoised: {denoised_path}\n  noisy: {noisy_path}"
            )

        logger.info(
            "Denoised cache missing; creating it from noisy cache. noisy=%s denoised=%s",
            noisy_path, denoised_path,
        )
        print("creating denoised cache from noisy cache:")
        print("  noisy:", noisy_path)
        print("  denoised:", denoised_path)

        denoise_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dae = self._load_stage1_dae_for_denoising(model).to(denoise_device)
        sigma_t = None
        if getattr(self.training_args, "dae_use_sigma_cond", False):
            sigma_t = torch.tensor(
                getattr(self.training_args, "noise_sigma", 0.01),
                device=denoise_device,
            ).unsqueeze(0)

        loaded = datasets.load_from_disk(noisy_path)
        batch_size = max(1, int(self.training_args.per_device_train_batch_size))

        if isinstance(loaded, datasets.DatasetDict):
            result = {}
            for key, split_ds in loaded.items():
                print(f"  denoising key '{key}' ({len(split_ds)} rows)")
                result[key] = split_ds.map(
                    lambda batch: self._denoise_batch_for_cache(
                        batch, dae, denoise_device, sigma_t
                    ),
                    batched=True,
                    batch_size=batch_size,
                    desc=f"Denoising {key}",
                    writer_batch_size=5000,
                )
            denoised = datasets.DatasetDict(result)
        else:
            print(f"  denoising dataset ({len(loaded)} rows)")
            denoised = loaded.map(
                lambda batch: self._denoise_batch_for_cache(
                    batch, dae, denoise_device, sigma_t
                ),
                batched=True,
                batch_size=batch_size,
                desc="Denoising",
                writer_batch_size=5000,
            )

        os.makedirs(os.path.dirname(denoised_path) or ".", exist_ok=True)
        denoised.save_to_disk(denoised_path)
        print("saved denoised cache:", denoised_path)
        self._barrier_if_distributed()

    def _ensure_denoised_cache(
        self,
        denoised_path: str,
        split: str,
        model: transformers.PreTrainedModel,
    ) -> None:
        noisy_path = self._infer_noisy_cache_path(denoised_path, split=split)
        self._save_denoised_cache_from_noisy(
            noisy_path=noisy_path,
            denoised_path=denoised_path,
            model=model,
        )

    def _setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # if self.training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_error()

    def run(self):
        if self.training_args.do_eval:
            self.evaluate()
        else:
            self.train()

    def train(self) -> Dict:
        # *** Training ***
        training_args = self.training_args
        logger.info("*** Training ***")

        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Checkpointing logic
        checkpoint = self._get_checkpoint()
        logging.info("Experiment::train() loaded checkpoint %s", checkpoint)
        trainer = self.load_trainer()

        # Save model_args and data_args before training. Trainer will save training_args.
        if training_args.local_rank <= 0:
            torch.save(
                self.data_args, os.path.join(training_args.output_dir, "data_args.bin")
            )
            torch.save(
                self.model_args,
                os.path.join(training_args.output_dir, "model_args.bin"),
            )

        # train.   :)
        print(f"train() called – resume-from_checkpoint = {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics

    def evaluate(self) -> Dict:
        # *** Evaluation ***
        logger.info("*** Evaluate ***")
        trainer = self.load_trainer()
        num_eval_samples = len(trainer.eval_dataset)
        metrics = trainer.evaluate()
        max_eval_samples = (
            self.data_args.max_eval_samples
            if self.data_args.max_eval_samples is not None
            else num_eval_samples
        )
        if not getattr(self.training_args, "eval_log_text_metrics_only", False):
            metrics["eval_samples"] = min(max_eval_samples, num_eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    def _get_checkpoint(self) -> Optional[str]:
        training_args = self.training_args
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
                training_args.output_dir
            )
            if (
                last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint:
            logger.info("Loading from checkpoint %s", checkpoint)
        else:
            logger.info("No checkpoint found, training from scratch")

        return checkpoint

    @property
    def kwargs_hash(self) -> str:
        all_args = {
            **vars(self.model_args),
            **vars(self.data_args),
            **vars(self.training_args),
        }
        all_args.pop("local_rank")
        # print("all_args:", all_args)
        return md5_hash_kwargs(**all_args)

    @property
    def _world_size(self) -> int:
        try:
            return torch.distributed.get_world_size()
        except (RuntimeError, ValueError):
            return 1

    @property
    def _is_main_worker(self) -> bool:
        return (self.training_args.local_rank <= 0) and (
            int(os.environ.get("LOCAL_RANK", 0)) <= 0
        )

    @property
    @abc.abstractmethod
    def _wandb_project_name(self) -> str:
        raise NotImplementedError()

    @property
    def _wandb_exp_name(self) -> str:
        name_args = [
            self.training_args.exp_group_name,
            self.training_args.exp_name,
            self.model_args.model_name_or_path,
            self.model_args.embedder_model_name,
        ]
        name_args = [n for n in name_args if ((n is not None) and len(n))]
        return "__".join(name_args)

    def _consider_init_wandb(self) -> None:
        if self.training_args.use_wandb and self._is_main_worker:
            import wandb

            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_exp_name,
                id=self.kwargs_hash,
                resume=True,
            )
            training_args = vars(self.training_args)
            # deepspeed kwargs are not json serializable
            training_args = {
                k: v for k, v in training_args.items() if "deepspeed" not in k
            }
            wandb.config.update(
                {
                    **vars(self.model_args),
                    **vars(self.data_args),
                    **training_args,
                },
                allow_val_change=True,
            )
            # Long-running experiments have been killed because wandb
            # runs out of file descriptors to write summary files
            # to. Very silly error, but seems unfixed:
            # https://github.com/wandb/wandb/issues/2825
            #
            # Anyway, this line of code should (hopefully) set the
            # limit to infinity so this can't happen.
            resource.setrlimit(
                resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )
        else:
            # Disable W&B
            pass
            # os.environ["WANDB_MODE"] = "disabled"
            # os.environ["WANDB_DISABLED"] = "true"

    @abc.abstractmethod
    def load_trainer(self) -> transformers.Trainer:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self) -> transformers.PreTrainedModel:
        raise NotImplementedError()

    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            padding="max_length",
            truncation="max_length",
            max_length=self.model_args.max_seq_length,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Disable super annoying warning:
        # https://github.com/huggingface/transformers/issues/22638
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer

    def get_collator(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> transformers.DataCollatorForSeq2Seq:
        return transformers.DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            padding="max_length",
            max_length=self.model_args.max_seq_length,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

    def _load_train_dataset_uncached(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
    ) -> datasets.DatasetDict:
        data_args = self.data_args
        ###########################################################################
        # Load datasets
        logger.info("Loading dataset '%s'...", self.data_args.dataset_name)
        raw_datasets = dataset_from_args(self.data_args)

        # Remove extra features except for 'frozen_embeddings' which could be embeddings
        # saved to disk.
        column_names = list(raw_datasets["train"].features)
        ALLOWED_COLUMN_NAMES = {"frozen_embeddings"}
        column_names = [c for c in column_names if c not in ALLOWED_COLUMN_NAMES]

        # this argument allows us to *train* on less data (for example 1% of our training set).
        # When use_full_data_datasets is set, caps are applied inside dataset_from_args (and/or skipped);
        # do not truncate again here or full corpora would be cut to use_less_data.
        if (
            data_args.use_less_data
            and (data_args.use_less_data > 0)
            and not data_args.use_full_data_dataset_names
        ):
            for key in raw_datasets:
                new_length = min(len(raw_datasets[key]), data_args.use_less_data)
                raw_datasets[key] = raw_datasets[key].select(range(new_length))
        print(
            ">> using fast tokenizers:", tokenizer.is_fast, embedder_tokenizer.is_fast
        )

        tokenize_fn = (
            tokenize_function_llama_chat if self.is_llama_chat else tokenize_function
        )
        for key in raw_datasets:
            raw_datasets[key] = dataset_map_multi_worker(
                dataset=raw_datasets[key],
                map_fn=tokenize_fn(
                    tokenizer,
                    embedder_tokenizer,
                    "text",
                    self.model_args.max_seq_length,
                    padding=False,
                    prefix="search_document"
                    if self.model_args.embedder_model_name
                    == "nomic-ai/nomic-embed-text-v1"
                    else None,
                ),
                batched=True,
                num_proc=get_num_proc(),
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )
        tokenized_datasets = raw_datasets
        ###########################################################################
        tokenized_datasets["train"].set_format("pt")
        tokenized_datasets["train"] = tokenized_datasets["train"].add_column(
            "idx", range(len(tokenized_datasets["train"]))
        )
        ###########################################################################
        if self.model_args.use_frozen_embeddings_as_input:
            print(
                f"[Precomputing embeddings with batch size: {self.training_args.per_device_train_batch_size}]"
            )
            assert torch.cuda.is_available()
            model = model.to(device)

            dataset_mode = getattr(self.data_args, "dataset_mode", "standard")
            noise_sigma = 0.0
            if dataset_mode in ("noisy", "n2n"):
                noise_sigma = self.model_args.embedder_gaussian_noise_level
                if noise_sigma == 0:
                    noise_sigma = 0.01
                print(f"  [dataset_mode={dataset_mode}, noise_sigma={noise_sigma}]")

            if dataset_mode == "n2n":
                embed_fn_train = functools.partial(
                    embed_dataset_batch_n2n, model, noise_sigma=noise_sigma
                )
                embed_fn_val = embed_fn_train
            elif dataset_mode == "noisy":
                # For training splits we only need noisy embeddings.
                embed_fn_train = functools.partial(
                    embed_dataset_batch, model, noise_sigma=noise_sigma, store_clean=False
                )
                # For validation splits we also store clean (non-noisy) embeddings
                # so SURE evaluation can compare noisy vs. denoised vs. clean.
                embed_fn_val = functools.partial(
                    embed_dataset_batch, model, noise_sigma=noise_sigma, store_clean=True
                )
            else:
                embed_fn_train = functools.partial(embed_dataset_batch, model)
                embed_fn_val = embed_fn_train

            new_tokenized_datasets = {}
            for key, d in tokenized_datasets.items():
                split_name = "train" if key == "train" else "val"
                readable_cache_name = self._get_readable_embedding_cache_name(split=split_name)
                new_fingerprint = make_hf_fingerprint(
                    d._fingerprint, readable_cache_name
                )
                print("\tsaving precomputed embeddings to file:", readable_cache_name)
                embed_fn = embed_fn_train if key == "train" else embed_fn_val
                new_tokenized_datasets[key] = dataset_map_multi_worker(
                    dataset=d,
                    map_fn=embed_fn,
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    new_fingerprint=new_fingerprint,
                    num_proc=1,
                )
            tokenized_datasets = datasets.DatasetDict(new_tokenized_datasets)
        ###########################################################################
        max_eval_samples = min(
            len(tokenized_datasets["validation"]), self.data_args.max_eval_samples
        )
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(max_eval_samples)
        )
        tokenized_datasets["validation"] = tokenized_datasets["validation"].add_column(
            "idx", range(len(tokenized_datasets["validation"]))
        )
        tokenized_datasets["validation"].set_format("pt")
        ###########################################################################
        return tokenized_datasets

    def _prepare_val_datasets_dict(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
        val_datasets_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        for name, dataset in val_datasets_dict.items():
            max_eval_samples = min(len(dataset), self.data_args.max_eval_samples)
            val_datasets_dict[name] = val_datasets_dict[name].select(
                range(max_eval_samples)
            )
            val_datasets_dict[name] = val_datasets_dict[name].add_column(
                "idx", range(len(val_datasets_dict[name]))
            )
            val_datasets_dict[name].set_format("pt")

        tokenize_fn = (
            tokenize_function_llama_chat if self.is_llama_chat else tokenize_function
        )
        for key in val_datasets_dict:
            val_datasets_dict[key] = dataset_map_multi_worker(
                dataset=val_datasets_dict[key],
                map_fn=tokenize_fn(
                    tokenizer=tokenizer,
                    embedder_tokenizer=embedder_tokenizer,
                    text_column_name="text",
                    max_seq_length=self.model_args.max_seq_length,
                    padding=False,
                ),
                remove_columns=["text"],
                batched=True,
                batch_size=1024,
                num_proc=get_num_proc(),
                desc="Running tokenizer on dataset",
            )

        # filter out empty examples (these exist for xsum documents).
        val_datasets_dict = val_datasets_dict.filter(lambda ex: ex["length"] > 1)

        if self.model_args.use_frozen_embeddings_as_input:
            assert torch.cuda.is_available()
            model = model.to(device)

            dataset_mode = getattr(self.data_args, "dataset_mode", "standard")
            noise_sigma = 0.0
            if dataset_mode in ("noisy", "n2n"):
                noise_sigma = self.model_args.embedder_gaussian_noise_level
                if noise_sigma == 0:
                    noise_sigma = 0.01

            if dataset_mode == "n2n":
                # Train uses two-view N2N; auxiliary val is still used for SURE-style embedding
                # metrics, which require noisy ``frozen_embeddings`` + ``clean_embeddings``.
                embed_fn = functools.partial(
                    embed_dataset_batch, model, noise_sigma=noise_sigma, store_clean=True
                )
            elif dataset_mode == "noisy":
                # Match training val: noisy frozen_embeddings + clean_embeddings for SURE eval.
                embed_fn = functools.partial(
                    embed_dataset_batch, model, noise_sigma=noise_sigma, store_clean=True
                )
            else:
                embed_fn = functools.partial(embed_dataset_batch, model)

            new_tokenized_datasets = {}
            readable_cache_name = self._get_readable_embedding_cache_name(split="val")
            for key, d in val_datasets_dict.items():
                new_tokenized_datasets[key] = dataset_map_multi_worker(
                    dataset=d,
                    map_fn=embed_fn,
                    batched=True,
                    batch_size=self.training_args.per_device_train_batch_size,
                    new_fingerprint=make_hf_fingerprint(
                        d._fingerprint, readable_cache_name, key
                    ),
                    num_proc=1,
                )
            val_datasets_dict = datasets.DatasetDict(new_tokenized_datasets)
        return val_datasets_dict

    def _load_val_datasets_uncached(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
    ) -> datasets.DatasetDict:
        val_datasets_dict = load_standard_val_datasets()
        logger.info(
            "Loaded %d validation datasets: %s",
            len(val_datasets_dict),
            val_datasets_dict.keys(),
        )
        return self._prepare_val_datasets_dict(
            model=model,
            tokenizer=tokenizer,
            embedder_tokenizer=embedder_tokenizer,
            val_datasets_dict=val_datasets_dict,
        )

    def _ensure_val_clean_embeddings_if_needed(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
        train_datasets: datasets.DatasetDict,
        val_datasets_dict: datasets.DatasetDict,
    ) -> None:
        """Override in subclasses to backfill clean_embeddings for val when missing (e.g. SURE eval)."""
        pass

    def load_train_and_val_datasets(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
    ):
        dataset_kwargs: Dict[str, str] = self.dataset_kwargs

        # Only set this if it's true, for backwards-compatibility with
        # when we forgot to cache using this argument.
        if self.model_args.use_frozen_embeddings_as_input:
            dataset_kwargs["use_frozen_embeddings_as_input"] = "True"
            # Deprecated arg below. We used to cache different
            # embeddings for suffixes. Then they became the same.
            # Removing the below line will invalidate other
            # people's caches.
            dataset_kwargs["suffix_conditioning"] = "False"

        # os.environ["TOKENIZERS_PARALLELISM"] = "True"
        print(
            "Loading datasets with TOKENIZERS_PARALLELISM =",
            os.environ.get("TOKENIZERS_PARALLELISM"),
        )
        ######################################################################
        train_dataset_kwargs = {
            "dataset_name": self.data_args.dataset_name,
            **dataset_kwargs,
        }
        dataset_mode = getattr(self.data_args, "dataset_mode", "standard")
        # Use readable cache name instead of MD5 hash
        train_dataset_path = self._get_embedding_cache_path(split="train")
        # Optionally set a train dataset path override
        train_dataset_path = (
            os.environ.get("DAEI_TRAIN_DATASET_PATH")
            or os.environ.get("VEC2TEXT_TRAIN_DATASET_PATH")
            or train_dataset_path
        )
        if dataset_mode == "denoised":
            self._ensure_denoised_cache(
                denoised_path=train_dataset_path,
                split="train",
                model=model,
            )
        if os.path.exists(train_dataset_path):
            try:
                print("loading train dataset from path:", train_dataset_path)
                train_datasets = datasets.load_from_disk(train_dataset_path)
            except (FileNotFoundError, OSError) as e:
                if dataset_mode == "denoised":
                    raise RuntimeError(
                        f"Failed to load denoised train cache after creation/check: {train_dataset_path}"
                    ) from e
                logger.warning(
                    "Failed to load train cache (%s), trying train-only fallback: %s",
                    train_dataset_path, e,
                )
                # If validation split is missing but train exists, load train only and skip re-tokenizing
                train_split_path = os.path.join(train_dataset_path, "train")
                if os.path.exists(train_split_path):
                    try:
                        train_ds = datasets.Dataset.load_from_disk(train_split_path)
                        split = train_ds.train_test_split(test_size=0.01, seed=42)
                        val_ds = split["test"]
                        val_ds = val_ds.select(range(min(1000, len(val_ds))))
                        train_datasets = datasets.DatasetDict({
                            "train": split["train"],
                            "validation": val_ds,
                        })
                        logger.info(
                            "Loaded train from cache, created validation from train split "
                            "(skipped re-tokenization). train=%d, validation=%d",
                            len(train_datasets["train"]), len(train_datasets["validation"]),
                        )
                        train_datasets["validation"].set_format("pt")
                        train_datasets["train"].set_format("pt")
                        # Save validation only directly (~50k samples vs ~5M train - much faster)
                        validation_save_path = os.path.join(train_dataset_path, "validation")
                        if os.path.exists(validation_save_path):
                            shutil.rmtree(validation_save_path)
                        train_datasets["validation"].save_to_disk(validation_save_path)
                        # Update dataset_dict.json so next load finds both splits
                        dataset_dict_json = os.path.join(train_dataset_path, "dataset_dict.json")
                        if os.path.exists(dataset_dict_json):
                            with open(dataset_dict_json) as f:
                                d = json.load(f)
                        else:
                            d = {}
                        d.update({"train": "train", "validation": "validation"})
                        with open(dataset_dict_json, "w") as f:
                            json.dump(d, f, indent=2)
                        print("saved validation split to path:", train_dataset_path)
                    except (FileNotFoundError, OSError) as inner_e:
                        logger.warning(
                            "Train-only fallback failed (%s), doing full uncached: %s",
                            train_split_path, inner_e,
                        )
                        train_datasets = self._load_train_dataset_uncached(
                            model=model,
                            tokenizer=tokenizer,
                            embedder_tokenizer=embedder_tokenizer,
                        )
                        print("saving train_dataset to path:", train_dataset_path)
                        train_datasets.save_to_disk(
                            train_dataset_path,
                            max_shard_size="2GB",
                        )
                else:
                    train_datasets = self._load_train_dataset_uncached(
                        model=model,
                        tokenizer=tokenizer,
                        embedder_tokenizer=embedder_tokenizer,
                    )
                    print("saving train_dataset to path:", train_dataset_path)
                    train_datasets.save_to_disk(
                        train_dataset_path,
                        max_shard_size="2GB",
                    )
        else:
            if dataset_mode == "denoised":
                raise FileNotFoundError(
                    f"Denoised train cache does not exist after creation/check: {train_dataset_path}"
                )
            train_datasets = self._load_train_dataset_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            print("saving train_dataset to path:", train_dataset_path)
            train_datasets.save_to_disk(
                train_dataset_path,
                max_shard_size="2GB",
            )
        ######################################################################
        val_dataset_kwargs = {
            "dataset_name": "__".join(
                ["ag_news", "arxiv", "xsum_doc", "xsum_summ", "wikibio"]
            ),
            **dataset_kwargs,
        }
        # Use readable cache name instead of MD5 hash
        val_dataset_path = self._get_embedding_cache_path(split="val")
        val_dataset_path_override = os.environ.get(
            "DAEI_VAL_DATASET_PATH"
        ) or os.environ.get("VEC2TEXT_VAL_DATASET_PATH")
        if val_dataset_path_override:
            val_dataset_path = val_dataset_path_override
            print("loading val dataset from env path:", val_dataset_path)
        assert val_dataset_path != train_dataset_path
        if dataset_mode == "denoised":
            self._ensure_denoised_cache(
                denoised_path=val_dataset_path,
                split="val",
                model=model,
            )
        if os.path.exists(val_dataset_path):
            try:
                val_datasets_dict = datasets.load_from_disk(val_dataset_path)
                print("loaded dict of val datasets from", val_dataset_path)
                requested_eval = self.data_args.max_eval_samples
                if (
                    requested_eval is not None
                    and not val_dataset_path_override
                    and dataset_mode != "denoised"
                    and isinstance(val_datasets_dict, datasets.DatasetDict)
                    and val_datasets_dict
                ):
                    short_splits = {
                        k: len(v)
                        for k, v in val_datasets_dict.items()
                        if len(v) < requested_eval
                    }
                    if short_splits:
                        logger.warning(
                            "Cached val dataset %s is smaller than --max_eval_samples=%s "
                            "for splits %s; regenerating it.",
                            val_dataset_path,
                            requested_eval,
                            short_splits,
                        )
                        print(
                            "cached val dataset is smaller than --max_eval_samples; "
                            f"regenerating {val_dataset_path}: {short_splits}"
                        )
                        val_datasets_dict = self._load_val_datasets_uncached(
                            model=model,
                            tokenizer=tokenizer,
                            embedder_tokenizer=embedder_tokenizer,
                        )
                        print("saving val_dataset to path:", val_dataset_path)
                        val_datasets_dict.save_to_disk(val_dataset_path)
            except (FileNotFoundError, OSError) as e:
                if dataset_mode == "denoised":
                    raise RuntimeError(
                        f"Failed to load denoised val cache after creation/check: {val_dataset_path}"
                    ) from e
                logger.warning(
                    "Failed to load val cache (%s), falling back to uncached: %s",
                    val_dataset_path, e,
                )
                if val_dataset_path_override:
                    raise RuntimeError(
                        f"Validation dataset path override is set ({val_dataset_path}) but load failed: {e}"
                    ) from e
                val_datasets_dict = self._load_val_datasets_uncached(
                    model=model,
                    tokenizer=tokenizer,
                    embedder_tokenizer=embedder_tokenizer,
                )
                print("saving val_dataset to path:", val_dataset_path)
                val_datasets_dict.save_to_disk(val_dataset_path)
            print("loaded dict of val datasets from", val_dataset_path)
        else:
            if val_dataset_path_override:
                raise FileNotFoundError(
                    f"Validation dataset path override is set but path does not exist: {val_dataset_path}"
                )
            if dataset_mode == "denoised":
                raise FileNotFoundError(
                    f"Denoised val cache does not exist after creation/check: {val_dataset_path}"
                )
            val_datasets_dict = self._load_val_datasets_uncached(
                model=model,
                tokenizer=tokenizer,
                embedder_tokenizer=embedder_tokenizer,
            )
            print("saving val_dataset to path:", val_dataset_path)
            val_datasets_dict.save_to_disk(val_dataset_path)
        ######################################################################
        val_datasets_dict[self.data_args.dataset_name] = train_datasets["validation"]
        train_dataset = train_datasets["train"]

        self._ensure_val_clean_embeddings_if_needed(
            model=model,
            tokenizer=tokenizer,
            embedder_tokenizer=embedder_tokenizer,
            train_datasets=train_datasets,
            val_datasets_dict=val_datasets_dict,
        )

        for key in val_datasets_dict:
            new_length = min(
                len(val_datasets_dict[key]), self.data_args.max_eval_samples
            )
            val_datasets_dict[key] = val_datasets_dict[key].select(range(new_length))

        return (train_dataset, val_datasets_dict)


class InversionExperiment(Experiment):
    @property
    def trainer_cls(self):
        return DAEI.trainers.InversionTrainer

    @property
    def _wandb_project_name(self) -> str:
        return "emb-inv-4"

    def load_model(self) -> transformers.PreTrainedModel:
        return InversionModel(
            config=self.config,
        )

    def load_trainer(self) -> transformers.Trainer:
        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"
        )

        if self.training_args.mock_embedder:
            # This mode allows us to get the embedders off the GPU during training
            # once we've computed all the embeddings we need. :)
            assert (
                model.config.use_frozen_embeddings_as_input
            ), "must use frozen embeddings if mock_embedder=True"
            print(
                "IMPORTANT: Mocking embedder for the rest of training (to save GPU memory)."
                " Do not trust embedding-based evaluation metrics."
            )
            model.embedder.cpu()
            del model.embedder
            model.embedder = MockEmbedder(embedder_dim=model.embedder_dim)

        return self.trainer_cls(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
        )


class CorrectorExperiment(Experiment):
    @property
    def _wandb_project_name(self) -> str:
        return "emb-correct-1"

    def _inverter_checkpoint(self) -> Optional[str]:
        return getattr(self.training_args, "inverter_checkpoint", None) or getattr(
            self.training_args, "corrector_model_from_pretrained", None
        )

    def load_trainer(self) -> transformers.Trainer:
        inverter_checkpoint = self._inverter_checkpoint()
        if not inverter_checkpoint:
            raise ValueError(
                "Corrector training requires --inverter_checkpoint. "
                "Legacy alias loading has been removed."
            )
        (
            _,
            inversion_trainer,
        ) = DAEI.analyze_utils.load_experiment_and_trainer_from_pretrained(
            name=inverter_checkpoint,
            # max_seq_length=self.model_args.max_seq_length,
            use_less_data=self.data_args.use_less_data,
            data_args_overrides={
                "dataset_name": self.data_args.dataset_name,
                "dataset_mode": getattr(self.data_args, "dataset_mode", "standard"),
                "use_full_data_datasets": getattr(
                    self.data_args, "use_full_data_datasets", None
                ),
            },
            model_args_overrides={
                "embedder_gaussian_noise_level": getattr(
                    self.model_args, "embedder_gaussian_noise_level", 0.0
                ),
            },
        )
        # Sync bf16/fp16 so Corrector.__init__ assertion passes (pretrained loader may set bf16=0).
        inversion_trainer.args.bf16 = self.training_args.bf16
        inversion_trainer.args.fp16 = self.training_args.fp16
        model = self.load_model(inversion_trainer=inversion_trainer)
        return DAEI.trainers.Corrector(
            model=model,
            inversion_trainer=inversion_trainer,
            args=self.training_args,
            data_collator=DataCollatorForCorrection(
                tokenizer=inversion_trainer.model.tokenizer
            ),
            corrector_cache_path=CORRECTOR_CACHE_PATH,
        )

    def load_model(self, inversion_trainer) -> transformers.PreTrainedModel:
        return CorrectorEncoderModel(
            config=self.config,
        )


class DAEShieldCorrectorExperiment(CorrectorExperiment):
    """Corrector with a frozen DAE Shield for embedding denoising.

    Extends vanilla CorrectorExperiment by:
      1. Loading a frozen ResidualDAE from ``--dae_checkpoint``.
      2. Passing it to the Corrector trainer, which injects it into both the
         CorrectorEncoderModel (for denoising target & hypothesis embeddings
         before the diff/transform pipeline) and the generation loop (for
         noise-aware early stopping).
    """

    @property
    def _wandb_project_name(self) -> str:
        return "emb-correct-dae-shield"

    def _load_dae(self):
        """Instantiate a ResidualDAE and load weights from --dae_checkpoint."""
        from DAEI.models.dae import ResidualDAE

        ta = self.training_args
        ckpt_dir = ta.dae_checkpoint
        if not ckpt_dir:
            raise ValueError(
                "DAEShieldCorrectorExperiment requires --dae_checkpoint "
                "pointing to a JointExperiment checkpoint directory."
            )

        dae = ResidualDAE(
            emb_dim=768,  # GTR-base
            hidden_dim=ta.dae_hidden_dim,
            depth=ta.dae_depth,
            use_sigma_cond=getattr(ta, "dae_use_sigma_cond", False),
            use_spectral_norm=getattr(ta, "dae_use_spectral_norm", False),
        )

        # Try legacy standalone dae.pt first
        legacy_path = os.path.join(ckpt_dir, "dae.pt")
        if os.path.isfile(legacy_path):
            logger.info("Loading DAE from legacy dae.pt: %s", legacy_path)
            try:
                state = torch.load(legacy_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(legacy_path, map_location="cpu")
            dae.load_state_dict(state)
        else:
            # Extract DAE keys from the full model checkpoint
            weights_path = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"No dae.pt / model.safetensors / pytorch_model.bin in {ckpt_dir}"
                )
            logger.info("Extracting DAE weights from %s", weights_path)
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                full_state = load_file(weights_path)
            else:
                try:
                    full_state = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    full_state = torch.load(weights_path, map_location="cpu")
            dae_state = {
                k.replace("dae.", "", 1): v
                for k, v in full_state.items() if k.startswith("dae.")
            }
            if not dae_state:
                raise ValueError(
                    f"No 'dae.' prefixed keys found in {weights_path}. "
                    "Is this a JointExperiment checkpoint?"
                )
            dae.load_state_dict(dae_state)
            logger.info("Loaded %d DAE keys from model checkpoint", len(dae_state))
            del full_state

        for p in dae.parameters():
            p.requires_grad = False
        dae.eval()
        n_params = sum(p.numel() for p in dae.parameters())
        logger.info("Frozen DAE ready: %.2fM params", n_params / 1e6)
        return dae

    def load_trainer(self) -> transformers.Trainer:
        # Reuse CorrectorExperiment's inversion_trainer loading logic
        inverter_checkpoint = self._inverter_checkpoint()
        if not inverter_checkpoint:
            raise ValueError(
                "DAE Shield corrector training requires --inverter_checkpoint. "
                "Legacy alias loading has been removed."
            )
        (
            _,
            inversion_trainer,
        ) = DAEI.analyze_utils.load_experiment_and_trainer_from_pretrained(
            name=inverter_checkpoint,
            use_less_data=self.data_args.use_less_data,
            data_args_overrides={
                "dataset_name": self.data_args.dataset_name,
                "dataset_mode": getattr(self.data_args, "dataset_mode", "standard"),
                "use_full_data_datasets": getattr(
                    self.data_args, "use_full_data_datasets", None
                ),
            },
            model_args_overrides={
                "embedder_gaussian_noise_level": getattr(
                    self.model_args, "embedder_gaussian_noise_level", 0.0
                ),
            },
        )
        inversion_trainer.args.bf16 = self.training_args.bf16
        inversion_trainer.args.fp16 = self.training_args.fp16

        model = self.load_model(inversion_trainer=inversion_trainer)
        dae = self._load_dae()

        return DAEI.trainers.Corrector(
            model=model,
            inversion_trainer=inversion_trainer,
            args=self.training_args,
            data_collator=DataCollatorForCorrection(
                tokenizer=inversion_trainer.model.tokenizer
            ),
            dae_model=dae,
            corrector_cache_path=CORRECTOR_CACHE_PATH,
        )


class JointExperiment(InversionExperiment):
    """DAE + inverter joint training experiment (stages 0/1/2)."""

    @property
    def _wandb_project_name(self) -> str:
        return "sure-ce-joint"

    def _ensure_val_clean_embeddings_if_needed(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.AutoTokenizer,
        embedder_tokenizer: transformers.AutoTokenizer,
        train_datasets: datasets.DatasetDict,
        val_datasets_dict: datasets.DatasetDict,
    ) -> None:
        """For stage-1 SURE: re-embed each val dataset without noise → add as clean_embeddings.

        Instead of independently regenerating val datasets (which may differ from
        the cached noisy val in size/content), we directly re-embed the *already
        loaded* ``val_datasets_dict`` entries with ``embed_dataset_batch(model)``
        (noise_sigma=0). This guarantees identical rows/lengths.

        Cache: ``<dataset>_clean_<use_less_data>_<embedder>_val.arrow``
        """
        ta = self.training_args
        if ta.training_stage < 1 or (getattr(ta, "loss_mode", "n2n") or "n2n") != "sure":
            return
        # Auxiliary val sets may still lack clean_embeddings while the main val split
        # already has them (from training cache). Merge from clean cache per key until all
        # splits that appear in clean_val have clean_embeddings.
        if all(
            "clean_embeddings" in val_datasets_dict[k].column_names
            for k in val_datasets_dict
        ):
            return

        dataset_name_str = self.data_args.dataset_name_str
        use_less_data = str(self.data_args.use_less_data)
        embedder_name = self.model_args.embedder_model_name
        if embedder_name == "gte_base":
            embedder_name = "gte_base_mean_norm"

        def _sanitize(s: str) -> str:
            for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                s = s.replace(c, '_')
            return s

        fd_names = getattr(self.data_args, "use_full_data_dataset_names", None) or []
        fd_suffix = (
            "_fd_" + _sanitize("_".join(sorted(fd_names)))
            if fd_names
            else ""
        )

        clean_cache_name = (
            f"{_sanitize(dataset_name_str)}_clean_{_sanitize(use_less_data)}"
            f"{fd_suffix}_{_sanitize(embedder_name)}_val"
        )
        clean_cache_path = os.path.join(DATASET_CACHE_PATH, clean_cache_name + ".arrow")

        if os.path.exists(clean_cache_path):
            logger.info("Loading clean val cache: %s", clean_cache_path)
            clean_val = datasets.load_from_disk(clean_cache_path)
        else:
            logger.info("Generating clean val cache by re-embedding val without noise …")
            assert torch.cuda.is_available()
            model = model.to(torch.device("cuda"))

            clean_val_dict = {}
            for key, ds in val_datasets_dict.items():
                if "input_ids" not in ds.column_names:
                    logger.warning("Val '%s' has no input_ids — skipping clean embedding.", key)
                    continue
                logger.info("  Embedding val '%s' (%d samples) without noise …", key, len(ds))
                clean_ds = dataset_map_multi_worker(
                    dataset=ds,
                    map_fn=functools.partial(embed_dataset_batch, model),
                    batched=True,
                    batch_size=ta.per_device_train_batch_size,
                    new_fingerprint=make_hf_fingerprint(
                        ds._fingerprint, "clean", key
                    ),
                    num_proc=1,
                )
                clean_val_dict[key] = clean_ds

            clean_val = datasets.DatasetDict(clean_val_dict)
            os.makedirs(DATASET_CACHE_PATH, exist_ok=True)
            clean_val.save_to_disk(clean_cache_path)
            logger.info("Saved clean val cache: %s", clean_cache_path)

        # Merge: frozen_embeddings from clean cache → clean_embeddings in noisy val
        for key in list(val_datasets_dict.keys()):
            if key not in clean_val:
                continue
            if "clean_embeddings" in val_datasets_dict[key].column_names:
                continue
            noisy_len = len(val_datasets_dict[key])
            clean_len = len(clean_val[key])
            if noisy_len != clean_len:
                logger.warning(
                    "Length mismatch for val '%s': noisy=%d, clean=%d — skipping.",
                    key, noisy_len, clean_len,
                )
                continue
            # .with_format(None) ensures raw Arrow/numpy data, not PyTorch tensors
            clean_embs = clean_val[key].with_format(None)["frozen_embeddings"]
            val_datasets_dict[key] = val_datasets_dict[key].add_column(
                "clean_embeddings", clean_embs,
            )
            logger.info("Added clean_embeddings to val '%s' (%d samples)", key, noisy_len)

    def load_trainer(self) -> transformers.Trainer:
        from DAEI.models.dae import ResidualDAE
        from DAEI.trainers.joint import JointTrainer

        model = self.load_model()
        train_dataset, eval_dataset = self.load_train_and_val_datasets(
            model=model,
            tokenizer=model.tokenizer,
            embedder_tokenizer=model.embedder_tokenizer,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            "JointExperiment: v2t params = %.2fM", n_params / 2**20
        )

        if self.training_args.mock_embedder:
            assert model.config.use_frozen_embeddings_as_input
            model.embedder.cpu()
            del model.embedder
            model.embedder = MockEmbedder(embedder_dim=model.embedder_dim)

        ta = self.training_args
        stage = ta.training_stage

        # --- Load v2t weights ---
        # DAEI pipeline stages (1 / 3) do NOT use stage0_checkpoint:
        #   Stage 1: T5 is frozen and unused — no baseline needed.
        #   Stage 3: v2t weights come entirely from stage2_checkpoint.
        # The old pipeline's stage 2 (joint SURE+CE) still loads stage0 for
        # backward compat, but the DAEI pipeline is fully self-contained.
        if stage == 1:
            if ta.stage0_checkpoint:
                logger.info(
                    "Stage 1 (DAE-only): ignoring --stage0_checkpoint=%s. "
                    "T5 decoder is frozen and not used; no baseline needed.",
                    ta.stage0_checkpoint,
                )
        elif stage == 2 and ta.stage0_checkpoint:
            # Old pipeline's stage 2 (joint SURE+CE from baseline)
            ckpt_dir = ta.stage0_checkpoint
            logger.info("Loading stage-0 v2t checkpoint (old pipeline): %s", ckpt_dir)
            weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"No pytorch_model.bin or model.safetensors in {ckpt_dir}. "
                    "Point --stage0_checkpoint to your baseline or stage0 output_dir."
                )
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                ckpt_state = load_file(weights_path)
            else:
                try:
                    ckpt_state = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt_state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(ckpt_state, strict=False)
            del ckpt_state
        elif stage == 3 and ta.stage0_checkpoint:
            logger.info(
                "Stage 3 (DAEI): ignoring --stage0_checkpoint=%s. "
                "v2t weights come from --stage2_checkpoint; no baseline needed.",
                ta.stage0_checkpoint,
            )

        # Stage 3 (DAEI): load complete v2t from Stage 2 Inverter checkpoint.
        # This is a self-contained model (T5 + projection + embedder), no baseline needed.
        stage2_ckpt = _stage2_checkpoint(ta)
        if stage == 3 and stage2_ckpt:
            inv_ckpt = stage2_ckpt
            logger.info("Loading Stage 2 Inverter checkpoint: %s", inv_ckpt)
            weights_path = os.path.join(inv_ckpt, "model.safetensors")
            if not os.path.isfile(weights_path):
                weights_path = os.path.join(inv_ckpt, "pytorch_model.bin")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(
                    f"No model.safetensors or pytorch_model.bin in {inv_ckpt}"
                )
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                inv_state = load_file(weights_path)
            else:
                try:
                    inv_state = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    inv_state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(inv_state, strict=False)
            logger.info("Loaded Stage 2 Inverter weights (%d keys)", len(inv_state))
            del inv_state

        # --- Create DAE (stages 1 & 2) ---
        dae = None
        if stage >= 1:
            dae = ResidualDAE(
                emb_dim=model.embedder_dim,
                hidden_dim=ta.dae_hidden_dim,
                depth=ta.dae_depth,
                use_sigma_cond=getattr(ta, "dae_use_sigma_cond", False),
                use_spectral_norm=getattr(ta, "dae_use_spectral_norm", False),
            )
            dae_params = sum(p.numel() for p in dae.parameters())
            logger.info("Created DAE: %.2fM params", dae_params / 2**20)

            # Load stage-1 DAE checkpoint if entering stage 2.
            # DAE weights live inside the model checkpoint with 'dae.' prefix
            # (since JointTrainer mounts it as self.model.dae).
            # We also support legacy standalone dae.pt files.
            if stage >= 2 and ta.stage1_checkpoint:
                s1_dir = ta.stage1_checkpoint
                legacy_dae = os.path.join(s1_dir, "dae.pt")
                if os.path.isfile(legacy_dae):
                    logger.info("Loading DAE from legacy dae.pt: %s", legacy_dae)
                    try:
                        dae.load_state_dict(torch.load(legacy_dae, map_location="cpu", weights_only=True))
                    except TypeError:
                        dae.load_state_dict(torch.load(legacy_dae, map_location="cpu"))
                else:
                    # Extract DAE keys from the full model checkpoint
                    s1_weights = os.path.join(s1_dir, "model.safetensors")
                    if not os.path.isfile(s1_weights):
                        s1_weights = os.path.join(s1_dir, "pytorch_model.bin")
                    if os.path.isfile(s1_weights):
                        logger.info("Extracting DAE weights from %s", s1_weights)
                        if s1_weights.endswith(".safetensors"):
                            from safetensors.torch import load_file
                            full_state = load_file(s1_weights)
                        else:
                            try:
                                full_state = torch.load(s1_weights, map_location="cpu", weights_only=True)
                            except TypeError:
                                full_state = torch.load(s1_weights, map_location="cpu")
                        dae_state = {
                            k.replace("dae.", "", 1): v
                            for k, v in full_state.items() if k.startswith("dae.")
                        }
                        if dae_state:
                            dae.load_state_dict(dae_state)
                            logger.info("Loaded %d DAE keys from model checkpoint", len(dae_state))
                        else:
                            logger.warning("No 'dae.' keys found in %s", s1_weights)
                        del full_state

        return JointTrainer(
            model=model,
            args=ta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.get_collator(tokenizer=model.tokenizer),
            dae=dae,
            training_stage=stage,
            loss_mode=ta.loss_mode,
            noise_sigma=ta.noise_sigma,
            sure_n_probes=ta.sure_n_probes,
            lambda_ce=ta.lambda_ce,
            lambda_warmup_steps=ta.lambda_warmup_steps,
            dae_lr=ta.dae_lr,
            # DAEI pipeline enhancements
            dae_contrastive_weight=getattr(ta, "dae_contrastive_weight", 0.0),
            dae_contrastive_tau=getattr(ta, "dae_contrastive_tau", 0.07),
            dae_sigma_schedule=getattr(ta, "dae_sigma_schedule", "fixed"),
            dae_sigma_max=getattr(ta, "dae_sigma_max", 0.05),
            dae_sigma_min=getattr(ta, "dae_sigma_min", 0.01),
            dae_sigma_decay_span_fraction=getattr(ta, "dae_sigma_decay_span_fraction", 0.2),
            dae_grad_max_norm=getattr(ta, "dae_grad_max_norm", 0.0),
            pcgrad=getattr(ta, "dae_pcgrad", False),
        )


EXPERIMENT_CLS_MAP = {
    "inversion": InversionExperiment,
    "corrector": CorrectorExperiment,
    "corrector_encoder": CorrectorExperiment,
    "joint_inversion": JointExperiment,
    "dae_shield_corrector": DAEShieldCorrectorExperiment,
}


def experiment_from_args(model_args, data_args, training_args) -> Experiment:
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]  # type: ignore
    else:
        raise ValueError(f"Unknown experiment {training_args.experiment}")
    return experiment_cls(model_args, data_args, training_args)  # type: ignore

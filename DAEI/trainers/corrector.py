import functools
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from DAEI.models import CorrectorEncoderModel
from DAEI.models.model_utils import freeze_params
from DAEI.run_args import TrainingArguments
from DAEI.utils import dataset_map_multi_worker

from .base import BaseTrainer, filter_eval_metrics_for_log
from .inversion import InversionTrainer

logger = logging.getLogger(__name__)


def _sanitize_cache_component(s: str) -> str:
    """Same rules as Experiment._get_readable_embedding_cache_name."""
    invalid_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]
    for c in invalid_chars:
        s = s.replace(c, "_")
    return s


def _checkpoint_cache_tag(value: Optional[str], prefix: str) -> Optional[str]:
    if not value:
        return None
    normalized = os.path.abspath(os.path.expanduser(str(value)))
    digest = hashlib.md5(normalized.encode()).hexdigest()[:8]
    leaf = os.path.basename(normalized.rstrip(os.sep)) or "checkpoint"
    return f"{prefix}_{_sanitize_cache_component(leaf)}_{digest}"


def _dataset_name_str_from_raw(dataset_name: Optional[str]) -> str:
    """Match DataArguments.dataset_name_str (e.g. nq,msmarco -> nq_msmarco)."""
    if not dataset_name:
        return "unknown"
    if isinstance(dataset_name, list):
        parts = [str(x).strip() for x in dataset_name if str(x).strip()]
    else:
        parts = [n.strip() for n in str(dataset_name).split(",") if n.strip()]
    return "_".join(parts) if parts else "unknown"


def align_targets_to_hypotheses(
    target_embeddings: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
) -> torch.Tensor:
    target_batch = target_embeddings.shape[0]
    hypothesis_batch = hypothesis_embeddings.shape[0]
    if target_batch == hypothesis_batch:
        return target_embeddings
    if hypothesis_batch > target_batch and hypothesis_batch % target_batch == 0:
        repeat_factor = hypothesis_batch // target_batch
        return target_embeddings.repeat_interleave(repeat_factor, dim=0)
    if target_batch > hypothesis_batch and target_batch % hypothesis_batch == 0:
        repeat_factor = target_batch // hypothesis_batch
        return target_embeddings.reshape(
            hypothesis_batch, repeat_factor, *target_embeddings.shape[1:]
        )[:, 0]
    raise RuntimeError(
        "Cannot align target and hypothesis embeddings: "
        f"target batch={target_batch}, "
        f"hypothesis batch={hypothesis_batch}"
    )


def collapse_beam_outputs_to_batch(
    generated_ids: torch.Tensor,
    hypothesis_embeddings: torch.Tensor,
    target_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output_batch = generated_ids.shape[0]
    if output_batch == target_batch_size:
        return generated_ids, hypothesis_embeddings
    if output_batch > target_batch_size and output_batch % target_batch_size == 0:
        beam_width = output_batch // target_batch_size
        generated_ids = generated_ids.reshape(target_batch_size, beam_width, -1)[:, 0]
        hypothesis_embeddings = hypothesis_embeddings.reshape(
            target_batch_size, beam_width, *hypothesis_embeddings.shape[1:]
        )[:, 0]
        return generated_ids, hypothesis_embeddings
    raise RuntimeError(
        "Cannot collapse generated beam outputs: "
        f"output batch={output_batch}, target batch={target_batch_size}"
    )


def _best_scores_are_unchanged(
    current_scores: torch.Tensor,
    previous_scores: torch.Tensor,
    atol: float = 1e-3,
) -> bool:
    if current_scores.shape != previous_scores.shape:
        return False
    return bool(torch.isclose(current_scores, previous_scores, atol=atol).all().item())


class Corrector(BaseTrainer):
    """Trains an encoder model to generate embeddings that recursively correct of an
    InversionTrainer.
    """

    train_dataset: datasets.Dataset
    eval_dataset: Dict[str, datasets.Dataset]
    # TODO: don't assume that the encoder has to have the same tokenizer as the encoder_decoder
    # or embedder model.

    _hypothesis_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    # If set, only take hypothesis if it improves our distance to ground-truth.
    return_best_hypothesis: bool = False

    # Initialize from this hypothesis, if set
    initial_hypothesis_str: Optional[str] = None

    def __init__(
        self,
        model: CorrectorEncoderModel,
        inversion_trainer: InversionTrainer,
        args: Optional[TrainingArguments],
        dae_model: Optional[nn.Module] = None,
        corrector_cache_path: Optional[str] = None,
        **kwargs,
    ):
        # Freeze other model params
        freeze_params(inversion_trainer.model)
        # We're training this corrector model to correct outputs from
        # a model trained & loaded via the inversion trainer.
        self.inversion_trainer = inversion_trainer
        self.inversion_trainer.model.use_frozen_embeddings_as_input = True
        # dae_model is consumed here — do NOT let it leak into **kwargs / super()
        super().__init__(
            model=model,
            args=args,
            train_dataset=self.inversion_trainer.train_dataset,
            eval_dataset=self.inversion_trainer.eval_dataset,
            **kwargs,
        )
        self.tokenizer = self.inversion_trainer.model.tokenizer
        self.embedder_tokenizer = self.inversion_trainer.model.embedder_tokenizer
        self.embedder = self.inversion_trainer.embedder
        self.call_embedding_model = self.inversion_trainer.model.call_embedding_model

        self.initial_hypothesis_str = None

        # Number of steps of self-correction
        self.num_gen_recursive_steps = 1
        self.sequence_beam_width = 1

        # If set, return closest (in embedding space) hypothesis we see during generation
        self.return_best_hypothesis = False
        self._dae_hypothesis_call_verified = False
        self.corrector_cache_path = (
            corrector_cache_path
            or os.environ.get("DAEI_CACHE")
            or os.environ.get("VEC2TEXT_CACHE")
            or os.path.join("data", "corrector_cache")
        )

        # Data-level σ for DAE inference / set_dae — must exist before set_dae() below.
        self._dae_noise_sigma: float = float(getattr(args, "noise_sigma", 0.01))

        # DAE Shield: inject a frozen DAE into the CorrectorEncoderModel so
        # that both target and hypothesis embeddings are denoised before the
        # diff/transform pipeline.
        self.dae: Optional[nn.Module] = None
        if dae_model is not None:
            self.dae = dae_model
            for p in self.dae.parameters():
                p.requires_grad = False
            self.dae.eval()
            self.model.set_dae(dae_model, noise_sigma=self._dae_noise_sigma)
            logger.info("DAE Shield enabled in Corrector")

        # Noise-aware early stopping parameters
        self.early_stop_threshold: float = getattr(args, "early_stop_threshold", 1e-3)
        self.max_correction_steps: int = getattr(args, "max_correction_steps", 5)

        # Need to train with same device as the inversion model to avoid weird errors.
        assert self.args.fp16 == self.inversion_trainer.args.fp16
        assert self.args.bf16 == self.inversion_trainer.args.bf16

    def _dae_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Return σ tensor for DAE inference calls (data-level noise)."""
        return torch.full(
            (x.shape[0],), self._dae_noise_sigma, device=x.device, dtype=x.dtype
        )

    def _beam_distances_to_targets(
        self,
        hypothesis_embedding: torch.Tensor,
        target_embeddings: torch.Tensor,
        batch_size: int,
        beam_width: int,
    ) -> torch.Tensor:
        target_embeddings = align_targets_to_hypotheses(
            target_embeddings, hypothesis_embedding
        )
        if self.dae is not None:
            with torch.no_grad():
                target_embeddings = self.dae(
                    target_embeddings, sigma=self._dae_sigma(target_embeddings)
                )
                hypothesis_embedding = self.dae(
                    hypothesis_embedding, sigma=self._dae_sigma(hypothesis_embedding)
                )
        return torch.nn.CosineSimilarity(dim=2)(
            hypothesis_embedding.reshape((batch_size, beam_width, -1)),
            target_embeddings.reshape((batch_size, beam_width, -1)),
        )

    def evaluation_loop(
        self, dataloader: torch.utils.data.DataLoader, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.

        Override to compute ppl from eval loss.
        """
        self.inversion_trainer.model.to(self.args.device)
        if self.dae is not None:
            self.dae.to(self.args.device)

        metric_key_prefix = kwargs["metric_key_prefix"]
        output = super().evaluation_loop(dataloader=dataloader, *args, **kwargs)  # type: ignore
        if metric_key_prefix in {"eval_msmarco", "eval_nq"}:
            n_rounds = self.max_correction_steps if self.dae is not None else 5
            self.num_gen_recursive_steps = n_rounds
            multi_round_generation_metrics = self.eval_generation_metrics(
                dataloader=dataloader
            )
            multiround_generation_metrics = {
                f"{metric_key_prefix}_{n_rounds}round_{k}": v
                for k, v in multi_round_generation_metrics.items()
            }
            output.metrics.update(multiround_generation_metrics)
            self.num_gen_recursive_steps = 1

        self.inversion_trainer.model.cpu()
        self._broadcast_metric_for_best_model_if_missing(output.metrics)
        if getattr(self.args, "eval_log_text_metrics_only", False):
            filtered = filter_eval_metrics_for_log(
                output.metrics,
                getattr(self.args, "metric_for_best_model", None),
            )
            output.metrics.clear()
            output.metrics.update(filtered)
        return output

    def _precompute_hypothesis_and_embedding(
        self,
        ds_inputs: Dict[str, torch.Tensor],
        collator=None,
    ) -> Dict[str, torch.Tensor]:
        assert not self.model.training
        inputs = collator.tokenizer.pad(
            {k: v for k, v in ds_inputs.items() if k != "labels"},
            padding=collator.padding,
            max_length=collator.max_length,
            pad_to_multiple_of=collator.pad_to_multiple_of,
            return_tensors=collator.return_tensors,
        ).to(self.args.device)

        (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_embedding,
        ) = self._get_hypothesis_uncached(inputs=inputs)
        ds_inputs["frozen_embeddings"] = frozen_embeddings.cpu()
        ds_inputs["hypothesis_embedding"] = hypothesis_embedding.cpu()

        # cut padding so we can batch by length later
        ds_inputs["hypothesis_input_ids"] = []
        ds_inputs["hypothesis_attention_mask"] = []
        for input_ids, attention_mask in zip(
            hypothesis_input_ids.cpu(), hypothesis_attention_mask.cpu()
        ):
            num_tokens = attention_mask.sum()
            ds_inputs["hypothesis_input_ids"].append(input_ids[: num_tokens + 1])
            ds_inputs["hypothesis_attention_mask"].append(
                attention_mask[: num_tokens + 1]
            )
        print("input_ids[0]:", self.tokenizer.decode(ds_inputs["input_ids"][0]))
        print(
            "hypothesis_input_ids[0]:",
            self.tokenizer.decode(ds_inputs["hypothesis_input_ids"][0]),
        )
        return ds_inputs

    def _readable_hypotheses_stem(self, split_suffix: str) -> Optional[str]:
        """Stem aligned with Experiment._get_readable_embedding_cache_name (e.g. nq_msmarco_noisy_5000000_gtr_base_train)."""
        cfg = getattr(self.inversion_trainer.model, "config", None)
        if cfg is None:
            return None
        dataset_name = getattr(cfg, "dataset_name", None)
        if dataset_name is None:
            return None
        dataset_stem = _dataset_name_str_from_raw(dataset_name)
        dataset_mode = getattr(cfg, "dataset_mode", "standard") or "standard"
        use_less_data = str(getattr(cfg, "use_less_data", -1))
        embedder_name = getattr(cfg, "embedder_model_name", "gtr_base") or "gtr_base"
        fd_raw = getattr(cfg, "use_full_data_datasets", None) or ""
        fd_names = [
            n.strip() for n in str(fd_raw).split(",") if n.strip()
        ] if fd_raw else []
        fd_suffix = (
            "_fd_" + _sanitize_cache_component("_".join(sorted(fd_names)))
            if fd_names
            else ""
        )
        identity_tags = []
        source_tag = _checkpoint_cache_tag(
            getattr(self.args, "inverter_checkpoint", None)
            or getattr(self.args, "corrector_model_from_pretrained", None),
            "src",
        )
        if source_tag:
            identity_tags.append(source_tag)
        if self.dae is not None:
            dae_tag = _checkpoint_cache_tag(
                getattr(self.args, "dae_checkpoint", None),
                "dae",
            )
            identity_tags.append(dae_tag or "dae_unknown")
        identity_suffix = "_" + "_".join(identity_tags) if identity_tags else ""
        return (
            f"{_sanitize_cache_component(dataset_stem)}_"
            f"{_sanitize_cache_component(str(dataset_mode))}_"
            f"{_sanitize_cache_component(use_less_data)}"
            f"{fd_suffix}_"
            f"{_sanitize_cache_component(str(embedder_name))}_"
            f"{_sanitize_cache_component(split_suffix)}"
            f"{identity_suffix}"
        )

    def _preprocess_dataset_hypotheses(
        self,
        dataset: datasets.Dataset,
        filter_correct_examples: bool = False,
        split_suffix: str = "train",
    ) -> Tuple[datasets.Dataset, str]:
        #
        # In each model directory, we store a copy of the dataset with hypotheses
        # generated by the model that's checkpointed in this directory. This
        # won't scale well, but hopefully we don't do this with too many models,
        # and precomputing 5M hypotheses on A100 takes ~8 hours, so they're worth
        # storing.
        #
        # Cache dir name matches embedding .arrow naming (see Experiment._get_readable_embedding_cache_name):
        #   {dataset}_{mode}_{use_less}_{embedder}_{split}_hypotheses.cache
        # Legacy path {fingerprint}_hypotheses.cache is still loaded if present.
        cache_dir = self.corrector_cache_path
        os.makedirs(cache_dir, exist_ok=True)
        readable_stem = self._readable_hypotheses_stem(split_suffix)
        preferred_path = (
            os.path.join(cache_dir, f"{readable_stem}_hypotheses.cache")
            if readable_stem
            else None
        )
        legacy_path = os.path.join(cache_dir, f"{dataset._fingerprint}_hypotheses.cache")

        if preferred_path and os.path.exists(preferred_path):
            cache_path = preferred_path
        elif os.path.exists(legacy_path):
            cache_path = legacy_path
            if preferred_path:
                logger.info(
                    "Loading legacy hypotheses cache %s (readable name %s not found)",
                    legacy_path,
                    preferred_path,
                )
        else:
            cache_path = preferred_path if preferred_path else legacy_path

        if not os.path.exists(cache_path):
            print(
                f"\t[{dataset.builder_name}] Hypotheses cache miss: {cache_path} "
                f"(fingerprint={dataset._fingerprint}; readable_stem={readable_stem or 'n/a'})"
            )
            print(f"\t[{dataset.builder_name}] Saving hypotheses to path {cache_path}")

            # num_proc=None: precompute uses GPU (inversion model); forked workers cannot use CUDA.
            dataset = dataset_map_multi_worker(
                dataset=dataset,
                map_fn=functools.partial(
                    self._precompute_hypothesis_and_embedding,
                    collator=self.data_collator,
                ),
                batched=True,
                batch_size=(self.args.train_batch_size * 2),
                desc="Precomputing hypotheses for data",
                num_proc=None,
            )

            if filter_correct_examples:
                old_length = len(dataset)

                def embedding_is_not_correct(ex):
                    return (
                        ~torch.isclose(
                            ex["frozen_embeddings"].to(self.args.device),
                            ex["hypothesis_embedding"].to(self.args.device),
                        ).all(dim=1)
                    ).tolist()

                dataset = dataset.filter(
                    embedding_is_not_correct,
                    batched=True,
                    batch_size=1024,
                )
                print(f"filtered {old_length} datapoints to {len(dataset)}")
            dataset.save_to_disk(cache_path)
        else:
            logging.info("Loading hypotheses from path %s", cache_path)
            print(
                f"\t[{dataset.builder_name}] Loading hypotheses from path {cache_path}"
            )
            dataset = datasets.load_from_disk(cache_path)
        dataset.set_format("pt")
        return dataset, cache_path

    def precompute_hypotheses(self) -> None:
        """Generates and embeds hypotheses using `self.inversion_trainer`.

        Returns path to precomputed-and-saved train dataset, which is sometimes
        useful for outside processes.
        """
        logger.info("Precomputing frozen embedding & hypotheses before training")

        self.train_dataset, train_cache_path = self._preprocess_dataset_hypotheses(
            dataset=self.train_dataset,
            filter_correct_examples=True,
            split_suffix="train",
        )
        for k, v in self.eval_dataset.items():
            eval_key_stem = "_".join([x.strip() for x in str(k).split(",") if x.strip()]) or str(
                k
            ).replace(",", "_")
            split_suffix = f"val_{_sanitize_cache_component(eval_key_stem)}"
            self.eval_dataset[k], _ = self._preprocess_dataset_hypotheses(
                dataset=v,
                filter_correct_examples=False,
                split_suffix=split_suffix,
            )

    def _inner_training_loop(self, *args, **kwargs):

        print(
            "[Corrector] training start – args: per_device_train_batch_size=%s, "
            "gradient_accumulation_steps=%s, train_batch_size(_train_batch_size)=%s"
            % (
                getattr(self.args, "per_device_train_batch_size", "?"),
                getattr(self.args, "gradient_accumulation_steps", "?"),
                getattr(self, "_train_batch_size", "? (not set yet)"),
            )
        )
        
        self.model.eval()
        self.model.to(self.args.device)
        if self.dae is not None:
            self.dae.to(self.args.device)
        self.inversion_trainer.model.to(next(self.model.parameters()).device)
        self.precompute_hypotheses()
        self.model.train()
        if self.dae is not None:
            self.dae.eval()
        self.inversion_trainer.model.cpu()
        # torch.cuda.empty_cache()
        # logger.info(
        #     "After precompute: train_dataset=%d samples, GPU alloc=%.1f MiB, reserved=%.1f MiB",
        #     len(self.train_dataset),
        #     torch.cuda.memory_allocated() / 2**20,
        #     torch.cuda.memory_reserved() / 2**20,
        # )
        return super()._inner_training_loop(*args, **kwargs)

    def generate(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
    ) -> torch.Tensor:
        """Generates text using self-correction.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["frozen_embeddings"] = frozen_embeddings
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_embedding"] = hypothesis_embedding
        # print("generating with sequence_beam_width:", (sequence_beam_width or self.sequence_beam_width))

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        if self.dae is not None:
            num_recursive_steps = min(num_recursive_steps, self.max_correction_steps)
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        target_batch_size = frozen_embeddings.shape[0]
        total_best_scores_seen = None  # Track best scores for early stopping
        prev_cos_sim = None

        while num_recursive_steps >= 1:
            gen_text_ids, hypothesis_embedding, best_scores = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
            )
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_embedding"] = hypothesis_embedding
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1

            # Noise-aware early stopping (DAE) or vanilla early stopping
            if self.dae is not None:
                with torch.no_grad():
                    _fe = inputs["frozen_embeddings"]
                    _fe = align_targets_to_hypotheses(_fe, hypothesis_embedding)
                    denoised_target = self.dae(_fe, sigma=self._dae_sigma(_fe))
                    denoised_hyp = self.dae(hypothesis_embedding, sigma=self._dae_sigma(hypothesis_embedding))
                    cos_sim = F.cosine_similarity(
                        denoised_target, denoised_hyp, dim=-1
                    ).mean()
                if prev_cos_sim is not None:
                    cos_sim_change = (cos_sim - prev_cos_sim).item()
                    if cos_sim_change < self.early_stop_threshold:
                        msg = (
                            "[DAE early stop] "
                            f"round={num_recursive_steps_so_far} "
                            f"cos_sim_change={cos_sim_change:.6f} "
                            f"< threshold={self.early_stop_threshold:.6f} "
                            f"(prev={prev_cos_sim.item():.6f}, current={cos_sim.item():.6f})"
                        )
                        logger.info(
                            msg,
                        )
                        print(msg)
                        gen_text_ids, hypothesis_embedding = collapse_beam_outputs_to_batch(
                            gen_text_ids,
                            hypothesis_embedding,
                            target_batch_size=target_batch_size,
                        )
                        break
                prev_cos_sim = cos_sim
            else:
                if best_scores is not None:
                    if (
                        total_best_scores_seen is not None
                        and _best_scores_are_unchanged(best_scores, total_best_scores_seen)
                    ):
                        break
                    total_best_scores_seen = best_scores.detach().clone()

        return gen_text_ids

    def generate_with_hypotheses(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int = None,
        sequence_beam_width: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates text using self-correction. Works exactly like generate(), but returns all the intermediate hypotheses steps.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (List[torch.Tensor]): ids of generated text, for each hypothesis sequence
            hypothesis_embeddings (List[torch.Tensor]): embeddings of each hypothesis sequence
        """
        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        # Add beam dimension:
        #       (batch, ...) -> (batch, beam, ...)
        inputs["frozen_embeddings"] = frozen_embeddings
        inputs["hypothesis_input_ids"] = hypothesis_input_ids
        inputs["hypothesis_attention_mask"] = hypothesis_attention_mask
        inputs["hypothesis_embedding"] = hypothesis_embedding

        num_recursive_steps = num_recursive_steps or self.num_gen_recursive_steps
        if self.dae is not None:
            num_recursive_steps = min(num_recursive_steps, self.max_correction_steps)
        sequence_beam_width = sequence_beam_width or self.sequence_beam_width
        num_recursive_steps_so_far = 0

        total_best_scores_seen = None  # Track best scores for early stopping
        prev_cos_sim = None

        ground_truth_embedding = inputs["hypothesis_embedding"]
        hypothesis_embeddings = [ground_truth_embedding]  # Track hypothesis embeddings

        hypothesis_ids = [inputs["hypothesis_input_ids"]]  # Track hypothesis ids

        while num_recursive_steps >= 1:
            gen_text_ids, hypothesis_embedding, best_scores = self._generate_with_beam(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                num_recursive_steps=num_recursive_steps,
                num_recursive_steps_so_far=num_recursive_steps_so_far,
                sequence_beam_width=sequence_beam_width,
            )
            inputs["hypothesis_input_ids"] = gen_text_ids
            inputs["hypothesis_attention_mask"] = (
                gen_text_ids != self.model.encoder_decoder.config.pad_token_id
            ).int()
            inputs["hypothesis_embedding"] = hypothesis_embedding
            # step counters
            num_recursive_steps -= 1
            num_recursive_steps_so_far += 1

            # Noise-aware early stopping (DAE) or vanilla early stopping
            if self.dae is not None:
                with torch.no_grad():
                    _fe = inputs["frozen_embeddings"]
                    _fe = align_targets_to_hypotheses(_fe, hypothesis_embedding)
                    denoised_target = self.dae(_fe, sigma=self._dae_sigma(_fe))
                    denoised_hyp = self.dae(hypothesis_embedding, sigma=self._dae_sigma(hypothesis_embedding))
                    cos_sim = F.cosine_similarity(
                        denoised_target, denoised_hyp, dim=-1
                    ).mean()
                if prev_cos_sim is not None:
                    cos_sim_change = (cos_sim - prev_cos_sim).item()
                    if cos_sim_change < self.early_stop_threshold:
                        msg = (
                            "[DAE early stop hypotheses] "
                            f"round={num_recursive_steps_so_far} "
                            f"cos_sim_change={cos_sim_change:.6f} "
                            f"< threshold={self.early_stop_threshold:.6f} "
                            f"(prev={prev_cos_sim.item():.6f}, current={cos_sim.item():.6f})"
                        )
                        logger.info(
                            msg,
                        )
                        print(msg)
                        break
                prev_cos_sim = cos_sim
                closest_idx = 0
            else:
                if best_scores is not None:
                    closest_idx = torch.argmax(best_scores)
                    if (
                        total_best_scores_seen is not None
                        and _best_scores_are_unchanged(best_scores, total_best_scores_seen)
                    ):
                        break
                    total_best_scores_seen = best_scores.detach().clone()
                else:
                    closest_idx = 0

            hypothesis_embeddings.append(hypothesis_embedding[closest_idx].unsqueeze(0))
            hypothesis_ids.append(gen_text_ids[closest_idx].unsqueeze(0))

        return hypothesis_ids, hypothesis_embeddings

    def _generate_with_beam(
        self,
        inputs: Dict,
        generation_kwargs: Dict,
        num_recursive_steps: int,
        num_recursive_steps_so_far: int,
        sequence_beam_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates text using self-correction.

        Args:
            inputs (Dict[str, torch.Tensor]): inputs for generation, like the input embedding, hypothesis,
                and hypothesis embedding
            generation_kwargs (Dict): dictionary of parameters for generation, will be passed on to the model
            num_recursive_steps (int): Number of remaining steps of recursion, used to know when to stop
            num_recusive_steps_so_far (int): Number of steps of recursion performed so far. This is how we
                can check if it's the initial hypothesis or not.
            sequence_beam_width (int): beam width for sequence-level beam search
        Returns:
            generated_ids (torch.Tensor): ids of generated text
        """
        assert num_recursive_steps >= 1
        frozen_embeddings = inputs["frozen_embeddings"]
        ################################################################################
        if not generation_kwargs["do_sample"]:
            num_return_sequences = max(
                sequence_beam_width, generation_kwargs.get("num_beams", 1)
            )
            generation_kwargs["num_beams"] = num_return_sequences
            generation_kwargs["num_return_sequences"] = num_return_sequences

        if (num_recursive_steps_so_far == 0) and (
            self.initial_hypothesis_str is not None
        ):
            # Support setting a string as the initial hypothesis (for ablations)
            logger.info(f"Using initial hypothesis: {self.initial_hypothesis_str}")
            # If set, uses this string as the hypothesis for step 0 of self-correction
            batch_size = frozen_embeddings.shape[0]
            gen_text_ids = (
                self.embedder_tokenizer(
                    [self.initial_hypothesis_str],
                    return_tensors="pt",
                    max_length=inputs["hypothesis_input_ids"].shape[1],
                    truncation=True,
                    padding="max_length",
                )["input_ids"]
                .repeat((batch_size, 1))
                .to(self.args.device)
            )
            # gen_text_ids = (
            #     torch.randint(
            #         low=1,
            #         high=self.embedder_tokenizer.vocab_size,
            #         size=(1, inputs["hypothesis_input_ids"].shape[1]),
            #         dtype=torch.long,
            #     )
            #     .repeat((batch_size, 1))
            #     .to(self.args.device)
            # )
            bos_token_id = self.model.encoder_decoder.config.decoder_start_token_id
            bos_token_ids = (
                torch.ones(
                    (batch_size, 1), dtype=torch.long, device=gen_text_ids.device
                )
                * bos_token_id
            )
            gen_text_ids = torch.cat((bos_token_ids, gen_text_ids[:, :-1]), dim=1)
        else:
            outputs = self.model.generate(
                inputs=inputs,
                generation_kwargs=generation_kwargs,
                return_dict_in_generate=True,
            )
            gen_text_ids = outputs.sequences

            # get scores for sequences to compute sequence-level likelihood.
            # https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
            if "beam_indices" in outputs:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences,
                            outputs.scores,
                            outputs.beam_indices,
                            normalize_logits=True,
                        )
                    )
            else:
                with torch.no_grad():
                    transition_scores = (
                        self.model.encoder_decoder.compute_transition_scores(
                            outputs.sequences, outputs.scores, normalize_logits=True
                        )
                    )
            length_penalty = self.model.encoder_decoder.generation_config.length_penalty
            output_length = (transition_scores < 0).sum(1)
            del outputs.scores
            gen_text_scores = transition_scores.sum(axis=1) / (
                output_length**length_penalty
            )  # log probs

        # Re-embed generated text so we can rerank, and track the best we've seen so far.
        hypothesis_embedding = self.embed_generated_hypothesis(input_ids=gen_text_ids)

        if num_recursive_steps_so_far == 0:
            batch_size = frozen_embeddings.shape[0]
        else:
            # after the first step, we've already copied frozen embeddings across the beam
            batch_size = int(frozen_embeddings.shape[0] / sequence_beam_width)

        best_scores = None
        #
        #   BEAM SEARCH
        #
        if gen_text_ids.shape[0] > batch_size:
            if sequence_beam_width == 1:
                # This is "regular" beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                distances_per_beam = self._beam_distances_to_targets(
                    hypothesis_embedding=hypothesis_embedding,
                    target_embeddings=inputs["frozen_embeddings"],
                    batch_size=batch_size,
                    beam_width=beam_width,
                )
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                best_idx_in_beam = scores.argmax(1)
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam
                ]
                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
            elif num_recursive_steps == 1:
                # Base case for sequence-level beam search.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                frozen_embeddings_per_beam = (
                    inputs["frozen_embeddings"][:, None, :]
                    .repeat((1, num_return_sequences, 1))
                    .reshape((batch_size, beam_width, -1))
                )
                distances_per_beam = self._beam_distances_to_targets(
                    hypothesis_embedding=hypothesis_embedding,
                    target_embeddings=frozen_embeddings_per_beam.reshape(
                        batch_size * beam_width, -1
                    ),
                    batch_size=batch_size,
                    beam_width=beam_width,
                )
                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))
                best_idx_in_beam = scores.argmax(dim=1)
                # print("best_idx_in_beam:", best_idx_in_beam)
                # print("avg_distances:", distances_per_beam.mean(1).tolist(), "max_distances:", distances_per_beam.max(1).values.tolist())
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size), best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size), best_idx_in_beam
                ]
            else:
                # Now get top things in the beam like normal.
                beam_width = int(gen_text_ids.shape[0] / batch_size)
                assert (
                    beam_width % sequence_beam_width == 0
                ), "inner beam width must divide sequence beam width"

                if num_recursive_steps_so_far == 0:
                    # This is the first return for sequence-level beam search.
                    # First we have to copy the frozen embedding
                    frozen_embeddings_per_beam = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape((batch_size, num_return_sequences, -1))
                    )
                    inputs["frozen_embeddings"] = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, sequence_beam_width, 1))
                        .reshape((batch_size * sequence_beam_width, -1))
                    )
                else:
                    frozen_embeddings_per_beam = (
                        inputs["frozen_embeddings"][:, None, :]
                        .repeat((1, num_return_sequences, 1))
                        .reshape(
                            (batch_size, sequence_beam_width * num_return_sequences, -1)
                        )
                    )

                distances_per_beam = self._beam_distances_to_targets(
                    hypothesis_embedding=hypothesis_embedding,
                    target_embeddings=frozen_embeddings_per_beam.reshape(
                        batch_size * beam_width, -1
                    ),
                    batch_size=batch_size,
                    beam_width=beam_width,
                )

                if self.return_best_hypothesis:
                    scores = distances_per_beam
                else:
                    scores = gen_text_scores.reshape((batch_size, beam_width))

                # print("scores:")
                # for t, s in zip(self.tokenizer.batch_decode(gen_text_ids, skip_special_tokens=True), scores.flatten().tolist()):
                #     print(f"\t- {s:2f}", t)
                # print()

                # take top *unique* things in beam.
                best_idx_in_beam_total = scores.topk(dim=1, k=beam_width).indices
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))
                best_idx_in_beam = []
                for batch_idx in range(len(best_idx_in_beam_total)):
                    gen_text_set = set()  # track uniqueness
                    best_idx_in_beam.append([])
                    for j in best_idx_in_beam_total[batch_idx].tolist():
                        gen_text_i = tuple(gen_text_ids[batch_idx, j].tolist())
                        if gen_text_i not in gen_text_set:
                            gen_text_set.add(gen_text_i)
                            best_idx_in_beam[batch_idx].append(j)
                        if len(best_idx_in_beam[batch_idx]) == sequence_beam_width:
                            break
                best_idx_in_beam = torch.tensor(
                    best_idx_in_beam, device=best_idx_in_beam_total.device
                )
                # now take top unique things
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size, beam_width, -1)
                )[torch.arange(batch_size)[:, None], best_idx_in_beam]
                gen_text_ids = gen_text_ids.reshape((batch_size, beam_width, -1))[
                    torch.arange(batch_size)[:, None], best_idx_in_beam
                ]

                # Flatten again so we can do normal operations.
                gen_text_ids = gen_text_ids.reshape(
                    (batch_size * sequence_beam_width, -1)
                )
                hypothesis_embedding = hypothesis_embedding.reshape(
                    (batch_size * sequence_beam_width, -1)
                )

            # print scores for any type of beam search
            best_scores = scores.max(1).values.cpu()
        # make sure we reshape correctly
        # (can't do a shape check on gen_text_ids because of the dynamic length.)
        assert hypothesis_embedding.shape[-1] == inputs["frozen_embeddings"].shape[-1]

        return gen_text_ids, hypothesis_embedding, best_scores

    def get_frozen_embeddings(
        self,
        embedder_input_ids: torch.Tensor,
        embedder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            frozen_embeddings = self.inversion_trainer.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )

        return frozen_embeddings.to(self.args.device)

    def embed_generated_hypothesis(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds a generated hypothesis. Has to remove EOS token and add BOS token
        at the beginning.
        """
        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=self.model.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(input_ids.device)
        embedding = self.get_frozen_embeddings(
            embedder_input_ids=emb_input_ids.input_ids,
            embedder_attention_mask=emb_input_ids.attention_mask,
        )
        return embedding

    def _get_hypothesis_uncached(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "frozen_embeddings" in inputs:
            frozen_embeddings = inputs["frozen_embeddings"]
        elif "embedder_input_ids" in inputs:
            frozen_embeddings = self.get_frozen_embeddings(
                embedder_input_ids=inputs["embedder_input_ids"],
                embedder_attention_mask=inputs["embedder_attention_mask"],
            )
        else:
            assert (
                "input_ids" in inputs
            ), f"cannot generate hypothesis with input keys: {inputs.keys()}"
            frozen_embeddings = self.embed_generated_hypothesis(
                input_ids=inputs["input_ids"]
            )

        inverter_input_embeddings = frozen_embeddings
        if self.dae is not None:
            with torch.no_grad():
                self.dae.eval()
                inverter_input_embeddings = self.dae(
                    frozen_embeddings, sigma=self._dae_sigma(frozen_embeddings)
                )
                if not self._dae_hypothesis_call_verified:
                    self._dae_hypothesis_call_verified = True
                    l2 = (inverter_input_embeddings - frozen_embeddings).norm(dim=1).mean().item()
                    print(
                        "[DAE Shield hypothesis] initial hypothesis uses DAE-denoised "
                        f"embeddings | L2(denoised-noisy)={l2:.6f}"
                    )

        generation_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
            "max_length": self.model.config.max_seq_length,
        }

        hypothesis_input_ids = self.inversion_trainer.model.generate(
            inputs={
                "frozen_embeddings": inverter_input_embeddings,
            },
            generation_kwargs=generation_kwargs,
        )
        hypothesis_attention_mask = (
            hypothesis_input_ids != self.model.encoder_decoder.config.pad_token_id
        )
        hypothesis_embedding = self.embed_generated_hypothesis(
            input_ids=hypothesis_input_ids
        )
        return (
            frozen_embeddings,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            hypothesis_embedding,
        )

    def compute_loss(
        self,
        model: CorrectorEncoderModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        batch_size, seq_length = inputs["input_ids"].shape

        try:
            frozen_embeddings = inputs["frozen_embeddings"]
            hypothesis_input_ids = inputs["hypothesis_input_ids"]
            hypothesis_attention_mask = inputs["hypothesis_attention_mask"]
            hypothesis_embedding = inputs["hypothesis_embedding"]
        except KeyError:
            (
                frozen_embeddings,
                hypothesis_input_ids,
                hypothesis_attention_mask,
                hypothesis_embedding,
            ) = self._get_hypothesis_uncached(inputs=inputs)

        labels = inputs["labels"]
        outputs = self.model(
            embedding=frozen_embeddings,
            hypothesis_embedding=hypothesis_embedding,
            hypothesis_input_ids=hypothesis_input_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            labels=labels,
        )
        return outputs.loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`. Called during self.evalaute()
        """
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        with torch.no_grad():
            loss = self.compute_loss(model=model, inputs=inputs)

        logits, labels = None, None
        return loss, logits, labels

    def _remap_state_dict(self, state_dict: Dict) -> Dict:
        """Edit keys posthumously on model load."""
        # Rename keys for backward compatibility w/ model trained before
        # we stopped sharing params between the ff layers
        if {
            "embedding_transform.3.weight",
            "embedding_transform.3.bias",
        } <= state_dict.keys():
            print(
                "Renaming keys",
                {"embedding_transform.2.weight", "embedding_transform.2.bias"},
                "for backward compatibility.",
            )
            state_dict["embedding_transform_1.0.weight"] = state_dict.pop(
                "embedding_transform.0.weight"
            )
            state_dict["embedding_transform_1.0.bias"] = state_dict.pop(
                "embedding_transform.0.bias"
            )
            state_dict["embedding_transform_1.3.weight"] = state_dict.pop(
                "embedding_transform.3.weight"
            )
            state_dict["embedding_transform_1.3.bias"] = state_dict.pop(
                "embedding_transform.3.bias"
            )
            #
            state_dict["embedding_transform_2.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_2.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_2.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_2.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
            #
            state_dict["embedding_transform_3.0.weight"] = state_dict[
                "embedding_transform_1.0.weight"
            ]
            state_dict["embedding_transform_3.0.bias"] = state_dict[
                "embedding_transform_1.0.bias"
            ]
            state_dict["embedding_transform_3.3.weight"] = state_dict[
                "embedding_transform_1.3.weight"
            ]
            state_dict["embedding_transform_3.3.bias"] = state_dict[
                "embedding_transform_1.3.bias"
            ]
        return state_dict

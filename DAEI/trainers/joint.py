"""JointTrainer: multi-stage DAE + v2t joint training.

Stage 0 — warm-up v2t on noisy embeddings (identical to InversionTrainer).
Stage 1 — freeze v2t, train DAE with SURE or N2N loss (+optional contrastive).
Stage 2 — unfreeze both, L_total = L_DAE + λ · L_CE with λ warmup.

HF-native approach: DAE is registered as self.model.dae so that HF Trainer
automatically handles backward (with AMP), gradient accumulation, DDP sync,
and checkpoint save/load. We override compute_loss (not training_step) and
create_optimizer (for dual learning rates).
"""

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers

from DAEI.losses.contrastive import info_nce_loss
from DAEI.losses.n2n import n2n_loss
from DAEI.losses.sure import mc_sure_loss
from DAEI.trainers.base import (
    filter_eval_metrics_for_log,
    metric_for_best_model_candidate_keys,
)
from DAEI.trainers.inversion import InversionTrainer

logger = logging.getLogger(__name__)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DistributedDataParallel to access inner module (e.g. model.dae)."""
    return model.module if hasattr(model, "module") else model


def _scalar_loss(t: torch.Tensor) -> torch.Tensor:
    """Reduce loss to a 0-dim tensor (DataParallel can return one loss per replica)."""
    if not isinstance(t, torch.Tensor):
        return t
    return t.mean() if t.dim() > 0 else t


def _loss_log_value(t: torch.Tensor) -> float:
    """Safe float for logging (handles multi-element tensors from DataParallel)."""
    return float(_scalar_loss(t).detach())


class LambdaScheduler:
    """Linear warm-up from 0 to *target* over *warmup_steps*, then constant."""

    def __init__(self, target: float, warmup_steps: int) -> None:
        self.target = target
        self.warmup_steps = max(warmup_steps, 1)

    def get(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.target
        return self.target * (step / self.warmup_steps)


class SigmaScheduler:
    """Log-uniform noise schedule with random per-batch sampling.

    The *upper bound* decays from sigma_max → sigma_min over *total_steps*.
    Each training batch **randomly samples** σ ~ LogUniform[sigma_min, upper(t)]
    so the DAE (and its SigmaConditioner) see the full range of noise levels
    from the very first step.  This prevents the train-eval mismatch that occurs
    when the conditioner is only ever trained at one σ value.

    *total_steps* is the **span** of the upper-bound decay (e.g. first 20% of
    the run when ``dae_sigma_decay_span_fraction=0.2``).
    """

    def __init__(
        self, sigma_max: float, sigma_min: float, total_steps: int, mode: str = "fixed"
    ) -> None:
        self.sigma_max = sigma_max
        self.sigma_min = max(sigma_min, 1e-8)
        self.total_steps = max(total_steps, 1)
        self.mode = mode

    def get_upper(self, step: int) -> float:
        """Deterministic upper-bound σ at *step* (for logging / monitoring)."""
        if self.mode == "fixed":
            return self.sigma_min
        frac = min(step / self.total_steps, 1.0)
        log_sigma = math.log(self.sigma_max) + frac * (
            math.log(self.sigma_min) - math.log(self.sigma_max)
        )
        return math.exp(log_sigma)

    def sample(self, step: int) -> float:
        """Sample σ ~ LogUniform[sigma_min, upper(step)] for one training batch."""
        upper = self.get_upper(step)
        if self.mode == "fixed" or upper <= self.sigma_min:
            return self.sigma_min
        log_lo = math.log(self.sigma_min)
        log_hi = math.log(upper)
        return math.exp(log_lo + (log_hi - log_lo) * torch.rand(1).item())


def _filter_stage1_eval_metrics(
    metrics: Dict,
    metric_key_prefix: str,
    metric_for_best_model: Optional[str] = None,
) -> Dict:
    """Keep only DAE / SURE / throughput keys for Stage 1 eval (no text metrics)."""
    suffixes = (
        "_loss",
        "_perplexity",
        "_runtime",
        "_samples_per_second",
        "_steps_per_second",
    )
    out: Dict = {}
    for k, v in metrics.items():
        if k == "epoch":
            out[k] = v
            continue
        if "sure_" in k:
            out[k] = v
            continue
        if k.startswith(metric_key_prefix) and any(k.endswith(s) for s in suffixes):
            out[k] = v
            continue
    for cand in metric_for_best_model_candidate_keys(metric_for_best_model):
        if cand in metrics:
            out[cand] = metrics[cand]
    return out


def _pcgrad_project_dae(
    dae: nn.Module, sure_grads: Dict[str, torch.Tensor]
) -> None:
    """PCGrad: merge SURE and CE gradients on DAE, projecting CE when conflicting.

    After Phase 2 backward, each DAE param has CE-only gradients. For each param:
      g_sure = saved SURE gradient (from Phase 1)
      g_ce   = current .grad (from Phase 2 CE backward)
    If dot(g_sure, g_ce) < 0 (conflict), project g_ce onto the normal plane of g_sure:
      g_ce' = g_ce - (g_ce · g_sure / ||g_sure||²) * g_sure
    Final grad = g_sure + g_ce'   (SURE is always preserved in full)

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
    """
    n_conflict = 0
    for name, p in dae.named_parameters():
        g_sure = sure_grads.get(name)
        if g_sure is None or p.grad is None:
            if g_sure is not None and p.grad is None:
                p.grad = g_sure
            continue
        g_ce = p.grad
        dot = (g_sure * g_ce).sum()
        if dot < 0:
            norm_sq = (g_sure * g_sure).sum().clamp(min=1e-12)
            g_ce = g_ce - (dot / norm_sq) * g_sure
            n_conflict += 1
        p.grad = g_sure + g_ce
    if n_conflict > 0:
        logger.debug("PCGrad: projected %d/%d DAE params", n_conflict, len(sure_grads))


def _dae_grad_params(dae: nn.Module) -> List[nn.Parameter]:
    """DAE parameters that currently have gradients."""
    return [
        p for p in dae.parameters()
        if p.requires_grad and p.grad is not None
    ]


class JointTrainer(InversionTrainer):
    """Trainer for the three-stage DAE + inverter joint pipeline.

    Key design: DAE is mounted as ``self.model.dae`` so HF Trainer sees it as
    part of the model.  We only override ``compute_loss`` (not ``training_step``),
    leaving backward/AMP/gradient-accumulation/DDP to HF.
    """

    def __init__(
        self,
        *args,
        dae: Optional[nn.Module] = None,
        training_stage: int = 0,
        loss_mode: str = "n2n",
        noise_sigma: float = 0.01,
        sure_n_probes: int = 5,
        lambda_ce: float = 1.0,
        lambda_warmup_steps: int = 5000,
        dae_lr: float = 1e-3,
        # DAEI pipeline enhancements
        dae_contrastive_weight: float = 0.0,
        dae_contrastive_tau: float = 0.07,
        dae_sigma_schedule: str = "fixed",
        dae_sigma_max: float = 0.05,
        dae_sigma_min: float = 0.01,
        dae_sigma_decay_span_fraction: float = 0.2,
        dae_grad_max_norm: float = 0.0,
        pcgrad: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.training_stage = training_stage
        self.loss_mode = loss_mode
        self.noise_sigma = noise_sigma
        self.sure_n_probes = sure_n_probes
        self.dae_lr = dae_lr

        # Contrastive loss settings
        self.dae_contrastive_weight = dae_contrastive_weight
        self.dae_contrastive_tau = dae_contrastive_tau
        self.dae_grad_max_norm = dae_grad_max_norm
        self.pcgrad = pcgrad

        self.lambda_scheduler = LambdaScheduler(lambda_ce, lambda_warmup_steps)

        # ---- Estimate total optimizer steps from actual data, not defaults ----
        _ms = getattr(self.args, "max_steps", -1)
        if _ms is not None and _ms > 0:
            full_training_steps = int(_ms)
        else:
            # Compute actual steps/epoch from dataset size & batch config.
            # self.train_dataset is set by super().__init__().
            train_ds = getattr(self, "train_dataset", None)
            if train_ds is not None and hasattr(train_ds, "__len__") and len(train_ds) > 0:
                world_size = max(int(getattr(self.args, "world_size", 1)), 1)
                ga = max(int(getattr(self.args, "gradient_accumulation_steps", 1)), 1)
                bs = int(self.args.per_device_train_batch_size)
                actual_steps_per_epoch = max(len(train_ds) // (bs * world_size * ga), 1)
            else:
                actual_steps_per_epoch = int(getattr(self.args, "steps_per_epoch", 500_000))
            full_training_steps = int(
                float(getattr(self.args, "num_train_epochs", 30)) * actual_steps_per_epoch
            )

        _span = getattr(self.args, "dae_sigma_decay_span_fraction", dae_sigma_decay_span_fraction)
        span_frac = max(0.0, min(1.0, float(_span)))
        if dae_sigma_schedule == "log_uniform":
            sigma_decay_steps = max(int(full_training_steps * span_frac), 1)
        else:
            sigma_decay_steps = max(full_training_steps, 1)

        self.sigma_scheduler = SigmaScheduler(
            sigma_max=dae_sigma_max,
            sigma_min=dae_sigma_min,
            total_steps=sigma_decay_steps,
            mode=dae_sigma_schedule,
        )
        logger.info(
            "Sigma schedule: mode=%s, σ_max=%.4f → σ_min=%.4f, "
            "full_steps=%d, decay_fraction=%.2f → decay_steps=%d  "
            "(per-batch: σ ~ LogUniform[σ_min, upper(t)])",
            dae_sigma_schedule,
            dae_sigma_max,
            dae_sigma_min,
            full_training_steps,
            span_frac,
            self.sigma_scheduler.total_steps,
        )

        if dae is not None:
            self.model.dae = dae.to(self.args.device)

        if self.training_stage == 1:
            self._freeze_v2t()
        elif self.training_stage in (2, 3):
            self._unfreeze_v2t()

        if dae_contrastive_weight > 0:
            logger.warning(
                "dae_contrastive_weight=%.3f but contrastive loss requires "
                "clean_embeddings in each batch.  If the dataset does not "
                "contain clean_embeddings (the normal 'no-clean-emb' setting), "
                "contrastive loss will be silently skipped every step.  "
                "Set --dae_contrastive_weight 0 to suppress this warning.",
                dae_contrastive_weight,
            )

        logger.info(
            "JointTrainer init: stage=%d, loss_mode=%s, sigma=%.4f, "
            "sigma_schedule=%s (%.4f→%.4f), contrastive_weight=%.3f, "
            "pcgrad=%s, dae=%s",
            training_stage, loss_mode, noise_sigma,
            dae_sigma_schedule, dae_sigma_max, dae_sigma_min,
            dae_contrastive_weight, pcgrad,
            "yes" if dae is not None else "no",
        )

    # ------------------------------------------------------------------
    # PCGrad: project conflicting CE gradients on DAE params
    # ------------------------------------------------------------------
    def training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self._pcgrad_sure_snapshot = None
        loss = super().training_step(model, inputs)
        inner = _unwrap_model(model)
        if self._pcgrad_sure_snapshot is not None:
            _pcgrad_project_dae(inner.dae, self._pcgrad_sure_snapshot)
            self._pcgrad_sure_snapshot = None
        self._clip_dae_gradients(inner)
        return loss

    def _clip_dae_gradients(self, inner: nn.Module) -> None:
        """Clip only DAE gradients after SURE/CE/PCGrad have been merged."""
        if self.dae_grad_max_norm <= 0:
            return
        if not getattr(self.accelerator, "sync_gradients", True):
            return
        dae = getattr(inner, "dae", None)
        if dae is None:
            return
        params = _dae_grad_params(dae)
        if not params:
            return
        self.accelerator.clip_grad_norm_(params, self.dae_grad_max_norm)

    def log(self, logs: Dict, *args, **kwargs) -> None:
        """Stage 1: name training loss as ``train/dae_loss``; HF Trainer logs ``loss``."""
        if getattr(self, "training_stage", None) == 1 and logs and self.model.training:
            if "loss" in logs and "train/dae_loss" not in logs:
                logs = dict(logs)
                logs["train/dae_loss"] = logs.pop("loss")
        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------
    def _freeze_v2t(self) -> None:
        """Freeze all v2t parameters, keep DAE trainable."""
        for n, p in self.model.named_parameters():
            if "dae" not in n:
                p.requires_grad = False
        logger.info("Froze v2t parameters (stage 1). Only DAE is trainable.")

    def _unfreeze_v2t(self) -> None:
        """Unfreeze all parameters except the embedder (which is always frozen)."""
        for p in self.model.parameters():
            p.requires_grad = True
        if getattr(self.model, "embedder_no_grad", False) and hasattr(self.model, "embedder"):
            for p in self.model.embedder.parameters():
                p.requires_grad = False
        logger.info("Unfroze v2t parameters (stage 2).")

    # ------------------------------------------------------------------
    # Dual learning rate optimizer
    # ------------------------------------------------------------------
    def create_optimizer(self):
        """Create a single AdamW with parameter groups: v2t lr vs DAE lr."""
        if self.optimizer is None:
            groups = []

            v2t_params = [
                p for n, p in self.model.named_parameters()
                if "dae" not in n and p.requires_grad
            ]
            if v2t_params:
                groups.append({"params": v2t_params, "lr": self.args.learning_rate})

            inner = _unwrap_model(self.model)
            if hasattr(inner, "dae"):
                dae_params = [p for p in inner.dae.parameters() if p.requires_grad]
                if dae_params:
                    groups.append({"params": dae_params, "lr": self.dae_lr})

            if groups:
                self.optimizer = torch.optim.AdamW(
                    groups, weight_decay=self.args.weight_decay
                )
            else:
                super().create_optimizer()
        return self.optimizer

    # ------------------------------------------------------------------
    # Eval generation: use DAE-denoted embeddings so BLEU/ROUGE reflect DAE
    # ------------------------------------------------------------------
    def _get_decoded_sequences(
        self, dataloader: torch.utils.data.DataLoader, n: int
    ) -> Tuple[List, List]:
        """Like base implementation but feeds DAE(emb) to decoder when DAE exists."""
        assert not self.model.training
        gen_kwargs = copy.copy(self.gen_kwargs)
        all_preds: List = []
        all_labels: List = []
        for step, inputs in enumerate(
            tqdm.tqdm(dataloader, desc="generating from val", leave=False)
        ):
            inputs_cuda = {k: v.to(self.args.device) for k, v in inputs.items()}
            # So that eval BLEU/ROUGE reflect DAE quality: run embeddings through DAE
            inner = _unwrap_model(self.model)
            if hasattr(inner, "dae"):
                if "frozen_embeddings" in inputs_cuda:
                    with torch.no_grad():
                        _fe = inputs_cuda["frozen_embeddings"]
                        inputs_cuda["frozen_embeddings"] = inner.dae(
                            _fe, sigma=self._inference_sigma(_fe)
                        )
                elif "frozen_embeddings_a" in inputs_cuda:
                    with torch.no_grad():
                        _fe = inputs_cuda["frozen_embeddings_a"]
                        inputs_cuda["frozen_embeddings"] = inner.dae(
                            _fe, sigma=self._inference_sigma(_fe)
                        )
            max_length = self.model.config.max_seq_length
            gen_kwargs["max_length"] = max_length
            with torch.no_grad():
                generated_text = self.generate(
                    inputs=inputs_cuda, generation_kwargs=gen_kwargs
                )
            if generated_text.shape[1] < max_length:
                pad_tokens = (
                    torch.ones(
                        (generated_text.shape[0], max_length - generated_text.shape[1]),
                        dtype=torch.long,
                        device=generated_text.device,
                    )
                    * self.pad_token_id
                )
                generated_text = torch.cat((generated_text, pad_tokens), dim=1)
            true_input_ids = inputs["input_ids"]
            if true_input_ids.shape[1] < max_length:
                pad_tokens = (
                    torch.ones(
                        (true_input_ids.shape[0], max_length - true_input_ids.shape[1]),
                        dtype=torch.long,
                        device=true_input_ids.device,
                    )
                    * self.pad_token_id
                )
                true_input_ids = torch.cat((true_input_ids, pad_tokens), dim=1)
            all_preds.extend(generated_text.cpu().tolist())
            all_labels.extend(true_input_ids.cpu().tolist())
            if len(all_preds) >= n:
                break
        return all_preds, all_labels

    # ------------------------------------------------------------------
    # SURE-specific eval metrics in embedding space
    # ------------------------------------------------------------------
    def _compute_sure_eval_metrics(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Compare noisy vs. clean vs. DAE(noisy) embeddings on the val set.

        Requires that the eval dataset provides both ``frozen_embeddings`` (noisy)
        and ``clean_embeddings`` (clean, non-noisy) columns, which we add for
        validation splits when dataset_mode == 'noisy'.
        """
        inner = _unwrap_model(self.model)
        if not hasattr(inner, "dae"):
            return {}

        device = self.args.device
        dae = inner.dae
        was_training = dae.training
        dae.eval()

        total_noisy_mse = 0.0
        total_dae_mse = 0.0
        total_noisy_cos = 0.0
        total_dae_cos = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                if "frozen_embeddings" not in batch or "clean_embeddings" not in batch:
                    continue
                noisy = batch["frozen_embeddings"].to(device)
                clean = batch["clean_embeddings"].to(device)
                denoised = dae(noisy, sigma=self._inference_sigma(noisy))

                bsz = noisy.shape[0]
                noisy_flat = noisy.view(bsz, -1)
                clean_flat = clean.view(bsz, -1)
                denoised_flat = denoised.view(bsz, -1)

                noisy_mse = (noisy_flat - clean_flat).pow(2).sum(dim=-1).mean().item()
                dae_mse = (denoised_flat - clean_flat).pow(2).sum(dim=-1).mean().item()

                noisy_cos = F.cosine_similarity(
                    noisy_flat, clean_flat, dim=-1
                ).mean().item()
                dae_cos = F.cosine_similarity(
                    denoised_flat, clean_flat, dim=-1
                ).mean().item()

                total_noisy_mse += noisy_mse * bsz
                total_dae_mse += dae_mse * bsz
                total_noisy_cos += noisy_cos * bsz
                total_dae_cos += dae_cos * bsz
                count += bsz

        if was_training:
            dae.train()

        if count == 0:
            # Only warn when this eval dataset was supposed to have clean_embeddings (main val).
            # Other val sets (e.g. ag_news) never have it, so we skip them without warning.
            ds = getattr(dataloader, "dataset", None)
            cols = getattr(ds, "column_names", []) if ds is not None else []
            if "clean_embeddings" in cols:
                logger.warning(
                    "SURE embedding metrics skipped: dataset has 'clean_embeddings' but no batch "
                    "had both 'frozen_embeddings' and 'clean_embeddings' (e.g. collator dropped one)."
                )
            return {}

        avg_noisy_mse = total_noisy_mse / count
        avg_dae_mse = total_dae_mse / count
        avg_noisy_cos = total_noisy_cos / count
        avg_dae_cos = total_dae_cos / count

        return {
            "sure_noisy_mse": avg_noisy_mse,
            "sure_dae_mse": avg_dae_mse,
            "sure_delta_mse": avg_noisy_mse - avg_dae_mse,
            "sure_noisy_cos": avg_noisy_cos,
            "sure_dae_cos": avg_dae_cos,
            "sure_delta_cos": avg_dae_cos - avg_noisy_cos,
        }

    # ------------------------------------------------------------------
    # prediction_step: skip v2t logits in DAE-only stages
    # ------------------------------------------------------------------
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override for stage >= 1 so eval doesn't try to extract v2t logits."""
        if self.training_stage == 0:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Stage 3 has v2t unfrozen and produces logits — use default path
        if self.training_stage == 3:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with torch.enable_grad():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss.detach(), None, None)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:  # type: ignore[name-defined]
        """Eval: Stage 1 skips text generation (BaseTrainer.eval_generation_metrics)."""
        if self.training_stage == 1:
            # Call transformers.Trainer only — not BaseTrainer (which runs BLEU / generation).
            output = transformers.Trainer.evaluation_loop(self, *args, **kwargs)
            self._broadcast_metric_for_best_model_if_missing(output.metrics)
            metric_key_prefix = kwargs.get("metric_key_prefix", "eval")
            try:
                perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
            except KeyError:
                perplexity = -1
            except OverflowError:
                perplexity = float("inf")
            output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

            dataloader = kwargs.get("dataloader")
            if dataloader is None and len(args) > 0:
                dataloader = args[0]

            if self.loss_mode == "sure" and dataloader is not None:
                sure_metrics = self._compute_sure_eval_metrics(dataloader)
                sure_metrics = {
                    f"{metric_key_prefix}_{k}": v for k, v in sure_metrics.items()
                }
                output.metrics.update(sure_metrics)

            # Stage 1 is DAE-only, so its useful eval signals are SURE
            # diagnostics rather than text generation metrics.  Keep them even
            # when later stages use compact text-only eval logging.
            mbfm = getattr(self.args, "metric_for_best_model", None)
            filtered = _filter_stage1_eval_metrics(
                output.metrics, metric_key_prefix, mbfm
            )
            output.metrics.clear()
            output.metrics.update(filtered)
            return output

        output = super().evaluation_loop(*args, **kwargs)

        if getattr(self.args, "eval_log_text_metrics_only", False):
            filtered = filter_eval_metrics_for_log(
                output.metrics,
                getattr(self.args, "metric_for_best_model", None),
            )
            output.metrics.clear()
            output.metrics.update(filtered)

        return output

    # ------------------------------------------------------------------
    # compute_loss: the only place we inject custom logic
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.training_stage == 0:
            return super().compute_loss(model, inputs, return_outputs)
        elif self.training_stage == 1:
            return self._compute_loss_stage1(model, inputs, return_outputs)
        elif self.training_stage == 2:
            return self._compute_loss_stage2(model, inputs, return_outputs)
        elif self.training_stage == 3:
            return self._compute_loss_stage3(model, inputs, return_outputs)
        else:
            raise ValueError(f"Unknown training_stage {self.training_stage}")

    def _compute_loss_stage1(self, model, inputs, return_outputs=False):
        """Stage 1: DAE loss + optional contrastive loss (v2t frozen)."""
        if not model.training:
            with torch.enable_grad():
                dae_loss, e_denoised = self._compute_dae_loss(model, inputs, detach_denoised=True)
        else:
            dae_loss, e_denoised = self._compute_dae_loss(model, inputs, detach_denoised=True)

        total_loss = _scalar_loss(dae_loss)
        contrastive_loss_val = 0.0

        # Contrastive regularization: requires clean_embeddings in the batch
        if self.dae_contrastive_weight > 0 and model.training:
            clean = inputs.get("clean_embeddings")
            if clean is not None:
                dae = _unwrap_model(model).dae
                noisy_e = inputs["frozen_embeddings"]
                _ne = noisy_e.detach()
                denoised_for_cl = dae(_ne, sigma=self._inference_sigma(_ne))
                cl_loss = info_nce_loss(denoised_for_cl, clean.detach(), tau=self.dae_contrastive_tau)
                contrastive_loss_val = _loss_log_value(cl_loss)
                total_loss = total_loss + self.dae_contrastive_weight * cl_loss

        if model.training and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            log_dict: Dict = {}
            if self.dae_contrastive_weight > 0:
                log_dict["train/contrastive_loss"] = contrastive_loss_val
            if self.sigma_scheduler.mode != "fixed":
                log_dict["train/sigma_upper"] = self._get_sigma_upper()
            if log_dict:
                self.log(log_dict)

        return total_loss

    def _compute_loss_stage2(self, model, inputs, return_outputs=False):
        """Stage 2: DAE loss + λ · CE loss (both trainable).

        For SURE mode during training we split the backward into two phases
        to keep MC-SURE's heavy second-order computation graph strictly within
        the tiny DAE, preventing it from coupling with T5's 220M-param
        first-order graph.  This yields ~10x speedup.

        Phase 1: SURE forward → backward immediately → second-order graph freed
        Phase 2: fresh DAE forward (first-order) → T5 CE loss → returned to HF

        After HF Trainer's backward on the returned loss:
            DAE.grad  = SURE_grad (phase 1) + λ · CE_grad (phase 2)
            T5.grad   = λ · CE_grad (phase 2)
        The single optimizer (dual-LR param groups) then updates everything.
        """
        lam = self.lambda_scheduler.get(self.state.global_step)

        if model.training and self.loss_mode == "sure":
            inner = _unwrap_model(model)
            noisy_e = inputs["frozen_embeddings"]

            # --- Phase 1: SURE loss (second-order, DAE only) ---
            dae_loss, _ = mc_sure_loss(
                inner.dae, noisy_e, sigma=self.noise_sigma,
                n_probes=self.sure_n_probes, detach_denoised=True,
            )
            dae_loss = _scalar_loss(dae_loss)
            # Backward immediately via accelerator (handles gradient-accumulation
            # scaling).  Only DAE params receive gradients because T5 is not part
            # of SURE's graph.  The second-order graph is freed right after.
            self.accelerator.backward(dae_loss)

            # --- Phase 2: CE loss (pure first-order, clean graph) ---
            _ne = noisy_e.detach()
            e_denoised = inner.dae(_ne, sigma=self._inference_sigma(_ne))
            ce_inputs = self._prepare_ce_inputs(inputs)
            ce_inputs["frozen_embeddings"] = e_denoised
            ce_outputs = model(**ce_inputs)
            ce_loss = ce_outputs["loss"] if isinstance(ce_outputs, dict) else ce_outputs[0]
            ce_loss = _scalar_loss(ce_loss)

            # Return λ·CE for HF Trainer to backward — adds CE grads to the
            # existing SURE grads already sitting on DAE params.
            final_loss = lam * ce_loss

            if self.state.global_step % max(self.args.logging_steps, 1) == 0:
                self.log({
                    "train/dae_loss": _loss_log_value(dae_loss),
                    "train/ce_loss": _loss_log_value(ce_loss),
                    "train/lambda_ce": lam,
                    "train/total_loss": (
                        _loss_log_value(dae_loss) + lam * _loss_log_value(ce_loss)
                    ),
                })

            return (final_loss, ce_outputs) if return_outputs else final_loss

        # Eval mode (any loss_mode) or N2N training: no second-order graph issue
        if not model.training:
            with torch.enable_grad():
                dae_loss, e_denoised = self._compute_dae_loss(
                    model, inputs, detach_denoised=True
                )
        else:
            dae_loss, e_denoised = self._compute_dae_loss(
                model, inputs, detach_denoised=False
            )

        ce_inputs = self._prepare_ce_inputs(inputs)
        ce_inputs["frozen_embeddings"] = e_denoised
        ce_outputs = model(**ce_inputs)
        ce_loss = ce_outputs["loss"] if isinstance(ce_outputs, dict) else ce_outputs[0]
        ce_loss = _scalar_loss(ce_loss)
        dae_loss = _scalar_loss(dae_loss)

        total_loss = dae_loss + lam * ce_loss

        if model.training and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({
                "train/dae_loss": _loss_log_value(dae_loss),
                "train/ce_loss": _loss_log_value(ce_loss),
                "train/lambda_ce": lam,
                "train/total_loss": _loss_log_value(total_loss),
            })

        return (total_loss, ce_outputs) if return_outputs else total_loss

    def _compute_loss_stage3(self, model, inputs, return_outputs=False):
        """Stage 3 (DAEI joint fine-tuning): SURE + lambda * CE.

        Like Stage 2 but designed for the DAEI pipeline where the Inverter
        was already trained on denoised embeddings in Stage 2. The two-phase
        SURE backward is preserved (SURE second-order graph stays within DAE),
        and CE gradients flow to DAE via a fresh first-order forward pass.

        Additionally supports:
        - Contrastive regularization to prevent DAE collapse
        - DAE-specific gradient clipping
        """
        lam = self.lambda_scheduler.get(self.state.global_step)
        sigma = self._get_current_sigma()

        if model.training and self.loss_mode == "sure":
            inner = _unwrap_model(model)
            noisy_e = inputs["frozen_embeddings"]

            # Sigma schedule: add extra noise on top of data noise (no clean emb needed)
            sure_sigma = sigma
            if self.sigma_scheduler.mode != "fixed" and sigma > self.noise_sigma:
                extra_sigma = math.sqrt(sigma ** 2 - self.noise_sigma ** 2)
                noisy_e = noisy_e + extra_sigma * torch.randn_like(noisy_e)
            else:
                sure_sigma = self.noise_sigma

            # --- Phase 1: SURE loss (second-order, DAE only) ---
            dae_loss, _ = mc_sure_loss(
                inner.dae, noisy_e, sigma=sure_sigma,
                n_probes=self.sure_n_probes, detach_denoised=True,
            )
            dae_loss = _scalar_loss(dae_loss)
            self.accelerator.backward(dae_loss)

            # --- Phase 1b: Contrastive loss (first-order, DAE only) ---
            contrastive_loss_val = 0.0
            if self.dae_contrastive_weight > 0:
                clean = inputs.get("clean_embeddings")
                if clean is not None:
                    _ne = noisy_e.detach()
                    denoised_cl = inner.dae(_ne, sigma=self._inference_sigma(_ne))
                    cl_loss = info_nce_loss(denoised_cl, clean.detach(), tau=self.dae_contrastive_tau)
                    cl_loss_scaled = self.dae_contrastive_weight * cl_loss
                    self.accelerator.backward(cl_loss_scaled)
                    contrastive_loss_val = _loss_log_value(cl_loss)

            # --- PCGrad: snapshot SURE grads on DAE params before CE backward ---
            if self.pcgrad:
                self._pcgrad_sure_snapshot = {
                    n: p.grad.clone()
                    for n, p in inner.dae.named_parameters()
                    if p.grad is not None
                }
                for p in inner.dae.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            # --- Phase 2: CE loss (first-order, DAE + Inverter) ---
            _ne = noisy_e.detach()
            e_denoised = inner.dae(_ne, sigma=self._inference_sigma(_ne))
            ce_inputs = self._prepare_ce_inputs(inputs)
            ce_inputs["frozen_embeddings"] = e_denoised
            ce_outputs = model(**ce_inputs)
            ce_loss = ce_outputs["loss"] if isinstance(ce_outputs, dict) else ce_outputs[0]
            ce_loss = _scalar_loss(ce_loss)

            final_loss = lam * ce_loss

            if self.state.global_step % max(self.args.logging_steps, 1) == 0:
                log_dict = {
                    "train/dae_loss": _loss_log_value(dae_loss),
                    "train/ce_loss": _loss_log_value(ce_loss),
                    "train/lambda_ce": lam,
                    "train/total_loss": (
                        _loss_log_value(dae_loss) + lam * _loss_log_value(ce_loss)
                    ),
                    "train/sigma": sigma,
                    "train/sigma_upper": self._get_sigma_upper(),
                }
                if self.dae_contrastive_weight > 0:
                    log_dict["train/contrastive_loss"] = contrastive_loss_val
                self.log(log_dict)

            return (final_loss, ce_outputs) if return_outputs else final_loss

        # Eval mode: make eval_loss comparable to Stage 2 / plain inverter CE.
        # MC-SURE contains a high-variance Jacobian trace term; after joint
        # training it can spike even when generation metrics remain stable, so
        # mixing it into eval_loss makes checkpoint selection misleading.
        if not model.training:
            inner = _unwrap_model(model)
            noisy_e = inputs["frozen_embeddings"]
            e_denoised = inner.dae(noisy_e, sigma=self._inference_sigma(noisy_e))

            ce_inputs = self._prepare_ce_inputs(inputs)
            ce_inputs["frozen_embeddings"] = e_denoised
            ce_outputs = model(**ce_inputs)
            ce_loss = ce_outputs["loss"] if isinstance(ce_outputs, dict) else ce_outputs[0]
            ce_loss = _scalar_loss(ce_loss)
            return (ce_loss, ce_outputs) if return_outputs else ce_loss

        # N2N training or non-SURE paths: single graph, no second-order issue
        else:
            dae_loss, e_denoised = self._compute_dae_loss(
                model, inputs, detach_denoised=False
            )

        ce_inputs = self._prepare_ce_inputs(inputs)
        ce_inputs["frozen_embeddings"] = e_denoised
        ce_outputs = model(**ce_inputs)
        ce_loss = ce_outputs["loss"] if isinstance(ce_outputs, dict) else ce_outputs[0]
        ce_loss = _scalar_loss(ce_loss)
        dae_loss = _scalar_loss(dae_loss)

        total_loss = dae_loss + lam * ce_loss

        if model.training and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({
                "train/dae_loss": _loss_log_value(dae_loss),
                "train/ce_loss": _loss_log_value(ce_loss),
                "train/lambda_ce": lam,
                "train/total_loss": _loss_log_value(total_loss),
            })

        return (total_loss, ce_outputs) if return_outputs else total_loss

    # ------------------------------------------------------------------
    # DAE loss helpers
    # ------------------------------------------------------------------
    def _get_current_sigma(self) -> float:
        """Sample σ for this training batch from LogUniform[σ_min, upper(t)]."""
        return self.sigma_scheduler.sample(self.state.global_step)

    def _get_sigma_upper(self) -> float:
        """Deterministic upper bound of σ at this step (for logging)."""
        return self.sigma_scheduler.get_upper(self.state.global_step)

    def _inference_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """Return σ tensor for inference-time DAE calls (data-level noise)."""
        return torch.full(
            (x.shape[0],), self.noise_sigma, device=x.device, dtype=x.dtype
        )

    def _compute_dae_loss(self, model, inputs, detach_denoised=True):
        dae = _unwrap_model(model).dae
        sigma = self._get_current_sigma()

        if self.loss_mode == "sure":
            noisy_e = inputs["frozen_embeddings"]

            # Progressive noise curriculum: add extra Gaussian noise so the
            # DAE sees higher noise early on.  The data already has N(0,σ_data²)
            # noise; adding independent N(0,σ_extra²) yields total N(0,σ_target²)
            # where σ_extra = √(σ_target² − σ_data²).
            #
            # SURE's σ parameter MUST equal the **actual total noise std** of the
            # input it receives — that is σ_target (= `sigma` from the scheduler),
            # not the raw data noise σ_data.  This keeps SURE an unbiased MSE
            # estimator.  (Prior σ=0.05 divergence was caused by wrong total_steps
            # keeping σ stuck at 0.05 forever, not by using σ_target in the formula.)
            sure_sigma = sigma          # = σ_target when schedule is active
            if self.sigma_scheduler.mode != "fixed" and sigma > self.noise_sigma:
                extra_sigma = math.sqrt(sigma ** 2 - self.noise_sigma ** 2)
                noisy_e = noisy_e + extra_sigma * torch.randn_like(noisy_e)
            else:
                sure_sigma = self.noise_sigma

            loss, e_denoised = mc_sure_loss(
                dae, noisy_e, sigma=sure_sigma,
                n_probes=self.sure_n_probes,
                detach_denoised=detach_denoised,
            )
        elif self.loss_mode == "n2n":
            noisy_a = inputs.get("frozen_embeddings_a")
            noisy_b = inputs.get("frozen_embeddings_b")
            if noisy_a is None or noisy_b is None:
                raise ValueError(
                    "N2N loss requires 'frozen_embeddings_a' and 'frozen_embeddings_b'. "
                    "Use --dataset_mode n2n."
                )
            loss, e_denoised = n2n_loss(
                dae, noisy_a, noisy_b, detach_denoised=detach_denoised,
            )
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
        return loss, e_denoised

    def _prepare_ce_inputs(self, inputs: Dict) -> Dict:
        """Extract kwargs expected by InversionModel.forward()."""
        return {
            "embedder_input_ids": inputs.get("embedder_input_ids"),
            "embedder_attention_mask": inputs.get("embedder_attention_mask"),
            "labels": inputs.get("labels"),
            "frozen_embeddings": inputs.get("frozen_embeddings"),
        }

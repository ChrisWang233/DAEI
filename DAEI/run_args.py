import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from DAEI.models import (
    EMBEDDER_MODEL_NAMES,
    EMBEDDING_TRANSFORM_STRATEGIES,
    FREEZE_STRATEGIES,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DATASET_NAMES = [
    "nq",
    "luar_reddit",
    "msmarco",
    "yahoo",
    "one_million_instructions",
    "one_million_paired_instructions",
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="t5-base",
        metadata={
            "help": (
                "The model checkpoint for weights initialization .Don't set if you want to train a model from scratch."
            )
        },
    )
    embedder_model_name: str = field(
        ###
        ## huggingface.co/facebook/dpr-ctx_encoder-single-nq-base
        ###
        default="gtr_base",
        metadata={
            "help": "Model to get embeddings from (locally)",
            "choices": EMBEDDER_MODEL_NAMES,
        },
    )
    embedder_model_api: Optional[str] = field(
        default=None, metadata={"help": "API to get embeddings from"}
    )
    embedder_gaussian_noise_level: float = field(
        default=0.0, metadata={"help": "noise level to add during training to embedder"}
    )
    embedder_torch_dtype: str = field(
        default="float32",
        metadata={
            "help": "torch dtype of embedder",
            "choices": ["float32", "float16", "bfloat16"],
        },
    )
    embedding_transform_strategy: str = field(
        default="repeat",
        metadata={
            "help": "Strategy for transforming from sentence embedding into sequence-level input for encoder-decoder",
            "choices": EMBEDDING_TRANSFORM_STRATEGIES,
        },
    )
    encoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 encoder"}
    )
    decoder_dropout_disabled: bool = field(
        default=False, metadata={"help": "Disable dropout on T5 decoder"}
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum sequence length for tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    num_repeat_tokens: int = field(
        default=16,
        metadata={
            "help": "Number of times to repeat embedding along T5 input sequence length."
        },
    )
    embedding_zero_except_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": "For inverting with logits, will set all numbers in embedding except the top-K to -30."
        },
    )
    embedder_no_grad: bool = field(
        default=True, metadata={"help": "Whether to disable grads for DPR"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LORA+int8 for fine-tuning"}
    )
    embedder_fake_with_zeros: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass all zeros as embedding (and not use DPR at all)"
        },
    )
    use_frozen_embeddings_as_input: bool = field(
        default=False,
        metadata={
            "help": "Whether to pass a 'frozen_embedding' column and train on that instead of generating embeddings on-the-fly"
        },
    )
    corrector_ignore_hypothesis_embedding: bool = field(
        default=False,
        metadata={
            "help": "If set, and training corrector encoder, will ignore the hypothesis embedding"
        },
    )
    embeddings_from_layer_n: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, uses embeddings from layer n - for example set to 0 to use word embeddings"
        },
    )
    freeze_strategy: str = field(
        default="none",
        metadata={
            "help": "which part of the model to freeze",
            "choices": FREEZE_STRATEGIES,
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="msmarco",
        metadata={
            "help": "Dataset name(s), comma-separated for combined training (e.g. 'nq,msmarco').",
        },
    )
    max_eval_samples: int = field(
        default=1000,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )
    use_full_data_datasets: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated subset of --dataset_name entries that must use ALL rows (no cap from "
                "--use_less_data). Example: --dataset_name nq,msmarco,yahoo --use_less_data 5000000 "
                "--use_full_data_datasets yahoo  →  nq+msmarco are concatenated, shuffled, then capped at "
                "5M; Yahoo train/val are appended in full; final train is shuffled again. "
                "Each name must appear in --dataset_name."
            )
        },
    )
    dataset_mode: str = field(
        default="standard",
        metadata={
            "help": "Data mode: 'standard' (clean emb, cache path nq_standard_-1_gtr_base_train.arrow), "
            "'noisy' (noise added when generating cache, nq_noisy_-1_gtr_base_train.arrow), "
            "'denoised' (stage-2 DAEI cache, auto-created from matching noisy cache if missing). "
            "Cache reused if exists.",
            "choices": ["standard", "noisy", "n2n", "denoised"],
        },
    )

    @property
    def dataset_names(self):
        """Return list of dataset names (split on comma)."""
        if isinstance(self.dataset_name, list):
            return self.dataset_name
        return [n.strip() for n in self.dataset_name.split(",") if n.strip()]

    @property
    def dataset_name_str(self):
        """Joined dataset name for display / cache keys (e.g. 'nq_msmarco')."""
        return "_".join(self.dataset_names)

    @property
    def use_full_data_dataset_names(self) -> List[str]:
        """Names that use full corpus (from use_full_data_datasets)."""
        raw = getattr(self, "use_full_data_datasets", None)
        if not raw:
            return []
        return [n.strip() for n in str(raw).split(",") if n.strip()]

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")
        if self.use_full_data_datasets:
            allowed = set(self.dataset_names)
            for n in self.use_full_data_dataset_names:
                if n not in allowed:
                    raise ValueError(
                        f"use_full_data_datasets contains '{n}' which is not in --dataset_name "
                        f"({sorted(allowed)}). Names must match exactly (e.g. yahoo, not Yahoo)."
                    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # https://github.com/huggingface/transformers/blob/e82c1cb78e178519060b9391214727be75a218ca/src/transformers/training_args.py#L121
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to saves/<random hash>."
        },
    )
    corrector_model_from_pretrained: Optional[str] = field(
        default=None,
        metadata={
            "help": "Deprecated alias for --inverter_checkpoint."
        },
    )
    inverter_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Inverter checkpoint used to generate initial hypotheses for Corrector training."
        },
    )
    cheat_on_train_hypotheses: bool = field(
        default=False,
        metadata={
            "help": "When set, will interpolate true with pred train hypothesis for 'closer' training data"
        },
    )

    steps_per_epoch: int = field(
        default=500_000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=0.0005,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    report_to: str = "wandb"
    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )
    # torch_compile: bool = True # for torch 2

    ##################### Experimental Settings ####################
    experiment: str = field(
        default="inversion",
        metadata={
            "required": False,
            "help": "Which experiment to run (defines model, loss func, dataset...) ",
            "choices": [
                "inversion",  # our model: projects and feeds to encoder-decoder
                "reranking",
                "corrector",
                "corrector_encoder",
                "joint_inversion",  # DAE + inverter joint training (SURE_CE)
                "dae_shield_corrector",  # Corrector with frozen DAE Shield
            ],
        },
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )
    exp_group_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this sweep / series of experiments",
        },
    )

    # Need to *not* remove unused columns so we keep query_attention_mask, etc.
    # which huggingface doesn't think we need.
    remove_unused_columns: bool = False

    # Do evaluation and logging on certain num steps.
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 3  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=4000, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=10000,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=10000,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )
    mock_embedder: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, will delete the embedder and replace all embedder logits with"
                " zeros once training starts. You probably don't want to do this. But "
                " if you precomputed all the embeddings for train and val, this will"
                " work fine, except the embedding-based metrics (just cosine similarity"
                " I think) will be broken."
            )
        },
    )
    eval_log_text_metrics_only: bool = field(
        default=True,
        metadata={
            "help": (
                "If true, after each eval the metrics dict only keeps loss, BLEU, "
                "token-set F1, ROUGE, exact match, embedding cosine similarity, and "
                "epoch. Drops SEMs, runtime, token counts, n-gram counts, perplexity, "
                "accuracy, and other telemetry."
            )
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    include_inputs_for_metrics: bool = True

    ##################### Joint Training Settings ####################
    training_stage: int = field(
        default=0,
        metadata={
            "help": "Joint training stage: 0=v2t only, 1=freeze v2t train DAE, "
            "2=joint fine-tune (original), 3=DAEI joint fine-tune (CE grad flows to DAE)",
            "choices": [0, 1, 2, 3],
        },
    )
    loss_mode: str = field(
        default="n2n",
        metadata={
            "help": "DAE loss: 'sure' (MC-SURE) or 'n2n' (Noise2Noise)",
            "choices": ["sure", "n2n"],
        },
    )
    dae_hidden_dim: int = field(
        default=1024, metadata={"help": "Hidden dim of ResidualDAE blocks"}
    )
    dae_depth: int = field(
        default=2, metadata={"help": "Number of residual blocks in DAE"}
    )
    dae_lr: float = field(
        default=1e-3, metadata={"help": "Learning rate for the DAE optimizer"}
    )
    lambda_ce: float = field(
        default=1.0, metadata={"help": "Target CE weight in stage-2 joint loss"}
    )
    lambda_warmup_steps: int = field(
        default=5000, metadata={"help": "Steps to linearly warm up lambda_ce from 0 to target"}
    )
    sure_n_probes: int = field(
        default=5, metadata={"help": "Number of Rademacher probes for MC-SURE"}
    )
    noise_sigma: float = field(
        default=0.01, metadata={"help": "Known noise std for SURE / N2N"}
    )

    ##################### DAEI Pipeline Enhancements ####################
    dae_use_sigma_cond: bool = field(
        default=False, metadata={"help": "Enable sigma conditioning in ResidualDAE (for variable-sigma training)"}
    )
    dae_sigma_schedule: str = field(
        default="fixed",
        metadata={
            "help": "Noise sigma schedule for DAE training: 'fixed' uses noise_sigma, "
            "'log_uniform' decays from dae_sigma_max to dae_sigma_min over training",
            "choices": ["fixed", "log_uniform"],
        },
    )
    dae_sigma_max: float = field(
        default=0.02, metadata={"help": "Maximum sigma for log_uniform schedule (start of training). "
                                "Should be modest (e.g. 2-3x noise_sigma) — too large causes SURE to diverge."}
    )
    dae_sigma_min: float = field(
        default=0.01, metadata={"help": "Minimum sigma for log_uniform schedule (end of training)"}
    )
    dae_sigma_decay_span_fraction: float = field(
        default=0.2,
        metadata={
            "help": "For log_uniform only: decay σ from dae_sigma_max to dae_sigma_min over the "
            "first this fraction of estimated total optimizer steps (e.g. 0.2 = first fifth of the run). "
            "After that σ stays at dae_sigma_min so the model does not keep training on a "
            "dominant high-noise objective. Ignored when dae_sigma_schedule=fixed."
        },
    )
    dae_contrastive_weight: float = field(
        default=0.0, metadata={"help": "Weight (beta) for InfoNCE contrastive loss in DAE training (0 = disabled)"}
    )
    dae_contrastive_tau: float = field(
        default=0.07, metadata={"help": "Temperature for InfoNCE contrastive loss"}
    )
    dae_grad_max_norm: float = field(
        default=0.0, metadata={"help": "DAE-specific gradient clip norm (0 = use global max_grad_norm)"}
    )
    dae_use_spectral_norm: bool = field(
        default=False,
        metadata={"help": "Apply spectral normalization to DAE Linear layers (bounds Lipschitz constant, stabilizes SURE Jacobian trace)"},
    )
    dae_pcgrad: bool = field(
        default=False,
        metadata={"help": "Enable PCGrad (Projecting Conflicting Gradients) in Stage 3 joint training. "
                  "Projects CE gradient onto the normal plane of SURE gradient when they conflict."},
    )
    stage0_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Stage-0 v2t checkpoint path (used by stage ≥ 2; ignored for stage 1)"}
    )
    stage1_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Stage-1 DAE checkpoint path (used by stage 2)"}
    )
    stage2_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "DAEI Stage 2 checkpoint (used by stage 3)"}
    )
    stage2_inverter_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Deprecated alias for --stage2_checkpoint."}
    )

    ##################### DAE Shield Corrector Settings ####################
    use_dae_shield: bool = field(
        default=False, metadata={"help": "Enable DAE Shield in the Corrector pipeline"}
    )
    dae_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to frozen DAE checkpoint (model.safetensors or dae.pt dir)"},
    )
    denoised_embeddings_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to offline precomputed denoised embeddings (sharded .pt)"},
    )
    max_correction_steps: int = field(
        default=5,
        metadata={"help": "Maximum Corrector self-correction steps (DAE Shield default: 5)"},
    )
    early_stop_threshold: float = field(
        default=1e-3,
        metadata={"help": "Cosine similarity change threshold for noise-aware early stopping"},
    )

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        num_workers = torch.cuda.device_count()
        os.environ["RAYON_RS_NUM_CPUS"] = str(
            num_workers
        )  # Sets threads for hf tokenizers
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        # defaults from SentenceTransformers
        # lr 2e-5
        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"

        self.load_best_model_at_end = True
        if self.greater_is_better is None:
            metric_name = str(self.metric_for_best_model or "").lower()
            self.greater_is_better = not (
                metric_name == "loss" or metric_name.endswith("_loss")
            )

        self.do_eval = False
        # self.ddp_backend = "gloo"

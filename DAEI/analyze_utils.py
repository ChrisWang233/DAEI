import glob
import json
import os
import shlex
import sys
from typing import Any, Dict, Optional

from accelerate.state import PartialState
import pandas as pd
import torch
import transformers
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from DAEI import experiments
from DAEI.models.config import InversionConfig
from DAEI.run_args import DataArguments, ModelArguments, TrainingArguments
from DAEI import run_args as run_args

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
transformers.logging.set_verbosity_error()

#############################################################################


def _load_args_bin(path: str):
    """Load pickled DataArguments / ModelArguments / TrainingArguments from ``*.bin``.

    PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
    pickled dataclasses. Args shards are trusted local artifacts, not public checkpoints.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_experiment_and_trainer(
    checkpoint_folder: str,
    args_str: Optional[str] = None,
    checkpoint: Optional[str] = None,
    do_eval: bool = True,
    sanity_decode: bool = True,
    max_seq_length: Optional[int] = None,
    use_less_data: Optional[int] = None,
):  # (can't import due to circluar import) -> trainers.InversionTrainer:
    # import previous aliases so that .bin that were saved prior to the
    # existence of the DAEI module will still work.

    if checkpoint is None:
        checkpoint = get_last_checkpoint(checkpoint_folder)  # a checkpoint
    if checkpoint is None:
        # This happens in a weird case, where no model is saved to saves/xxx/checkpoint-*/pytorch_model.bin
        # because checkpointing never happened (likely a very short training run) but there is still a file
        # available in saves/xxx/pytorch_model.bin.
        checkpoint = checkpoint_folder
    print("[analyze_utils] Loading model from checkpoint:", checkpoint)


    cwd = os.path.dirname(
        os.path.abspath(__file__)
    )
    print(f"[analyze_utils] adding cwd to path: {cwd}")
    sys.path.append(cwd)

    if args_str is not None:
        args = shlex.split(args_str)
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=args
        )
    else:
        print("[analyze_utils] loading args from checkpoint", checkpoint)
        try:
            print("[analyze_utils] loading data_args from", os.path.join(checkpoint, os.pardir, "data_args.bin"))
            data_args = _load_args_bin(os.path.join(checkpoint, os.pardir, "data_args.bin"))
        except (FileNotFoundError):
            data_args = _load_args_bin(os.path.join(checkpoint, "data_args.bin"))
        try:
            model_args = _load_args_bin(
                os.path.join(checkpoint, os.pardir, "model_args.bin")
            )
        except (FileNotFoundError):
            model_args = _load_args_bin(os.path.join(checkpoint, "model_args.bin"))
        try:
            training_args = _load_args_bin(
                os.path.join(checkpoint, os.pardir, "training_args.bin")
            )
        except (FileNotFoundError):
            training_args = _load_args_bin(os.path.join(checkpoint, "training_args.bin"))

    training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.use_wandb = False
    training_args.report_to = []
    training_args.mock_embedder = False
    training_args.no_cuda = not torch.cuda.is_available()

    if max_seq_length is not None:
        print(
            f"Overwriting max sequence length from {model_args.max_seq_length} to {max_seq_length}"
        )
        model_args.max_seq_length = max_seq_length

    if use_less_data is not None:
        print(
            f"Overwriting use_less_data from {data_args.use_less_data} to {use_less_data}"
        )
        data_args.use_less_data = use_less_data

    # For batch decoding outputs during evaluation.
    # os.environ["TOKENIZERS_PARALLELISM"] = "True"

    ########################################################################
    print("> checkpoint:", checkpoint)
    if (
        checkpoint
        == "/home/jxm3/research/retrieval/inversion/saves/47d9c149a8e827d0609abbeefdfd89ac/checkpoint-558000"
    ):
        # Special handling for one case of backwards compatibility:
        #   set dataset (which used to be empty) to nq
        data_args.dataset_name = "nq"
        print("set dataset to nq")

    if not torch.cuda.is_available():
        print("[analyze_utils] No GPU available, loading model on CPU")
        training_args.use_cpu = True
        training_args._n_gpu = 0
        training_args.local_rank = -1  # Don't load in DDP
        training_args.distributed_state = PartialState()
        training_args.deepspeed_plugin = None  # For backwards compatibility
        training_args.bf16 = 0  # no bf16 in case no support from GPU
    else:
        # Pickled TrainingArguments from multi-GPU DDP runs may carry stale Accelerate
        # state; HF Trainer.evaluate() then hits:
        #   AcceleratorState has no attribute distributed_type
        # Rebuild a single-process PartialState before constructing the Trainer.
        training_args.use_cpu = False
        training_args.no_cuda = False
        training_args._n_gpu = 1
        training_args.local_rank = -1
        training_args.distributed_state = PartialState()
        training_args.deepspeed_plugin = None
    
    # Need to delete this cached property so that it's properly recomputed.
    if '__cached__setup_devices' in training_args.__dict__:
        del training_args.__dict__["__cached__setup_devices"]

    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer.model._keys_to_ignore_on_save = []
    # Trainer.__init__ runs create_accelerator_and_postprocess() before args._setup_devices;
    # _setup_devices resets AcceleratorState and leaves self.accelerator stale (evaluate() then
    # raises "AcceleratorState has no attribute distributed_type"). Rebuild the Accelerator once.
    trainer.create_accelerator_and_postprocess()
    try:
        trainer._load_from_checkpoint(checkpoint)
    except RuntimeError:
        # backwards compatibility from adding/removing layernorm
        trainer.model.use_ln = False
        trainer.model.layernorm = None
        # try again without trying to load layernorm
        trainer._load_from_checkpoint(checkpoint)
    if torch.cuda.is_available() and sanity_decode:
        trainer.sanity_decode()
    return experiment, trainer


def load_trainer(
    *args, **kwargs
):  # (can't import due to circluar import) -> trainers.Inversion
    experiment, trainer = load_experiment_and_trainer(*args, **kwargs)
    return trainer


def load_results_from_folder(name: str) -> pd.DataFrame:
    filenames = glob.glob(os.path.join(name, "*.json"))
    data = []
    for f in filenames:
        d = json.load(open(f, "r"))
        if "_eval_args" in d:
            # unnest args for evaluation
            d.update(d.pop("_eval_args"))
        data.append(d)
    return pd.DataFrame(data)


def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args


def load_experiment_and_trainer_from_pretrained(
    name: str,
    use_less_data: int = 1000,
    data_args_overrides: Optional[Dict[str, Any]] = None,
    model_args_overrides: Optional[Dict[str, Any]] = None,
    training_args_overrides: Optional[Dict[str, Any]] = None,
):
    config = InversionConfig.from_pretrained(name)
    model_args = args_from_config(ModelArguments, config)
    data_args = args_from_config(DataArguments, config)
    training_args = args_from_config(TrainingArguments, config)

    data_args.use_less_data = use_less_data
    for key, value in (data_args_overrides or {}).items():
        if hasattr(data_args, key):
            setattr(data_args, key, value)
    for key, value in (model_args_overrides or {}).items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
    for key, value in (training_args_overrides or {}).items():
        if hasattr(training_args, key):
            setattr(training_args, key, value)
    #######################################################################
    training_args._n_gpu = 1 if torch.cuda.is_available() else 0  # Don't load in DDP
    training_args.bf16 = 0  # no bf16 in case no support from GPU
    training_args.local_rank = -1  # Don't load in DDP
    training_args.distributed_state = PartialState()
    training_args.deepspeed_plugin = None  # For backwards compatibility
    # training_args.dataloader_num_workers = 0  # no multiprocessing :)
    training_args.use_wandb = False
    training_args.report_to = []
    training_args.mock_embedder = False
    training_args.output_dir = "saves/" + name.replace("/", "__")
    ########################################################################

    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    trainer = experiment.load_trainer()
    trainer.model = trainer.model.__class__.from_pretrained(name)
    for key, value in (data_args_overrides or {}).items():
        setattr(trainer.model.config, key, value)
    for key, value in (model_args_overrides or {}).items():
        setattr(trainer.model.config, key, value)
    setattr(trainer.model.config, "use_less_data", use_less_data)
    trainer.model.to(training_args.device)
    return experiment, trainer

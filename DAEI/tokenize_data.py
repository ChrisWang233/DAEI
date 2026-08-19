from typing import Callable, Dict

import torch
import transformers

from DAEI.models import InversionModel


def tokenize_function(
    tokenizer: transformers.PreTrainedTokenizer,
    embedder_tokenizer: transformers.PreTrainedTokenizer,
    text_column_name: str,
    max_seq_length: int,
    padding: bool = False,
    prefix: str = None,
) -> Callable[[Dict], Dict]:
    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if prefix:
            texts = [f"{prefix}: {text}" for text in examples[text_column_name]]
        else:
            texts = examples[text_column_name]
        output = tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            examples[text_column_name],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def tokenize_function_noisy_frozen(
    tokenizer: transformers.PreTrainedTokenizer,
    text_column_name: str,
    max_seq_length: int,
    padding: bool = False,
    prefix: str = None,
) -> Callable[[Dict], Dict]:
    """Tokenize for noisy_frozen mode: no embedder call, use frozen_embeddings from dataset.

    Produces: input_ids, attention_mask, labels, length, embedder_input_ids, embedder_attention_mask (dummy),
    and preserves frozen_embeddings from the batch.
    """
    def tokenize_function_inner(examples) -> Dict:
        if prefix:
            texts = [f"{prefix}: {t}" for t in examples[text_column_name]]
        else:
            texts = examples[text_column_name]
        output = tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )
        output["labels"] = [
            [(-100 if tid == tokenizer.pad_token_id else tid) for tid in ids]
            for ids in output["input_ids"]
        ]
        output["length"] = [
            (torch.tensor(ids) != tokenizer.pad_token_id).sum().item()
            for ids in output["input_ids"]
        ]
        # Dummy embedder inputs (model uses frozen_embeddings when present)
        batch_size = len(output["input_ids"])
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        output["embedder_input_ids"] = [[pad_id] * max_seq_length for _ in range(batch_size)]
        output["embedder_attention_mask"] = [[1] * max_seq_length for _ in range(batch_size)]
        # frozen_embeddings is already in examples, pass through
        if "frozen_embeddings" in examples:
            output["frozen_embeddings"] = examples["frozen_embeddings"]
        return output

    return tokenize_function_inner


def tokenize_function_llama_chat(
    tokenizer,
    embedder_tokenizer,
    text_column_name,
    max_seq_length,
    padding: bool = False,
    # no-op for compatibility with other tokenization functions
    prefix: str = None,
) -> Callable[[Dict], Dict]:
    """Use special tokenization for LLAMA chat models."""

    def tokenize_function_inner(examples) -> Dict[str, torch.Tensor]:
        if "prefix" not in examples:
            # hacky way to turn datasets into the right format for LLAMA chat.
            # "real" prompt datasets like one_million_paired_instructions
            # have "prefix" and "suffix" already.
            #
            # so this is only for evaluation datasets that may not have
            # actual prefix-suffix pairing.
            #
            examples["prefix"] = [""] * len(examples[text_column_name])
            examples["suffix"] = examples[text_column_name]

        output = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        # copy to 'labels' for language modeling loss
        # but set padding to -100
        # github.com/huggingface/transformers/blob/cbe63949d76efd153a1f389f38fe9ce1287e06b0/src/transformers/models/t5/modeling_t5.py#L1504-L1507
        output["labels"] = [
            [
                (-100 if token_id == tokenizer.pad_token_id else token_id)
                for token_id in ids
            ]
            for ids in output["input_ids"]
        ]
        embedder_output = embedder_tokenizer(
            text=[
                f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n {instruction} [/INST]"
                for (system_message, instruction) in zip(
                    examples["prefix"], examples["suffix"]
                )
            ],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedder_output = {f"embedder_{k}": v for k, v in embedder_output.items()}

        output["length"] = [
            (torch.tensor(input_ids) != tokenizer.pad_token_id).sum().item()
            for input_ids in output["input_ids"]
        ]

        return {**output, **embedder_output}

    return tokenize_function_inner


def embed_dataset_batch(
    model: InversionModel,
    batch: Dict,
    noise_sigma: float = 0.0,
    store_clean: bool = False,
) -> Dict:
    """Compute embeddings and optionally add Gaussian noise (for dataset_mode=noisy).

    When ``store_clean`` is True and ``noise_sigma > 0``, also store the clean
    (non-noisy) embeddings under the ``clean_embeddings`` key. This is used for
    SURE evaluation to compare noisy vs. denoised vs. clean embeddings.
    """
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        emb = model.call_embedding_model(**emb_input_ids)
        if noise_sigma > 0:
            if store_clean:
                batch["clean_embeddings"] = emb
            emb = emb + noise_sigma * torch.randn_like(emb, device=emb.device)
        batch["frozen_embeddings"] = emb
    return batch


def embed_dataset_batch_clean_only(
    model: InversionModel,
    batch: Dict,
) -> Dict:
    """Add only clean (no-noise) embeddings as 'clean_embeddings'. Does not modify 'frozen_embeddings'.
    Used to backfill clean_embeddings for existing val caches that only have noisy frozen_embeddings.
    """
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        batch["clean_embeddings"] = model.call_embedding_model(**emb_input_ids)
    return batch


def embed_dataset_batch_n2n(
    model: InversionModel,
    batch: Dict,
    noise_sigma: float,
) -> Dict:
    """Compute embeddings once, then add two independent noise vectors for N2N training.

    Produces columns: frozen_embeddings_a, frozen_embeddings_b (and removes frozen_embeddings).
    """
    assert "input_ids" in batch.keys(), f"invalid keys {batch.keys()}"
    assert hasattr(model, "call_embedding_model")
    assert noise_sigma > 0, "N2N requires noise_sigma > 0"

    input_ids = batch["input_ids"]
    inputs_str = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    emb_input_ids = model.embedder_tokenizer(
        inputs_str,
        max_length=model.config.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(next(model.parameters()).device)

    with torch.no_grad():
        clean_emb = model.call_embedding_model(**emb_input_ids)
        batch["frozen_embeddings_a"] = clean_emb + noise_sigma * torch.randn_like(clean_emb)
        batch["frozen_embeddings_b"] = clean_emb + noise_sigma * torch.randn_like(clean_emb)
    return batch


def get_tokenizer_mapping(
    lm: str, inverter: str, inverter_vocab_size: int
) -> torch.Tensor:
    """Computes the mapping from token outputs in `lm`'s vocabulary to those in `inverter's
    vocabulary. Makes some assumptions about spacing.
    """
    lm_tokenizer = transformers.AutoTokenizer.from_pretrained(lm)
    inverter_tokenizer = transformers.AutoTokenizer.from_pretrained(inverter)

    lm_vocab = lm_tokenizer.vocab
    mapping = torch.zeros(len(lm_vocab), dtype=torch.long)
    for k, idx in lm_tokenizer.vocab.items():
        # We replace space tokens with nothing and allow the call to
        # inverter_tokenizer.decode to determine this. We also
        # filter out 2 and 3 as first tokens which are extremely common
        # when the T5 tokenizer processes unicode. (These are hacks
        # specific to the LLAMA-T5 lm-inverter pairing, and it would
        # be better to find an automated wa to do this later.)
        mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[0]
        if mapping[idx] in [2, 3]:
            mapping[idx] = inverter_tokenizer.encode(k.replace("▁", " "))[1]

    preservation = len(set(mapping.tolist())) / len(lm_vocab)
    print(
        f"Mapped tokenizer {lm} to {inverter}. Preserved {preservation*100:.1f}% of unique tokens."
    )
    return mapping

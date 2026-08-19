import logging
import html
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import datasets
import numpy as np
import torch

from DAEI.run_args import DataArguments
from DAEI.utils import dataset_map_multi_worker, get_num_proc


def retain_dataset_columns(
    d: datasets.Dataset, allowed_columns: List[str]
) -> datasets.Dataset:
    column_names_to_remove = [c for c in d.features if c not in allowed_columns]
    return d.remove_columns(column_names_to_remove)


def load_nq_dpr_corpus() -> datasets.Dataset:
    dataset_dict = datasets.load_dataset("jxm/nq_corpus_dpr")
    return dataset_dict["train"]


def load_msmarco_corpus() -> datasets.Dataset:
    # has columns ["title", "text"]. only one split ("train")
    dataset_dict = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
    return dataset_dict["train"]



def load_yahoo_corpus() -> datasets.Dataset:
    dataset_dict = datasets.load_dataset(
        "sentence-transformers/yahoo-answers",
        "question-answer-pair",
    )
    d = dataset_dict["train"]
    d = dataset_map_multi_worker(
        d,
        map_fn=create_yahoo_qa_ex,
        num_proc=get_num_proc(),
    )
    d = d.filter(lambda ex: bool(ex["text"].strip()))
    return retain_dataset_columns(d, ["text"])


def _clean_yahoo_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def create_yahoo_qa_ex(ex: Dict[str, str]) -> Dict[str, str]:
    question = _clean_yahoo_text(ex.get("question", ""))
    answer = _clean_yahoo_text(ex.get("answer", ""))
    ex["text"] = "\n".join(part for part in (question, answer) if part)
    return ex


def create_omi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["text"] = ex["user"]
    return ex


def create_ompi_ex(ex: Dict[str, str]) -> Dict[str, str]:
    ex["user"] = ex["user"].strip()
    ex["system"] = ex["system"].strip()
    ex["text"] = ex["system"] + "\n\n" + ex["user"]
    ex["prefix"] = ex["system"] + "\n\n"
    ex["suffix"] = ex["user"]
    return ex


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def load_one_million_paired_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-paired-instructions")
    dataset_dict = dataset_map_multi_worker(
        dataset_dict,
        map_fn=create_ompi_ex,
        num_proc=get_num_proc(),
    )

    return dataset_dict["train"]


def load_one_million_instructions() -> datasets.Dataset:
    # has only "train" split, and "system" (system prompt)
    # and "user" (user input) columns
    dataset_dict = datasets.load_dataset("wentingzhao/one-million-instructions")
    dataset_dict = dataset_map_multi_worker(dataset_dict, create_ompi_ex)

    return dataset_dict["train"]


def load_anthropic_toxic_prompts() -> datasets.Dataset:
    d = datasets.load_dataset("wentingzhao/anthropic-hh-first-prompt")["train"]
    d = d.rename_column("user", "text")
    return d


def load_luar_reddit() -> datasets.Dataset:
    d = datasets.load_dataset("friendshipkim/reddit_eval_embeddings_luar")
    d = d.rename_column("full_text", "text")
    d = d.rename_column("embedding", "frozen_embeddings")
    return d


def _load_single_dataset(name: str) -> datasets.DatasetDict:
    """Load a single dataset by name, returning DatasetDict with 'train' and 'validation'."""
    if name == "nq":
        raw = load_nq_dpr_corpus()
        raw = raw.train_test_split(test_size=0.01)
        raw["validation"] = raw["test"]
        return raw
    elif name == "msmarco":
        raw = load_msmarco_corpus()
        raw = raw.train_test_split(test_size=0.01)
        raw["validation"] = raw["test"]
        return raw
    elif name == "yahoo":
        raw = load_yahoo_corpus()
        raw = raw.train_test_split(test_size=0.01)
        raw["validation"] = raw["test"]
        return raw
    elif name == "one_million_instructions":
        raw = load_one_million_instructions()
        raw = raw.train_test_split(test_size=0.01)
        raw["validation"] = raw["test"]
        return raw
    elif name == "one_million_paired_instructions":
        raw = load_one_million_paired_instructions()
        raw = raw.train_test_split(test_size=0.01)
        raw["validation"] = raw["test"]
        return raw
    elif name == "luar_reddit":
        all_luar = load_luar_reddit()
        return datasets.DatasetDict(
            {"train": all_luar["candidates"], "validation": all_luar["queries"]}
        )
    else:
        raise ValueError(f"unsupported dataset '{name}'")


def _load_train_val_trimmed(name: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Load one corpus; keep only text + frozen_embeddings columns."""
    ds = _load_single_dataset(name)
    for split in ("train", "validation"):
        if split in ds:
            ds[split] = retain_dataset_columns(ds[split], ["text", "frozen_embeddings"])
    return ds["train"], ds["validation"]


def dataset_from_args(data_args: DataArguments) -> datasets.DatasetDict:
    """Loads dataset(s) from data_args. Supports comma-separated multi-dataset (e.g. 'nq,msmarco').

    If ``use_full_data_datasets`` is set, those names are never truncated by ``use_less_data``;
    other names are concatenated, shuffled, then capped at ``use_less_data`` (when >0), then merged
    with the full corpora and shuffled again.
    """
    names = data_args.dataset_names
    full_set: Set[str] = set(data_args.use_full_data_dataset_names)

    if len(names) == 1:
        return _load_single_dataset(names[0])

    if not full_set:
        all_train = []
        all_val = []
        for name in names:
            tr, va = _load_train_val_trimmed(name)
            all_train.append(tr)
            all_val.append(va)

        combined_train = datasets.concatenate_datasets(all_train)
        combined_val = datasets.concatenate_datasets(all_val)
        combined_train = combined_train.shuffle(seed=42)
        combined_val = combined_val.shuffle(seed=42)
        logging.info(
            "Combined %d datasets: train=%d, validation=%d",
            len(names), len(combined_train), len(combined_val),
        )
        return datasets.DatasetDict({"train": combined_train, "validation": combined_val})

    # --- Split: limited (subject to use_less_data) vs full (always all rows) ---
    limited_names = [n for n in names if n not in full_set]
    full_names = [n for n in names if n in full_set]

    if not full_names:
        raise ValueError("use_full_data_datasets was set but no matching names in dataset_name")

    uld = getattr(data_args, "use_less_data", -1) or -1

    if not limited_names:
        # All corpora are "full": concatenate everything, no cap here (global truncate skipped in Experiment).
        all_train = []
        all_val = []
        for name in names:
            tr, va = _load_train_val_trimmed(name)
            all_train.append(tr)
            all_val.append(va)
        combined_train = datasets.concatenate_datasets(all_train).shuffle(seed=42)
        combined_val = datasets.concatenate_datasets(all_val).shuffle(seed=42)
        logging.info(
            "Combined %d full datasets (use_full_data_datasets): train=%d, validation=%d",
            len(names), len(combined_train), len(combined_val),
        )
        return datasets.DatasetDict({"train": combined_train, "validation": combined_val})

    lim_tr_parts = []
    lim_va_parts = []
    for name in limited_names:
        tr, va = _load_train_val_trimmed(name)
        lim_tr_parts.append(tr)
        lim_va_parts.append(va)
    lim_tr = datasets.concatenate_datasets(lim_tr_parts).shuffle(seed=42)
    lim_va = datasets.concatenate_datasets(lim_va_parts).shuffle(seed=42)
    if uld > 0:
        lim_tr = lim_tr.select(range(min(len(lim_tr), uld)))
        lim_va = lim_va.select(range(min(len(lim_va), uld)))

    ful_tr_parts = []
    ful_va_parts = []
    for name in full_names:
        tr, va = _load_train_val_trimmed(name)
        ful_tr_parts.append(tr)
        ful_va_parts.append(va)
    ful_tr = datasets.concatenate_datasets(ful_tr_parts).shuffle(seed=42)
    ful_va = datasets.concatenate_datasets(ful_va_parts).shuffle(seed=42)

    combined_train = datasets.concatenate_datasets([lim_tr, ful_tr]).shuffle(seed=42)
    combined_val = datasets.concatenate_datasets([lim_va, ful_va]).shuffle(seed=42)
    logging.info(
        "Combined limited %s (train_rows=%d, val_rows=%d; %s) + full %s → total train=%d, val=%d",
        limited_names,
        len(lim_tr),
        len(lim_va),
        f"capped at use_less_data={uld}" if uld > 0 else "not capped (use_less_data<=0)",
        full_names,
        len(combined_train),
        len(combined_val),
    )
    return datasets.DatasetDict({"train": combined_train, "validation": combined_val})


def load_n2n_dataset(data_path: str) -> datasets.DatasetDict:
    """Load a cached N2N dataset with (text, frozen_embeddings_a, frozen_embeddings_b) triplets.

    Expected columns: text, frozen_embeddings_a, frozen_embeddings_b
    (plus tokenized fields if already processed).
    """
    ds = datasets.load_from_disk(data_path)
    for split in ds:
        cols = set(ds[split].column_names)
        assert "frozen_embeddings_a" in cols and "frozen_embeddings_b" in cols, (
            f"N2N dataset at {data_path} missing N2N columns in split '{split}'. "
            f"Found: {cols}"
        )
    logging.info("Loaded N2N dataset from %s: %s", data_path, {k: len(v) for k, v in ds.items()})
    return ds


def load_ag_news_test() -> datasets.Dataset:
    return datasets.load_dataset("ag_news")["test"]


def load_xsum_val(col: str) -> datasets.Dataset:
    d = datasets.load_dataset("xsum")["validation"]
    d = d.rename_column(col, "text")
    return d


def load_wikibio_val() -> datasets.Dataset:
    d = datasets.load_dataset("wiki_bio", trust_remote_code=True)["val"]
    d = d.rename_column("target_text", "text")
    return d


def load_arxiv_val() -> datasets.Dataset:
    d = datasets.load_dataset("ccdv/arxiv-summarization")["validation"]
    d = d.rename_column("abstract", "text")
    return d

def load_indomain_val() -> datasets.Dataset:
    return _first_available_split(datasets.load_dataset("ChrisWang233/DAEI_indomain"))



def load_python_code_instructions_18k_alpaca() -> datasets.Dataset:
    d = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    d = d.rename_column("instruction", "text")
    return d


def _first_available_split(
    dataset_dict: datasets.DatasetDict,
    preferred_splits: Tuple[str, ...] = ("validation", "test", "train"),
) -> datasets.Dataset:
    for split in preferred_splits:
        if split in dataset_dict:
            return dataset_dict[split]
    available = list(dataset_dict.keys())
    if not available:
        raise ValueError("dataset has no splits")
    return dataset_dict[available[0]]


def _load_text_column_val_dataset(
    dataset_name: str,
    text_column: str,
) -> datasets.Dataset:
    dataset_dict = datasets.load_dataset(dataset_name)
    d = _first_available_split(dataset_dict)
    if text_column != "text":
        d = d.rename_column(text_column, "text")
    return d


def load_climate_fever_val() -> datasets.Dataset:
    return _load_text_column_val_dataset("tdiggelm/climate_fever", "claim")


def load_medmcqa_val() -> datasets.Dataset:
    return _load_text_column_val_dataset("openlifescienceai/medmcqa", "question")


def load_beir_corpus(name: str) -> List[str]:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    #### Download scifact.zip dataset and unzip the dataset
    beir_datasets_cache_dir = "/home/jxm3/research/retrieval/distractor_exp"

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        name
    )
    out_dir = os.path.join(beir_datasets_cache_dir, "datasets")
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Limit each corpus to first 100k documents.
    MAX_N = 100_000

    if name == "cqadupstack":
        full_corpus = []
        for folder in [
            "android",
            "english",
            "gaming",
            "gis",
            "mathematica",
            "physics",
            "programmers",
            "stats",
            "tex",
            "unix",
            "webmasters",
            "wordpress",
        ]:
            corpus, _queries, _qrels = GenericDataLoader(
                data_folder=os.path.join(data_path, folder)
            ).load(split="test")
            full_corpus.extend([k["text"] for k in corpus.values()])
        random.shuffle(full_corpus)
        return full_corpus[:MAX_N]
    else:
        corpus, _queries, _qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )
        corpus = [k["text"] for k in corpus.values()]
        return corpus[:MAX_N]


def load_beir_dataset(name: str) -> datasets.Dataset:
    cache_path = (
        datasets.config.HF_DATASETS_CACHE
    )  # something like /home/jxm3/.cache/huggingface/datasets
    dataset_path = os.path.join(cache_path, "emb_inv_beir", name)
    # print(f"loading BEIR dataset: {name}")
    if os.path.exists(dataset_path):
        logging.info("Loading BEIR dataset %s path %s", dataset_path)
        dataset = datasets.load_from_disk(dataset_path)
    else:
        logging.info(
            "Loading BEIR dataset %s from JSON (slow) at path %s", dataset_path
        )
        corpus = load_beir_corpus(name=name)
        dataset = datasets.Dataset.from_list([{"text": t} for t in corpus])
        os.makedirs(os.path.join(cache_path, "emb_inv_beir"), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        logging.info("Saved BEIR dataset as HF path %s", dataset_path)
    return dataset


def load_beir_datasets() -> datasets.DatasetDict:
    all_beir_datasets = [
        ####### public datasets #######
        "arguana",
        "climate-fever",
        "cqadupstack",
        "dbpedia-entity",
        "fever",
        "fiqa",
        "hotpotqa",
        "msmarco",
        "yahoo",
        "nfcorpus",
        "nq",
        "quora",
        "scidocs",
        "scifact",
        "trec-covid",
        "webis-touche2020",
        ####### private datasets #######
        "signal1m",
        "trec-news",
        "robust04",
        "bioasq",
    ]
    return datasets.DatasetDict({k: load_beir_dataset(k) for k in all_beir_datasets})

def load_yahoo_val() -> datasets.Dataset:
    """Sample up to 1,000 Yahoo examples for validation with a fixed seed."""
    d = load_yahoo_corpus()
    n = min(1000, len(d))
    return d.shuffle(seed=42).select(range(n))

def load_standard_val_datasets() -> datasets.DatasetDict:
    """Loads a pre-defined set of standard val datasets."""
    d = {
        "ag_news": load_ag_news_test(),
        "anthropic_toxic_prompts": load_anthropic_toxic_prompts(),
        # "arxiv": load_arxiv_val(),
        "python_code_alpaca": load_python_code_instructions_18k_alpaca(),
        # "xsum_doc": load_xsum_val("document"),
        # "xsum_summ": load_xsum_val("summary"),
        # "wikibio": load_wikibio_val(),
        # "yahoo": load_yahoo_val(),
        "indomain": load_indomain_val(),
        "climate_fever": load_climate_fever_val(),
        "medmcqa": load_medmcqa_val(),

    }
    d = {k: retain_dataset_columns(v, ["text"]) for k, v in d.items()}

    return datasets.DatasetDict(d)

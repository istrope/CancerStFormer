#!/usr/bin/env python3
"""
classifier_utils.py

Utility functions for data preprocessing for stFormer classification.
"""

import logging
import random
from collections import defaultdict, Counter
import os
from typing import Dict,Tuple
import pickle
from datasets import Dataset,DatasetDict,load_from_disk
import torch
from transformers import DataCollatorWithPadding


logger = logging.getLogger(__name__)
def load_and_filter(filter_data, nproc, input_data_file):
    """
    Load a dataset and apply filtering criteria.
    """
    data = load_from_disk(input_data_file)
    if filter_data:
        for key, values in filter_data.items():
            data = data.filter(lambda ex: ex[key] in values, num_proc=nproc)
    return data

def remove_rare(data, rare_threshold, state_key, nproc):
    """
    Remove rare labels based on a threshold.
    """
    total = len(data)
    counts = Counter(data[state_key])
    rare_labels = [label for label, count in counts.items() if count / total < rare_threshold]
    if rare_labels:
        data = data.filter(lambda ex: ex[state_key] not in rare_labels, num_proc=nproc)
    return data

def downsample_and_shuffle(data, max_ncells, max_ncells_per_class, cell_state_dict):
    """
    Shuffle the dataset and downsample overall and per-class if limits are provided.
    """
    data = data.shuffle(seed=42)
    if max_ncells and len(data) > max_ncells:
        data = data.select(range(max_ncells))
    if max_ncells_per_class:
        class_labels = data[cell_state_dict["state_key"]]
        indices = subsample_by_class(class_labels, max_ncells_per_class)
        data = data.select(indices)
    return data

def subsample_by_class(labels, N):
    """
    Subsample indices to at most N per class.
    """
    label_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_indices[label].append(idx)
    selected = []
    for label, indices in label_indices.items():
        if len(indices) > N:
            selected.extend(random.sample(indices, N))
        else:
            selected.extend(indices)
    return selected

def rename_cols(data, state_key):
    """
    Rename the state key column to the standard "label".
    """
    return data.rename_column(state_key, "label")

def flatten_list(l):
    """
    Flatten a list of lists.
    """
    return [item for sublist in l for item in sublist]



def _ensure_dataset(obj):
    if isinstance(obj, (Dataset, DatasetDict)):
        return obj
    raise TypeError(f"Expected Dataset or DatasetDict, got {type(obj)}")


def _map_over_splits(ds, fn, num_proc=1, **kwargs):
    if isinstance(ds, DatasetDict):
        return DatasetDict({k: v.map(fn, batched=True, num_proc=num_proc, **kwargs) for k, v in ds.items()})
    return ds.map(fn, batched=True, num_proc=num_proc, **kwargs)


def _filter_over_splits(ds, fn, num_proc=1):
    if isinstance(ds, DatasetDict):
        return DatasetDict({k: v.filter(fn, num_proc=num_proc) for k, v in ds.items()})
    return ds.filter(fn, num_proc=num_proc)


def label_classes(
    classifier: str,
    data: Dataset | DatasetDict,
    class_dict: Dict[str, list] | None,
    token_dict_path: str | None,
    nproc: int,
    model_mode: str = "spot",   # "spot" or "extended"
) -> Tuple[Dataset | DatasetDict, Dict[str, int] | None]:
    """
    Build label arrays for classification.

    For model_mode == "extended", masks tokens after halfway point in each sequence
    (len(input_ids) // 2) by setting labels to -100 beyond that index.
    """
    data = _ensure_dataset(data)

    if classifier == "sequence":
        logger.info("Sequence classification — labels handled in prepare_data()")
        return data, None
    if classifier != "gene":
        raise ValueError(f"Unknown classifier type: {classifier}")

    if class_dict is None:
        raise ValueError("class_dict is required for gene classification.")
    if not token_dict_path or not os.path.exists(token_dict_path):
        raise FileNotFoundError(f"token_dict_path not found: {token_dict_path}")

    # Map class names to IDs
    class_id_dict = {name: i for i, name in enumerate(sorted(class_dict.keys()))}

    # Load gene->token dictionary
    with open(token_dict_path, "rb") as f:
        gene2token = pickle.load(f)

    # Reverse map: token_id -> class_id
    token2class = {}
    skipped = []
    for cname, genes in class_dict.items():
        cid = class_id_dict[cname]
        for g in genes:
            tid = gene2token.get(g)
            if tid is None:
                skipped.append(g)
                continue
            token2class[tid] = cid
    if skipped:
        logger.warning("Skipped %d genes missing from token dict (showing up to 20): %s",
                       len(skipped), skipped[:20])

    # Mapper: assign class IDs and mask after midpoint for extended mode
    def _map_gene_labels(batch):
        out_labels = []
        for ids in batch["input_ids"]:
            labs = [token2class.get(int(t), -100) for t in ids]
            if model_mode == "extended":
                cutoff = len(labs) // 2
                labs[cutoff:] = [-100] * (len(labs) - cutoff)
            out_labels.append(labs)
        batch["labels"] = out_labels
        return batch

    data = _map_over_splits(data, _map_gene_labels, num_proc=nproc, batch_size=1000)

    # Drop examples with no supervised positions
    def _has_any_supervised(ex):
        return any(l != -100 for l in ex["labels"])

    data = _filter_over_splits(data, _has_any_supervised, num_proc=nproc)

    logger.info("Built gene labels (%s mode) for %d classes", model_mode, len(class_id_dict))
    return data, class_id_dict

class DataCollatorForCellClassification(DataCollatorWithPadding):
    """
    A data collator for cell classification that pads input_ids and collates labels.
    """
    def __call__(self, features):
        labels = [f["label"] for f in features]
        input_ids = [f["input_ids"] for f in features]
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return batch

_tf_tokenizer_logger = logging.getLogger("transformers.tokenization_utils_base")

class DataCollatorForGeneClassification:
    """
    Pads input_ids, attention_mask, and per-token labels for gene classification.
    - Uses tokenizer.pad() to handle all special tokens and masks.
    - Pads labels to the same length with label_pad_token_id (-100).
    """
    def __init__(
        self,
        tokenizer,
        padding="longest",          # or 'max_length'
        max_length=None,            # e.g. tokenizer.model_max_length
        label_pad_token_id=-100,
        return_tensors="pt"
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features):
        raw_labels = [f.pop("labels") for f in features]
        prev_level = _tf_tokenizer_logger.level
        _tf_tokenizer_logger.setLevel(logging.ERROR)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors
        )
        _tf_tokenizer_logger.setLevel(prev_level)

        labels = []
        for lab in raw_labels:
            if isinstance(lab, torch.Tensor):
                lab = lab.tolist()
            labels.append(lab)

        seq_len = batch["input_ids"].shape[1]
        padded_labels = [
            lab + [self.label_pad_token_id] * (seq_len - len(lab))
            for lab in labels
        ]

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

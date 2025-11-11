#!/usr/bin/env python3
"""
classifier_utils.py

Utility functions for data preprocessing for stFormer classification.
"""

import logging
import random
from collections import defaultdict, Counter
import os
from typing import Dict,Tuple,Union
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


def _ensure_dataset(data: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
    if isinstance(data, (Dataset, DatasetDict)):
        return data
    raise TypeError("data must be a datasets.Dataset or DatasetDict")

def _map_over_splits(data, fn, **kwargs):
    if isinstance(data, DatasetDict):
        return DatasetDict({k: v.map(fn, **kwargs) for k, v in data.items()})
    return data.map(fn, **kwargs)

def _filter_over_splits(data, fn, **kwargs):
    if isinstance(data, DatasetDict):
        return DatasetDict({k: v.filter(fn, **kwargs) for k, v in data.items()})
    return data.filter(fn, **kwargs)

def label_classes(
    classifier: str,
    data: Union[Dataset, DatasetDict],
    class_dict: Dict[str, list] | None,
    token_dict_path: str | None,
    nproc: int,
    model_mode: str = "spot",           # "spot" or "extended"
    boundary_col: str = "neighbor_boundary",
) -> Tuple[Union[Dataset, DatasetDict], Dict[str, int] | None]:
    """
    Build label arrays for classification.

    - classifier == "sequence": labels handled elsewhere (no-op here).
    - classifier == "gene":
        * Map tokens -> class IDs using class_dict and token_dict_path.
        * If model_mode == "extended":
              For each example, read `boundary_col` (index of neighbor start):
                - if boundary is None or <0: keep all tokens (spot-only)
                - else clamp to [0, len(input_ids)] and mask labels at indices >= boundary.
          If boundary_col is missing, fallback to midpoint (len // 2).
        * Drop examples with no supervised positions (all -100).
    """
    data = _ensure_dataset(data)

    if classifier == "sequence":
        logger.info("Sequence classification — labels handled in prepare_data(); returning data unchanged.")
        return data, None
    if classifier != "gene":
        raise ValueError(f"Unknown classifier type: {classifier!r}")

    if class_dict is None:
        raise ValueError("class_dict is required for gene classification.")
    if not token_dict_path or not os.path.exists(token_dict_path):
        raise FileNotFoundError(f"token_dict_path not found: {token_dict_path}")

    # class name -> id
    class_id_dict = {name: i for i, name in enumerate(sorted(class_dict.keys()))}

    # gene -> token_id
    with open(token_dict_path, "rb") as f:
        gene2token = pickle.load(f)

    # token_id -> class_id
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
        logger.warning(
            "Skipped %d genes missing from token dict (showing up to 20): %s",
            len(skipped), skipped[:20]
        )

    # core mapper
    def _map_gene_labels(batch):
        out_labels = []
        ids_list = batch["input_ids"]
        # boundary column may or may not exist in this split
        has_boundary = boundary_col in batch

        for i, ids in enumerate(ids_list):
            # label every token first
            labs = [ token2class.get(int(t), -100) for t in ids ]

            if model_mode == "extended":
                # Determine effective cutoff
                if has_boundary:
                    raw_nb = batch[boundary_col][i]
                    try:
                        nb = int(raw_nb)
                    except Exception:
                        nb = -1
                    if nb is None or nb < 0:
                        cutoff = len(labs)           # keep all (spot-only)
                    else:
                        cutoff = max(0, min(nb, len(labs)))
                else:
                    # fallback to midpoint if column not present
                    cutoff = len(labs) // 2

                # mask neighbor region
                for j in range(cutoff, len(labs)):
                    labs[j] = -100

            out_labels.append(labs)

        batch["labels"] = out_labels
        return batch

    data = _map_over_splits(data, _map_gene_labels, num_proc=nproc, batched=True, batch_size=1000)

    # keep only examples with ≥1 supervised token
    def _has_any_supervised(ex):
        return any(l != -100 for l in ex["labels"])

    data = _filter_over_splits(data, _has_any_supervised, num_proc=nproc)

    logger.info("Built gene labels (%s mode) for %d classes; boundary_col=%s",
                model_mode, len(class_id_dict), boundary_col)
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

#!/usr/bin/env python3
"""
classifier_utils.py

Utility functions for data preprocessing for stFormer classification.
"""

import logging
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from datasets import load_from_disk


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

def label_classes(classifier, data, class_dict, nproc):
    """
    Map class labels to numeric IDs. For cell classifiers, uses the "label" column.
    For gene classifiers, processes "input_ids" into "labels".
    Returns the modified dataset and a dictionary mapping original labels to IDs.
    """
    if classifier == "cell":
        label_set = set(data["label"])
        class_id_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        data = data.map(lambda ex: {"label": class_id_dict[ex["label"]]}, num_proc=nproc)
    elif classifier == "gene":
        class_id_dict = {k: idx for idx, k in enumerate(sorted(class_dict.keys()))}
        def map_gene_labels(example):
            example["labels"] = [class_id_dict.get(token, -100) for token in example["input_ids"]]
            return example
        data = data.map(map_gene_labels, num_proc=nproc)
    return data, class_id_dict


import torch
from transformers import DataCollatorWithPadding

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

class DataCollatorForGeneClassification(DataCollatorWithPadding):
    """
    A data collator for gene classification that pads input_ids and collates sequence labels.
    """
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        input_ids = [f["input_ids"] for f in features]
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return batch
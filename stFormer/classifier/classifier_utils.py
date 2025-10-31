#!/usr/bin/env python3
"""
classifier_utils.py

Utility functions for data preprocessing for stFormer classification.
"""

import logging
import random
from collections import defaultdict, Counter

import pickle
from datasets import load_from_disk


logger = logging.getLogger(__name__)

def load_and_filter(filter_data, nproc, input_data):
    """
    Load a dataset and apply filtering criteria.
    """
    if isinstance(input_data,str):
        data = load_from_disk(input_data)
    else: #already HF dataset
        data = input_data
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

def label_classes(classifier, data, class_dict, token_dict_path, nproc):
    """
    Map class labels to numeric IDs.
    - For cell classifiers, uses the `label` column.
    - For gene classifiers, loads the token dictionary so we can
      map each input_ids token back to its ENSEMBL ID, and then
      to your provided gene classes.
    Returns (dataset, class_id_dict).
    """
    if classifier == "cell":
        # unchanged from before
        label_set = set(data["label"])
        class_id_dict = {label: idx for idx, label in enumerate(sorted(label_set))}
        data = data.map(
            lambda ex: {"label": class_id_dict[ex["label"]]},
            num_proc=nproc
        )
        return data, class_id_dict

    elif classifier == "gene":
        # 1) build a class-name → integer map
        class_id_dict = {name: i for i, name in enumerate(sorted(class_dict.keys()))}

        # 2) load your vocab (ENSEMBL string → token ID) and invert it
        with open(token_dict_path, "rb") as f:
            gene2token = pickle.load(f)
        token2gene = {v: k for k, v in gene2token.items()}

        # 3) build token ID → class ID map
        token2class = {}
        for class_name, gene_list in class_dict.items():
            cid = class_id_dict[class_name]
            for gene in gene_list:
                tid = gene2token.get(gene)
                if tid is not None:
                    token2class[tid] = cid

        def map_gene_labels_batch(batch):
            # batch["input_ids"] is a list of lists
            all_labels = []
            for seq in batch["input_ids"]:
                # map each token to its class (or -100)
                labels = [ token2class.get(int(t), -100) for t in seq ]
                all_labels.append(labels)
            batch["labels"] = all_labels
            return batch


        data = data.map(
            map_gene_labels_batch, 
            num_proc=nproc,
            batched=True,
            batch_size=1000
            )

        data = data.filter(
            lambda ex: any(l != -100 for l in ex["labels"]),
            num_proc=nproc
        )

        n_remaining = len(data)
        if n_remaining == 0 :
            raise ValueError(
                'No gene-class examples remain after filtering; '
                'review class_dict or token dictionary for this discrepency'
            )
        return data, class_id_dict

    else:
        raise ValueError(f"Unknown classifier type: {classifier}")

def predict_from_checkpoint(
    model_dir: str,
    dataset_path: str,
    classifier_type: str = "sequence",
    batch_size: int = 32,
    num_workers: int = 4,
    return_logits: bool = False
):
    """
    Load a trained model from `model_dir` and run predictions on a dataset saved at `dataset_path`.

    Args:
        model_dir (str): Path to the saved model directory.
        dataset_path (str): Path to the Hugging Face dataset directory.
        classifier_type (str): 'sequence' or 'token'.
        batch_size (int): Batch size for prediction.
        num_workers (int): Number of workers for dataloader.
        return_logits (bool): If True, return raw logits as well.

    Returns:
        Tuple[List[int], Optional[np.ndarray]]: Predicted class indices, optionally logits.
    """
    from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
    )
    from datasets import load_from_disk
    import os
   
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except:
        raise ValueError(f'Tokenizer not found in model dir: {model_dir}, try specifying tokenizer path')

    dataset = load_from_disk(dataset_path)

    if classifier_type == 'sequence':
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        collator = DataCollatorForCellClassification(tokenizer, padding='max_length', max_length=tokenizer.model_max_length)
        columns = ['input_ids', 'label']
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        collator = DataCollatorForGeneClassification(tokenizer, padding='max_length', max_length=tokenizer.model_max_length)
        columns = ['input_ids', 'labels']

    dataset.set_format(type='torch', columns=columns)

    dummy_args = TrainingArguments(
        output_dir=os.path.join(model_dir, 'tmp_pred'),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        do_train=False,
        do_eval=False,
        logging_dir=os.path.join(model_dir, 'logs')
    )

    trainer = Trainer(
        model=model,
        args=dummy_args,
        data_collator=collator
    )
    
    predictions = trainer.predict(dataset)
    predicted_classes = predictions.predictions.argmax(-1)

    if return_logits:
        return predicted_classes.tolist(), predictions.predictions
    return predicted_classes.tolist()

import torch
from transformers import DataCollatorWithPadding

class DataCollatorForCellClassification(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding='max_length', max_length=2048):
        super().__init__(tokenizer=tokenizer, padding=padding, max_length=max_length)

    def __call__(self, features):
        labels = []
        for i, f in enumerate(features):
            lab = f.pop("label")
            labels.append(lab.item())
        batch = super().__call__(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

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

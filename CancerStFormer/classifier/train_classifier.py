"""
classifier.py

A comprehensive pipeline for classification tasks using Geneformer tokenized datasets.
This file implements all necessary functionality for data loading, label mapping,
dataset splitting, model instantiation (with optional freezing), training, evaluation,
and saving of classifiers.

The file provides two main functions:
  - train_cell_classifier: For cell state classification.
  - train_stage_classifier: For stage classification.

Usage Example:
    from classifier_pipeline import train_cell_classifier, train_stage_classifier

    # For cell state classification:
    trainer_cell = train_cell_classifier(
        dataset_path="/path/to/cell_classifier.dataset",
        output_dir="/path/to/cell_classifier_output",
        model_checkpoint="/path/to/pretrained_checkpoint",
        target_names=["African-American", "Non-Hispanic White"],
        label_column="cell_state",  # column containing cell state labels
        num_epochs=3,
        batch_size=12,
        freeze_layers=2,
        seed=42
    )

    # For stage classification:
    trainer_stage = train_stage_classifier(
        dataset_path="/path/to/stage_classifier.dataset",
        output_dir="/path/to/stage_classifier_output",
        model_checkpoint="/path/to/pretrained_checkpoint",
        target_names=["IIA", "IIB", "III"],
        label_column="stage",  # column containing stage labels
        num_epochs=3,
        batch_size=12,
        freeze_layers=0,
        seed=42
    )
"""


import os
import random
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk, DatasetDict
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score


def label_examples(example, label_map, label_column):
    """
    Map a string label to a numeric label using the provided label_map.

    Parameters:
      example (dict): A single example from the dataset.
      label_map (dict): A dictionary mapping string labels to integers.
      label_column (str): Column name in the example containing the label.

    Returns:
      dict: Modified example with a new key "label".
    """
    example["label"] = label_map[example[label_column]]
    return example


def compute_metrics(pred):
    """
    Compute accuracy for Trainer predictions.

    Parameters:
      pred: Predictions output from the Trainer.

    Returns:
      dict: A dictionary containing the accuracy.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def split_dataset(dataset, seed=42):
    """
    Split a dataset into train (80%), validation (10%), and test (10%) sets.

    Parameters:
      dataset: A Hugging Face Dataset.
      seed (int): Random seed.

    Returns:
      DatasetDict: A DatasetDict with keys "train", "validation", and "test".
    """
    ds = dataset.train_test_split(test_size=0.2, seed=seed)
    test_val = ds["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({"train": ds["train"], "validation": test_val["train"], "test": test_val["test"]})


def freeze_model_layers(model, num_layers_to_freeze):
    """
    Freeze the first num_layers_to_freeze transformer layers in the model.
    Assumes model.bert.encoder.layer exists.

    Parameters:
      model: A transformers model instance (e.g., BertForSequenceClassification).
      num_layers_to_freeze (int): Number of layers to freeze.

    Returns:
      model: Modified model with frozen layers.
    """
    if num_layers_to_freeze > 0 and hasattr(model, "bert"):
        for layer in model.bert.encoder.layer[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
    return model


def train_cell_classifier(
    dataset_path: str,
    output_dir: str,
    model_checkpoint: str,
    target_names: list,
    label_column: str = "cell_state",
    num_epochs: int = 3,
    batch_size: int = 12,
    freeze_layers: int = 0,
    seed: int = 42
):
    """
    Train a cell state classifier using a pretrained BERT model.

    Parameters:
      dataset_path (str): Path to the cell classifier dataset (in .dataset format).
      output_dir (str): Directory where training outputs and model checkpoints will be saved.
      model_checkpoint (str): Path to a pretrained model checkpoint.
      target_names (list): List of possible cell state class labels (strings).
      label_column (str): Column name that contains cell state labels in the dataset.
      num_epochs (int): Number of training epochs.
      batch_size (int): Batch size per device.
      freeze_layers (int): Number of initial transformer layers to freeze.
      seed (int): Random seed for reproducibility.

    Returns:
      Trainer: The Hugging Face Trainer object after training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the dataset.
    dataset = load_from_disk(dataset_path)
    # Create mapping from string label to int.
    label_map = {name: i for i, name in enumerate(target_names)}
    # Map dataset to create a "label" field.
    dataset = dataset.map(lambda ex: label_examples(ex, label_map, label_column))
    # Split dataset into train, validation, and test sets.
    split_ds = split_dataset(dataset, seed=seed)

    # Load the pretrained model.
    model = BertForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(target_names)
    )
    # Freeze the specified number of layers.
    model = freeze_model_layers(model, freeze_layers)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=100,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model.
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
    return trainer


def train_stage_classifier(
    dataset_path: str,
    output_dir: str,
    model_checkpoint: str,
    target_names: list,
    label_column: str = "stage",
    num_epochs: int = 3,
    batch_size: int = 12,
    freeze_layers: int = 0,
    seed: int = 42
):
    """
    Train a stage classifier (e.g., developmental or disease stages) using a pretrained BERT model.

    Parameters:
      dataset_path (str): Path to the stage classifier dataset (in .dataset format).
      output_dir (str): Directory where training outputs and model checkpoints will be saved.
      model_checkpoint (str): Path to a pretrained model checkpoint.
      target_names (list): List of possible stage labels.
      label_column (str): Column name that contains stage labels in the dataset.
      num_epochs (int): Number of training epochs.
      batch_size (int): Batch size per device.
      freeze_layers (int): Number of initial transformer layers to freeze.
      seed (int): Random seed for reproducibility.

    Returns:
      Trainer: The Hugging Face Trainer object after training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the dataset.
    dataset = load_from_disk(dataset_path)
    # Create mapping from stage label to numeric label.
    label_map = {name: i for i, name in enumerate(target_names)}
    # Map the dataset to add a "label" field.
    dataset = dataset.map(lambda ex: label_examples(ex, label_map, label_column))
    # Split dataset into train, validation, and test sets.
    split_ds = split_dataset(dataset, seed=seed)

    # Load the pretrained model.
    model = BertForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(target_names)
    )
    model = freeze_model_layers(model, freeze_layers)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_steps=100,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Create the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["validation"],
        compute_metrics=compute_metrics,
    )

    # Train the model.
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
    return trainer
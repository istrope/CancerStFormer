#!/usr/bin/env python3
"""
evaluation_utils.py

Utility functions for evaluating Geneformer classifiers.

Provides methods to compute metrics, plot confusion matrices, and visualize predictions.
"""

import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

logger = logging.getLogger(__name__)

def compute_metrics(pred):
    """
    Compute accuracy from Trainer predictions.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def evaluate_model(trainer, eval_dataset):
    """
    Evaluate model on eval_dataset using the trainer.
    """
    metrics = trainer.evaluate(eval_dataset)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics

def plot_confusion_matrix(conf_mat_dict, output_directory, output_prefix, custom_class_order):
    """
    Plot and save a confusion matrix.
    """
    #cm = conf_mat_dict.get("Geneformer")
    if cm is None:
        logger.error("Confusion matrix not found.")
        return
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=custom_class_order, yticklabels=custom_class_order, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    output_path = f"{output_directory}/{output_prefix}_confusion_matrix.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def plot_predictions(predictions_file, id_class_dict_file, title, output_directory, output_prefix, custom_class_order):
    """
    Plot and save prediction results as a confusion matrix.
    """
    with open(predictions_file, "rb") as f:
        predictions = pickle.load(f)
    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)
    y_true = predictions.get("y_true")
    y_pred = predictions.get("y_pred")
    cm = confusion_matrix(y_true, y_pred, labels=list(id_class_dict.values()))
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=custom_class_order, yticklabels=custom_class_order, cmap="Oranges")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    output_path = f"{output_directory}/{output_prefix}_predictions.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

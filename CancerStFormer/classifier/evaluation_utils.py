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
import torch
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import seaborn as sns

logger = logging.getLogger(__name__)

def py_softmax(vector: np.ndarray) -> np.ndarray:
    """
    Stable softmax: subtract max before exponentiation.
    """
    v = vector - np.max(vector)
    e = np.exp(v)
    return e / np.sum(e)


def compute_metrics(pred):
    """
    Compute accuracy, precision, recall, F1 (macro), and ROC-AUC (binary)
    Masks out any label == -100 (padding) and handles empty cases by returning 0.0.
    """
    labels = pred.label_ids
    logits = pred.predictions if hasattr(pred, 'predictions') else None

    # get raw preds
    preds = np.argmax(logits, axis=-1) if logits is not None else None

    # flatten for token classification
    if preds is not None and preds.ndim > 1:
        labels_flat = labels.flatten()
        preds_flat = preds.flatten()
        mask = labels_flat != -100
        labels_flat = labels_flat[mask]
        preds_flat = preds_flat[mask]
        logits_flat = logits.reshape(-1, logits.shape[-1])[mask]
    else:
        labels_flat = labels
        preds_flat = preds
        logits_flat = logits

    # guard empty
    if labels_flat is None or len(labels_flat) == 0:
        logger.warning("No valid labels found in compute_metrics; returning zeros")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    acc = accuracy_score(labels_flat, preds_flat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_flat, preds_flat, average="macro", zero_division=0
    )

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # binary roc_auc
    if logits_flat is not None and len(np.unique(labels_flat)) == 2:
        probs = np.array([py_softmax(log)[1] for log in logits_flat])
        fpr, tpr, _ = roc_curve(labels_flat, probs)
        metrics["roc_auc"] = float(auc(fpr, tpr))

    return metrics


def evaluate_model(trainer, eval_dataset, label_names=None):
    """
    Evaluate model: combine Trainer.evaluate with detailed classification report
    and confusion matrix. Sanitizes NaN/Inf in base metrics.
    """
    # base metrics includes eval_loss
    base_metrics = trainer.evaluate(eval_dataset)
    # sanitize any NaN/Inf
    for k, v in list(base_metrics.items()):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            logger.warning(f"Metric '{k}' is {v}, setting to 0.0")
            base_metrics[k] = 0.0

    # detailed preds
    pred_out = trainer.predict(eval_dataset)
    cls_metrics = compute_metrics(pred_out)

    # classification report & confusion matrix
    labels = pred_out.label_ids.flatten()
    preds = pred_out.predictions.argmax(-1).flatten()
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    report = classification_report(
        labels, preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(labels, preds)

    # merge all
    base_metrics.update(cls_metrics)
    base_metrics["classification_report"] = report
    base_metrics["confusion_matrix"] = cm.tolist()
    return base_metrics

def plot_confusion_matrix(conf_mat_dict, output_directory, output_prefix, custom_class_order):
    """
    Plot and save a confusion matrix.
    """
    #cm = conf_mat_dict.get("Geneformer")
    if conf_mat_dict is None:
        logger.error("Confusion matrix not found.")
        return
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_mat_dict, annot=True, fmt="d", xticklabels=custom_class_order, yticklabels=custom_class_order, cmap="Blues")
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

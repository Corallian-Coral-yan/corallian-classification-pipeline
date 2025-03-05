import logging

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, F1, precision, and recall metrics
    :param y_true: ground truth labels
    :param y_pred: predicted labels
    :return: dictionary of metrics
    """
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        metrics["f1_score"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception as e:
        logging.warning(f"Warning: F1-score computation failed due to {e}")
        metrics["f1_score"] = 0.0  # Ensure it always exists

    return metrics
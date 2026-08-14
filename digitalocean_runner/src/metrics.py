"""Binary metrics with malignant as the positive class."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)

METRIC_NAMES = ["malignant_recall", "malignant_precision", "pr_auc", "macro_f1", "balanced_accuracy", "mcc", "specificity", "roc_auc", "accuracy", "weighted_f1", "npv", "fnr", "fpr", "brier_score", "log_loss"]

def calculate_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    return {
        "malignant_recall": recall_score(y, pred, zero_division=0), "malignant_precision": precision_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, p), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "mcc": matthews_corrcoef(y, pred), "specificity": div(tn, tn + fp),
        "roc_auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, pred), "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "npv": div(tn, tn + fn), "fnr": div(fn, fn + tp), "fpr": div(fp, fp + tn), "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

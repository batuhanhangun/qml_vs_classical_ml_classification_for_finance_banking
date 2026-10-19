"""Evaluation metrics and threshold selection.

Sample weights let metrics computed on an undersampled set estimate their value at the natural
class rate: negatives are weighted by the inverse of the fraction of negatives kept.
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

FPR_TARGETS = (0.001, 0.01)


def select_threshold(y, score, sample_weight=None):
    """Threshold that maximises minority-class F1 on (y, score)."""
    precision, recall, thresholds = precision_recall_curve(y, score, sample_weight=sample_weight)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return thresholds[np.argmax(f1[:-1])]


def recall_at_fpr(y, score, target):
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= target].max())


def evaluate(y, score, threshold):
    y = np.asarray(y)
    pred = (score >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    recall = tp / max(np.sum(y == 1), 1)
    specificity = tn / max(np.sum(y == 0), 1)
    out = {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": tp / max(pred.sum(), 1),
        "recall": recall,
        "specificity": specificity,
        "g_mean": float(np.sqrt(recall * specificity)),
    }
    for t in FPR_TARGETS:
        out[f"recall_at_fpr_{t:g}"] = recall_at_fpr(y, score, t)
    return out

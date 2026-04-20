from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from app.ml.data_loader import quarter_to_period


def generate_walk_forward_splits(
    quarters: Iterable[str],
    train_window: int = 12,
    validation_window: int = 2,
    step: int = 2,
) -> list[tuple[list[str], list[str]]]:
    ordered = sorted(set(quarters), key=quarter_to_period)
    splits: list[tuple[list[str], list[str]]] = []
    start = 0
    while start + train_window + validation_window <= len(ordered):
        train_quarters = ordered[start : start + train_window]
        validation_quarters = ordered[start + train_window : start + train_window + validation_window]
        splits.append((train_quarters, validation_quarters))
        start += step
    return splits


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.45) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, probabilities))
        fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
        roc_payload = {
            "fpr": fpr.round(6).tolist(),
            "tpr": tpr.round(6).tolist(),
            "thresholds": roc_thresholds.round(6).tolist(),
        }
    else:
        auc = float("nan")
        roc_payload = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / max(tn + fp, 1))
    npv = float(tn / max(tn + fn, 1))
    ppv = float(tp / max(tp + fp, 1))

    return {
        "auc_roc": auc,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "derived_metrics": {
            "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
        },
        "roc_curve": roc_payload,
    }


def aggregate_fold_metrics(folds: list[dict]) -> dict:
    metric_names = ["auc_roc", "precision", "recall", "f1", "brier_score"]
    aggregate = {}
    for metric in metric_names:
        values = np.array([fold[metric] for fold in folds], dtype=float)
        aggregate[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
    return aggregate

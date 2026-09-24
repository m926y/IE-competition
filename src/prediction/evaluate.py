"""Metrics and operating-threshold selection."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true, probability) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    result: dict[str, float | int] = {
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "probability_min": float(p.min()),
        "probability_max": float(p.max()),
        "probability_mean": float(p.mean()),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    return result


def aggregate_fold_metrics(records: list[dict]) -> dict[str, float]:
    aucs = np.asarray([record["roc_auc"] for record in records], dtype=float)
    prs = np.asarray([record["pr_auc"] for record in records], dtype=float)
    losses = np.asarray([record["log_loss"] for record in records], dtype=float)
    return {
        "mean_roc_auc": float(np.nanmean(aucs)),
        "min_roc_auc": float(np.nanmin(aucs)),
        "mean_pr_auc": float(np.nanmean(prs)),
        "mean_log_loss": float(np.nanmean(losses)),
    }


def metric_sort_key(summary: dict[str, float]) -> tuple[float, float, float]:
    def finite(value: float, default: float) -> float:
        return value if math.isfinite(value) else default

    return (
        finite(summary["mean_roc_auc"], -1.0),
        finite(summary["min_roc_auc"], -1.0),
        -finite(summary["mean_log_loss"], 1e9),
    )


def choose_operating_threshold(
    y_true,
    probability,
    *,
    minimum_recall: float = 0.70,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(np.unique(y)) < 2:
        return {"threshold": 0.5, "recall": float(y.mean()), "specificity": 0.0}
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.flatnonzero(tpr >= minimum_recall)
    if len(valid):
        index = valid[np.argmax((tpr - fpr)[valid])]
    else:
        index = int(np.argmax(tpr - fpr))
    threshold = float(np.clip(thresholds[index], 0.0, 1.0))
    return {
        "threshold": threshold,
        "recall": float(tpr[index]),
        "specificity": float(1 - fpr[index]),
    }

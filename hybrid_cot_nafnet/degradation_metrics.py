"""Dependency-free multi-label metrics for degradation reasoning."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    positives = score[target]
    negatives = score[~target]
    if not len(positives) or not len(negatives):
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    positive_count = int(target.sum())
    if not positive_count:
        return float("nan")
    thresholds = np.unique(score)[::-1]
    previous_recall = 0.0
    average_precision = 0.0
    for threshold in thresholds:
        predicted = score >= threshold
        true_positive = int(np.count_nonzero(predicted & target))
        false_positive = int(np.count_nonzero(predicted & ~target))
        recall = true_positive / positive_count
        precision = true_positive / max(1, true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def multilabel_degradation_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Calculate threshold and ranking metrics for an ``[N, L]`` label matrix."""
    targets = np.asarray(targets, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError(
            "targets and probabilities must have the same two-dimensional shape"
        )
    if targets.shape[1] != len(label_names):
        raise ValueError("label_names must match the label dimension")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")

    predicted = probabilities >= threshold
    per_label: Dict[str, Dict[str, float | int]] = {}
    total_true_positive = total_false_positive = total_false_negative = 0
    for index, name in enumerate(label_names):
        target = targets[:, index]
        prediction = predicted[:, index]
        true_positive = int(np.count_nonzero(prediction & target))
        false_positive = int(np.count_nonzero(prediction & ~target))
        false_negative = int(np.count_nonzero(~prediction & target))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        per_label[str(name)] = {
            "positive_support": int(target.sum()),
            "negative_support": int((~target).sum()),
            "predicted_positive": int(prediction.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auroc": _binary_auroc(target, probabilities[:, index]),
            "average_precision": _binary_average_precision(
                target, probabilities[:, index]
            ),
        }
        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative

    micro_denominator = (
        2 * total_true_positive + total_false_positive + total_false_negative
    )
    return {
        "threshold": float(threshold),
        "micro_f1": (
            2 * total_true_positive / micro_denominator
            if micro_denominator
            else 0.0
        ),
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "exact_match": float(np.mean(np.all(predicted == targets, axis=1))),
        "per_label": per_label,
    }


def flattened_degradation_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    """Flatten metrics for CSV logging and DDP scalar broadcasting."""
    flattened = {
        "degradation_micro_f1": float(metrics["micro_f1"]),
        "degradation_macro_f1": float(metrics["macro_f1"]),
        "degradation_exact_match": float(metrics["exact_match"]),
    }
    per_label = metrics["per_label"]
    if not isinstance(per_label, dict):
        raise TypeError("per_label metrics must be a dictionary")
    for label, values in per_label.items():
        if not isinstance(values, dict):
            raise TypeError("each per-label metric must be a dictionary")
        for metric in ("precision", "recall", "f1", "auroc", "average_precision"):
            flattened[f"degradation_{label}_{metric}"] = float(values[metric])
    return flattened

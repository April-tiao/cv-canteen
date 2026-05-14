from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


CLASS_NAMES = ("food", "chart", "other")


@dataclass
class Thresholds:
    food: float = 0.75
    chart: float = 0.85


def targets_to_class(targets: torch.Tensor | np.ndarray) -> np.ndarray:
    array = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    labels = np.full(array.shape[0], 2, dtype=np.int64)
    labels[array[:, 0] >= 0.5] = 0
    labels[array[:, 1] >= 0.5] = 1
    return labels


def logits_to_predictions(logits: torch.Tensor | np.ndarray, thresholds: Thresholds) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(logits)
    scores = torch.sigmoid(tensor).detach().cpu().numpy()
    food_scores = scores[:, 0]
    chart_scores = scores[:, 1]
    preds = np.full(scores.shape[0], 2, dtype=np.int64)
    food_mask = (food_scores >= thresholds.food) & (food_scores >= chart_scores)
    chart_mask = (chart_scores >= thresholds.chart) & (chart_scores > food_scores)
    preds[food_mask] = 0
    preds[chart_mask] = 1
    return preds, scores


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def precision_recall_f1(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        result[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return result


def compute_metrics(logits: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray, thresholds: Thresholds) -> dict:
    y_true = targets_to_class(targets)
    y_pred, scores = logits_to_predictions(logits, thresholds)
    matrix = confusion_matrix(y_true, y_pred)
    per_class = precision_recall_f1(matrix)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(np.mean([per_class[name]["f1"] for name in CLASS_NAMES]))

    other_total = max(1, matrix[2, :].sum())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "food_precision": per_class["food"]["precision"],
        "food_recall": per_class["food"]["recall"],
        "chart_precision": per_class["chart"]["precision"],
        "chart_recall": per_class["chart"]["recall"],
        "other_recall": per_class["other"]["recall"],
        "other_to_food_fpr": float(matrix[2, 0] / other_total),
        "other_to_chart_fpr": float(matrix[2, 1] / other_total),
        "confusion_matrix": matrix.tolist(),
        "thresholds": {"food": thresholds.food, "chart": thresholds.chart},
        "score_means": {
            "food_score": float(scores[:, 0].mean()) if len(scores) else 0.0,
            "chart_score": float(scores[:, 1].mean()) if len(scores) else 0.0,
        },
    }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"accuracy: {metrics['accuracy']:.4f}",
        f"macro_f1: {metrics['macro_f1']:.4f}",
        f"food precision/recall: {metrics['food_precision']:.4f}/{metrics['food_recall']:.4f}",
        f"chart precision/recall: {metrics['chart_precision']:.4f}/{metrics['chart_recall']:.4f}",
        f"other recall: {metrics['other_recall']:.4f}",
        f"other -> food FPR: {metrics['other_to_food_fpr']:.4f}",
        f"other -> chart FPR: {metrics['other_to_chart_fpr']:.4f}",
        "confusion_matrix rows=true cols=pred [food, chart, other]:",
        str(metrics["confusion_matrix"]),
    ]
    return "\n".join(lines)

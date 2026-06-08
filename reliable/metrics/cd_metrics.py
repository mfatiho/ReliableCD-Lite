from __future__ import annotations

import numpy as np


def binary_confusion_counts(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred) > 0
    gt = np.asarray(gt) > 0
    tp = float(np.logical_and(pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_cd_metrics(pred: np.ndarray, gt: np.ndarray, empty_policy: str = "zero") -> dict[str, float]:
    """Return F1, IoU, precision, recall, OA."""
    if empty_policy not in {"zero", "one", "nan"}:
        raise ValueError("empty_policy must be one of: zero, one, nan")
    counts = binary_confusion_counts(pred, gt)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    if tp + fp + fn == 0:
        neutral = 0.0 if empty_policy == "zero" else 1.0 if empty_policy == "one" else float("nan")
        oa = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
        return {
            "f1": neutral,
            "iou": neutral,
            "precision": neutral,
            "recall": neutral,
            "oa": oa,
        }
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall, "oa": oa}


def binary_dataset_metrics(preds: np.ndarray, gts: np.ndarray) -> dict[str, float]:
    counts = binary_confusion_counts(preds, gts)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    oa = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "oa": oa,
    }

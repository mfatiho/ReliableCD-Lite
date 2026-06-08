from __future__ import annotations

import numpy as np


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def best_iou(component_mask: np.ndarray, gt_labeled: np.ndarray) -> float:
    component_mask = np.asarray(component_mask).astype(bool)
    best = 0.0
    for gt_id in np.unique(gt_labeled):
        if gt_id == 0:
            continue
        gt_component = gt_labeled == gt_id
        inter = np.logical_and(component_mask, gt_component).sum()
        union = np.logical_or(component_mask, gt_component).sum()
        if union == 0:
            continue
        best = max(best, float(inter / union))
    return best

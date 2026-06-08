from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.components.matching import best_iou, mask_bbox
from reliable.components.extraction import extract_components


def label_components_against_gt(
    labeled_pred: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5,
    error_pixel_ratio_threshold: float = 0.5,
) -> pd.DataFrame:
    """Label predicted components as erroneous or reliable by oracle GT matching."""
    gt_labeled, _ = extract_components(gt_mask, connectivity=8, min_area=1)
    gt_binary = np.asarray(gt_mask) > 0
    rows: list[dict] = []
    for component_id in sorted(np.unique(labeled_pred)):
        if component_id == 0:
            continue
        component_mask = labeled_pred == component_id
        area = int(component_mask.sum())
        overlap = np.logical_and(component_mask, gt_binary).sum()
        error_pixel_ratio = 1.0 - (float(overlap) / area if area else 0.0)
        best_gt = best_iou(component_mask, gt_labeled)
        is_error = (best_gt < iou_threshold) or (error_pixel_ratio > error_pixel_ratio_threshold)
        bbox_x, bbox_y, bbox_w, bbox_h = mask_bbox(component_mask)
        rows.append(
            {
                "component_id": int(component_id),
                "area": area,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "best_gt_iou": best_gt,
                "error_pixel_ratio": error_pixel_ratio,
                "is_error_component": bool(is_error),
            }
        )
    return pd.DataFrame(rows)

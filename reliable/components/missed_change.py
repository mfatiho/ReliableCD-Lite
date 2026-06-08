from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.components.extraction import extract_components
from reliable.components.matching import mask_bbox


def analyze_missed_gt_components(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    min_gt_area: int = 16,
    overlap_threshold: float = 0.1,
) -> pd.DataFrame:
    """Identify GT change components that are mostly missed by prediction."""
    gt_labeled, _ = extract_components(gt_mask, connectivity=8, min_area=min_gt_area)
    pred_binary = np.asarray(pred_mask) > 0
    rows: list[dict] = []
    for gt_component_id in sorted(np.unique(gt_labeled)):
        if gt_component_id == 0:
            continue
        component_mask = gt_labeled == gt_component_id
        gt_area = int(component_mask.sum())
        pred_overlap_area = int(np.logical_and(component_mask, pred_binary).sum())
        ratio = pred_overlap_area / gt_area if gt_area else 0.0
        bbox_x, bbox_y, bbox_w, bbox_h = mask_bbox(component_mask)
        rows.append(
            {
                "gt_component_id": int(gt_component_id),
                "gt_area": gt_area,
                "pred_overlap_area": pred_overlap_area,
                "pred_overlap_ratio": float(ratio),
                "is_missed": bool(ratio < overlap_threshold),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            }
        )
    return pd.DataFrame(rows)

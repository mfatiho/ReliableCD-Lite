from __future__ import annotations

import math

import numpy as np
import pandas as pd

from reliable.components.boundary import boundary_uncertainty
from reliable.components.matching import mask_bbox
from reliable.components.oracle_labeling import label_components_against_gt


def probability_margin(mean_prob: float) -> float:
    return abs(float(mean_prob) - 0.5)


def compactness(component_mask: np.ndarray) -> float:
    area = float(component_mask.sum())
    if area == 0:
        return 0.0
    perimeter = float(np.logical_xor(component_mask, np.pad(component_mask[1:, :], ((0, 1), (0, 0)), constant_values=False)).sum())
    perimeter += float(np.logical_xor(component_mask, np.pad(component_mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)).sum())
    if perimeter == 0:
        return 1.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def eccentricity(component_mask: np.ndarray) -> float:
    ys, xs = np.where(component_mask)
    if len(xs) < 2:
        return 0.0
    coords = np.column_stack([xs, ys]).astype(float)
    cov = np.cov(coords.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    major = float(np.sqrt(max(eigenvalues[-1], 0.0)))
    minor = float(np.sqrt(max(eigenvalues[0], 0.0)))
    if major == 0:
        return 0.0
    ratio = minor / major
    return float(np.sqrt(max(0.0, 1.0 - ratio * ratio)))


def compute_component_features(
    labeled_pred: np.ndarray,
    prob_map: np.ndarray,
    entropy_map: np.ndarray | None = None,
    mi_map: np.ndarray | None = None,
    tta_map: np.ndarray | None = None,
    shift_map: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    image_id: str = "unknown",
    dataset: str = "unknown",
    model_name: str = "unknown",
) -> pd.DataFrame:
    feature_columns = [
        "model_name",
        "dataset",
        "image_id",
        "component_id",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "mean_prob",
        "max_prob",
        "mean_entropy",
        "mean_mi",
        "max_mi",
        "mean_tta",
        "mean_shift",
        "boundary_uncertainty",
        "probability_margin",
        "compactness",
        "eccentricity",
        "is_error_component",
        "best_gt_iou",
        "error_pixel_ratio",
    ]
    rows: list[dict] = []
    prob_map = np.asarray(prob_map, dtype=float)
    entropy_map = np.asarray(entropy_map if entropy_map is not None else np.zeros_like(prob_map), dtype=float)
    mi_map_array = None if mi_map is None else np.asarray(mi_map, dtype=float)
    tta_map = np.asarray(tta_map if tta_map is not None else np.zeros_like(prob_map), dtype=float)
    shift_map = np.asarray(shift_map if shift_map is not None else np.zeros_like(prob_map), dtype=float)
    oracle_df = None if gt_mask is None else label_components_against_gt(labeled_pred, gt_mask)
    oracle_rows = (
        {}
        if oracle_df is None or oracle_df.empty or "component_id" not in oracle_df.columns
        else oracle_df.set_index("component_id").to_dict(orient="index")
    )

    for component_id in sorted(np.unique(labeled_pred)):
        if component_id == 0:
            continue
        component_mask = labeled_pred == component_id
        area = int(component_mask.sum())
        bbox_x, bbox_y, bbox_w, bbox_h = mask_bbox(component_mask)
        mean_prob = float(prob_map[component_mask].mean())
        row = {
            "model_name": model_name,
            "dataset": dataset,
            "image_id": image_id,
            "component_id": int(component_id),
            "area": area,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "mean_prob": mean_prob,
            "max_prob": float(prob_map[component_mask].max()),
            "mean_entropy": float(entropy_map[component_mask].mean()),
            "mean_mi": float(mi_map_array[component_mask].mean()) if mi_map_array is not None else np.nan,
            "max_mi": float(mi_map_array[component_mask].max()) if mi_map_array is not None else np.nan,
            "mean_tta": float(tta_map[component_mask].mean()),
            "mean_shift": float(shift_map[component_mask].mean()),
            "boundary_uncertainty": boundary_uncertainty(component_mask, mi_map_array if mi_map_array is not None else entropy_map),
            "probability_margin": probability_margin(mean_prob),
            "compactness": compactness(component_mask),
            "eccentricity": eccentricity(component_mask),
        }
        oracle = oracle_rows.get(int(component_id), {})
        row["is_error_component"] = bool(oracle.get("is_error_component", False))
        row["best_gt_iou"] = float(oracle.get("best_gt_iou", np.nan))
        row["error_pixel_ratio"] = float(oracle.get("error_pixel_ratio", np.nan))
        rows.append(row)
    return pd.DataFrame(rows, columns=feature_columns)

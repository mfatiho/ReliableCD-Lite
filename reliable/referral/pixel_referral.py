from __future__ import annotations

import numpy as np

from reliable.metrics.cd_metrics import binary_cd_metrics
from reliable.referral.oracle_correction import oracle_correct_pixels


def compute_referral_metrics(
    pred_mask: np.ndarray,
    corrected_mask: np.ndarray,
    gt_mask: np.ndarray,
    review_mask: np.ndarray,
) -> dict[str, float]:
    pred_binary = (np.asarray(pred_mask) > 0).astype(np.uint8)
    corrected_binary = (np.asarray(corrected_mask) > 0).astype(np.uint8)
    gt_binary = (np.asarray(gt_mask) > 0).astype(np.uint8)
    review_mask = np.asarray(review_mask).astype(bool)
    error_mask = pred_binary != gt_binary
    reviewed_error = np.logical_and(error_mask, review_mask).sum()
    total_error = error_mask.sum()
    before = binary_cd_metrics(pred_binary, gt_binary)
    after = binary_cd_metrics(corrected_binary, gt_binary)
    return {
        "error_pixel_recall": float(reviewed_error / total_error) if total_error else 0.0,
        "reviewed_area_pct": float(review_mask.mean()),
        "f1_before": before["f1"],
        "f1_after_oracle": after["f1"],
        "f1_gain_upper_bound": after["f1"] - before["f1"],
        "iou_before": before["iou"],
        "iou_after_oracle": after["iou"],
        "iou_gain_upper_bound": after["iou"] - before["iou"],
    }


def pixel_level_referral(
    uncertainty_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    review_budget: float,
) -> dict[str, float | np.ndarray]:
    """Select highest-risk pixels until area budget is reached."""
    uncertainty_map = np.asarray(uncertainty_map, dtype=float)
    total_pixels = uncertainty_map.size
    k = max(1, int(total_pixels * review_budget))
    flat = uncertainty_map.reshape(-1)
    idx = np.argpartition(flat, -k)[-k:]
    review_mask = np.zeros_like(flat, dtype=bool)
    review_mask[idx] = True
    review_mask = review_mask.reshape(uncertainty_map.shape)
    corrected = oracle_correct_pixels(pred_mask, gt_mask, review_mask)
    metrics = compute_referral_metrics(pred_mask, corrected, gt_mask, review_mask)
    metrics["review_mask"] = review_mask
    return metrics


def dataset_pixel_referral(
    uncertainty_maps: dict[str, np.ndarray],
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    review_budget: float,
    return_per_image: bool = False,
) -> dict[str, float] | tuple[dict[str, float], "pd.DataFrame"]:
    """Apply pixel-level referral independently per image and aggregate metrics."""
    metric_names = (
        "error_pixel_recall",
        "reviewed_area_pct",
        "f1_before",
        "f1_after_oracle",
        "f1_gain_upper_bound",
        "iou_before",
        "iou_after_oracle",
        "iou_gain_upper_bound",
    )
    per_image_rows: list[dict[str, float | str]] = []
    for image_id, uncertainty_map in uncertainty_maps.items():
        out = pixel_level_referral(
            uncertainty_map=uncertainty_map,
            pred_mask=pred_masks[image_id],
            gt_mask=gt_masks[image_id],
            review_budget=review_budget,
        )
        total_error_pixels = float((pred_masks[image_id] != gt_masks[image_id]).sum())
        reviewed_error_pixels = float(out["error_pixel_recall"]) * total_error_pixels
        total_pixels = float(out["review_mask"].size)
        reviewed_pixels = float(out["reviewed_area_pct"]) * total_pixels
        per_image_rows.append(
            {
                "image_id": str(image_id),
                **{metric_name: float(out[metric_name]) for metric_name in metric_names},
                "reviewed_error_pixels": reviewed_error_pixels,
                "total_error_pixels": total_error_pixels,
                "reviewed_pixels": reviewed_pixels,
                "total_pixels": total_pixels,
            }
        )
    if not per_image_rows:
        summary = {metric_name: 0.0 for metric_name in metric_names}
        summary["error_pixel_recall_pooled"] = 0.0
        summary["reviewed_area_pct_pooled"] = 0.0
        if return_per_image:
            import pandas as pd

            return summary, pd.DataFrame(columns=["image_id", *metric_names])
        return summary
    import pandas as pd

    per_image_df = pd.DataFrame(per_image_rows)
    summary = {metric_name: float(per_image_df[metric_name].mean()) for metric_name in metric_names}
    total_error_pixels = float(per_image_df["total_error_pixels"].sum())
    total_pixels = float(per_image_df["total_pixels"].sum())
    summary["error_pixel_recall_pooled"] = (
        float(per_image_df["reviewed_error_pixels"].sum() / total_error_pixels) if total_error_pixels else 0.0
    )
    summary["reviewed_area_pct_pooled"] = (
        float(per_image_df["reviewed_pixels"].sum() / total_pixels) if total_pixels else 0.0
    )
    if return_per_image:
        return summary, per_image_df
    return summary

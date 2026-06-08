from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.metrics.cd_metrics import binary_cd_metrics
from reliable.referral.oracle_correction import oracle_correct_pixels


def select_components_by_area_budget(
    components_df: pd.DataFrame,
    score_col: str,
    image_area_by_id: dict[str, int],
    review_budget: float,
) -> pd.DataFrame:
    """Select components by descending score until cumulative reviewed area reaches area budget."""
    df = components_df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    total_area = int(sum(image_area_by_id.values()))
    max_area = total_area * review_budget
    if total_area == 0 or len(df) == 0 or max_area <= 0:
        selected = df.iloc[0:0].copy()
        reviewed_area = 0
    else:
        cumulative_area = df["area"].astype(int).cumsum().to_numpy()
        selected_count = min(len(df), int(np.searchsorted(cumulative_area, max_area, side="left") + 1))
        reviewed_area = int(cumulative_area[selected_count - 1])
        selected = df.iloc[:selected_count].copy()
    selected.attrs["target_review_budget"] = review_budget
    selected.attrs["reviewed_area_pct"] = (reviewed_area / total_area) if total_area else 0.0
    selected.attrs["budget_overshoot_pct"] = ((reviewed_area / total_area) - review_budget) if total_area else 0.0
    return selected


def compute_dataset_referral_metrics(
    pred_masks: dict[str, np.ndarray],
    corrected_preds: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    review_masks: dict[str, np.ndarray],
    selected_components: pd.DataFrame,
    components_df: pd.DataFrame,
    return_per_image: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    before_metrics = []
    after_metrics = []
    per_image_rows: list[dict[str, float | str]] = []
    total_error_components_by_image = (
        components_df.groupby("image_id")["is_error_component"].apply(lambda s: int(s.astype(bool).sum())).to_dict()
        if len(components_df)
        else {}
    )
    reviewed_error_components_by_image = (
        selected_components.groupby("image_id")["is_error_component"].apply(lambda s: int(s.astype(bool).sum())).to_dict()
        if len(selected_components)
        else {}
    )
    for image_id, pred in pred_masks.items():
        gt = (gt_masks[image_id] > 0).astype(np.uint8)
        corrected = (corrected_preds[image_id] > 0).astype(np.uint8)
        review_mask = np.asarray(review_masks[image_id]).astype(bool)
        error_mask = (pred > 0) != gt
        before = binary_cd_metrics(pred, gt)
        after = binary_cd_metrics(corrected, gt)
        before_metrics.append(before)
        after_metrics.append(after)
        total_error_components_img = int(total_error_components_by_image.get(image_id, 0))
        reviewed_error_components_img = int(reviewed_error_components_by_image.get(image_id, 0))
        per_image_rows.append(
            {
                "image_id": str(image_id),
                "reviewed_error_components": reviewed_error_components_img,
                "total_error_components": total_error_components_img,
                "component_error_recall": (
                    reviewed_error_components_img / total_error_components_img if total_error_components_img else 0.0
                ),
                "reviewed_error_pixels": int(np.logical_and(error_mask, review_mask).sum()),
                "total_error_pixels": int(error_mask.sum()),
                "error_pixel_recall": float(np.logical_and(error_mask, review_mask).sum() / error_mask.sum()) if error_mask.sum() else 0.0,
                "reviewed_pixels": int(review_mask.sum()),
                "total_pixels": int(review_mask.size),
                "reviewed_area_pct": float(review_mask.mean()),
                "f1_before": before["f1"],
                "f1_after_oracle": after["f1"],
                "f1_gain_upper_bound": after["f1"] - before["f1"],
                "iou_before": before["iou"],
                "iou_after_oracle": after["iou"],
                "iou_gain_upper_bound": after["iou"] - before["iou"],
            }
        )

    if not per_image_rows:
        metrics = {
            "error_recall": 0.0,
            "error_pixel_recall": 0.0,
            "reviewed_area_pct": 0.0,
            "f1_before": 0.0,
            "f1_after_oracle": 0.0,
            "f1_gain_upper_bound": 0.0,
            "iou_before": 0.0,
            "iou_after_oracle": 0.0,
            "iou_gain_upper_bound": 0.0,
            "error_recall_pooled": 0.0,
            "error_pixel_recall_pooled": 0.0,
            "reviewed_area_pct_pooled": 0.0,
        }
        if return_per_image:
            return metrics, pd.DataFrame(columns=["image_id"])
        return metrics

    per_image_df = pd.DataFrame(per_image_rows)
    f1_before = float(per_image_df["f1_before"].mean())
    f1_after = float(per_image_df["f1_after_oracle"].mean())
    iou_before = float(per_image_df["iou_before"].mean())
    iou_after = float(per_image_df["iou_after_oracle"].mean())
    total_error_components = float(per_image_df["total_error_components"].sum())
    total_error_pixels = float(per_image_df["total_error_pixels"].sum())
    total_pixels = float(per_image_df["total_pixels"].sum())
    metrics = {
        "error_recall": float(per_image_df["component_error_recall"].mean()),
        "error_pixel_recall": float(per_image_df["error_pixel_recall"].mean()),
        "reviewed_area_pct": float(per_image_df["reviewed_area_pct"].mean()),
        "f1_before": f1_before,
        "f1_after_oracle": f1_after,
        "f1_gain_upper_bound": f1_after - f1_before,
        "iou_before": iou_before,
        "iou_after_oracle": iou_after,
        "iou_gain_upper_bound": iou_after - iou_before,
        "error_recall_pooled": (
            float(per_image_df["reviewed_error_components"].sum() / total_error_components) if total_error_components else 0.0
        ),
        "error_pixel_recall_pooled": (
            float(per_image_df["reviewed_error_pixels"].sum() / total_error_pixels) if total_error_pixels else 0.0
        ),
        "reviewed_area_pct_pooled": (
            float(per_image_df["reviewed_pixels"].sum() / total_pixels) if total_pixels else 0.0
        ),
    }
    if return_per_image:
        return metrics, per_image_df
    return metrics


def component_level_referral(
    components_df: pd.DataFrame,
    score_col: str,
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    labeled_masks: dict[str, np.ndarray],
    image_area_by_id: dict[str, int],
    review_budget: float,
    return_per_image: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    selected = select_components_by_area_budget(
        components_df=components_df,
        score_col=score_col,
        image_area_by_id=image_area_by_id,
        review_budget=review_budget,
    )
    review_masks = {img_id: np.zeros_like(pred, dtype=bool) for img_id, pred in pred_masks.items()}
    if len(selected):
        for image_id, group in selected.groupby("image_id", sort=False):
            component_ids = group["component_id"].astype(int).to_numpy()
            review_masks[image_id] = np.isin(labeled_masks[image_id], component_ids)

    corrected_preds = {
        img_id: oracle_correct_pixels(pred, gt_masks[img_id], review_masks[img_id])
        for img_id, pred in pred_masks.items()
    }
    dataset_metrics = compute_dataset_referral_metrics(
        pred_masks=pred_masks,
        corrected_preds=corrected_preds,
        gt_masks=gt_masks,
        review_masks=review_masks,
        selected_components=selected,
        components_df=components_df,
        return_per_image=return_per_image,
    )
    if return_per_image:
        metrics, per_image_df = dataset_metrics
    else:
        metrics = dataset_metrics
    metrics.update(
        {
            "score_col": score_col,
            "target_review_budget": review_budget,
            "reviewed_area_pct_pooled": selected.attrs["reviewed_area_pct"],
            "budget_overshoot_pct": selected.attrs["budget_overshoot_pct"],
            "reviewed_component_count": len(selected),
        }
    )
    if return_per_image:
        return metrics, per_image_df
    return metrics

from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.referral.component_referral import component_level_referral, select_components_by_area_budget
from reliable.referral.pixel_referral import dataset_pixel_referral, pixel_level_referral
from scripts.run_referral import _build_early_budget_table, _paired_referral_stats
from scripts.run_referral_component_ablation import _build_primary_budget_table


def test_pixel_referral_selected_area_equals_budget() -> None:
    uncertainty = np.arange(100, dtype=float).reshape(10, 10)
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    out = pixel_level_referral(uncertainty, pred, gt, review_budget=0.1)
    assert abs(out["reviewed_area_pct"] - 0.1) < 1e-9


def test_component_referral_uses_cumulative_area_not_component_count() -> None:
    components = pd.DataFrame(
        [
            {"image_id": "img1", "component_id": 1, "area": 40, "score_crs4": 0.9, "is_error_component": True},
            {"image_id": "img1", "component_id": 2, "area": 5, "score_crs4": 0.8, "is_error_component": False},
            {"image_id": "img1", "component_id": 3, "area": 5, "score_crs4": 0.7, "is_error_component": True},
        ]
    )
    selected = select_components_by_area_budget(components, "score_crs4", {"img1": 100}, 0.1)
    assert len(selected) == 1
    assert selected.iloc[0]["component_id"] == 1


def test_component_referral_budget_overshoot_reported() -> None:
    components = pd.DataFrame(
        [{"image_id": "img1", "component_id": 1, "area": 12, "score_crs4": 1.0, "is_error_component": True}]
    )
    pred_masks = {"img1": np.ones((10, 10), dtype=np.uint8)}
    gt_masks = {"img1": np.zeros((10, 10), dtype=np.uint8)}
    labeled_masks = {"img1": np.ones((10, 10), dtype=np.int32)}
    out = component_level_referral(components, "score_crs4", pred_masks, gt_masks, labeled_masks, {"img1": 100}, 0.1)
    assert out["budget_overshoot_pct"] >= 0


def test_component_referral_selects_all_when_budget_exceeds_total_component_area() -> None:
    components = pd.DataFrame(
        [
            {"image_id": "img1", "component_id": 1, "area": 10, "score_crs4": 0.9, "is_error_component": True},
            {"image_id": "img1", "component_id": 2, "area": 5, "score_crs4": 0.8, "is_error_component": False},
        ]
    )
    selected = select_components_by_area_budget(components, "score_crs4", {"img1": 1000}, 0.5)
    assert len(selected) == 2
    assert abs(selected.attrs["reviewed_area_pct"] - 0.015) < 1e-12
    assert selected.attrs["budget_overshoot_pct"] < 0


def test_component_and_pixel_referral_can_return_per_image_details() -> None:
    components = pd.DataFrame(
        [{"image_id": "img1", "component_id": 1, "area": 12, "score_crs4": 1.0, "is_error_component": True}]
    )
    pred_masks = {"img1": np.ones((10, 10), dtype=np.uint8)}
    gt_masks = {"img1": np.zeros((10, 10), dtype=np.uint8)}
    labeled_masks = {"img1": np.ones((10, 10), dtype=np.int32)}
    component_out, component_per_image = component_level_referral(
        components,
        "score_crs4",
        pred_masks,
        gt_masks,
        labeled_masks,
        {"img1": 100},
        0.1,
        return_per_image=True,
    )
    assert component_out["reviewed_component_count"] == 1
    assert component_per_image["image_id"].tolist() == ["img1"]
    assert component_out["error_pixel_recall"] == component_per_image["error_pixel_recall"].mean()
    assert "error_pixel_recall_pooled" in component_out

    pixel_out, pixel_per_image = dataset_pixel_referral(
        uncertainty_maps={"img1": np.ones((10, 10), dtype=float)},
        pred_masks=pred_masks,
        gt_masks=gt_masks,
        review_budget=0.1,
        return_per_image=True,
    )
    assert pixel_out["error_pixel_recall"] == pixel_per_image["error_pixel_recall"].mean()
    assert "error_pixel_recall_pooled" in pixel_out


def test_early_budget_table_keeps_key_budgets_and_diff_columns() -> None:
    wide_df = pd.DataFrame(
        [
            {
                "model_name": "BIT",
                "dataset": "whu-256",
                "dataset_canonical": "WHU-CD",
                "budget": 0.05,
                "pixel_source": "entropy",
                "component_score": "score_crs4",
                "pixel_error_pixel_recall": 0.4,
                "pixel_error_pixel_recall_ci_lo": 0.3,
                "pixel_error_pixel_recall_ci_hi": 0.5,
                "component_error_pixel_recall": 0.5,
                "component_error_pixel_recall_ci_lo": 0.4,
                "component_error_pixel_recall_ci_hi": 0.6,
                "component_error_recall": 0.9,
                "component_error_recall_ci_lo": 0.8,
                "component_error_recall_ci_hi": 1.0,
                "pixel_f1_gain_upper_bound": 0.03,
                "pixel_f1_gain_upper_bound_ci_lo": 0.02,
                "pixel_f1_gain_upper_bound_ci_hi": 0.04,
                "component_f1_gain_upper_bound": 0.02,
                "component_f1_gain_upper_bound_ci_lo": 0.01,
                "component_f1_gain_upper_bound_ci_hi": 0.03,
                "error_pixel_recall_diff_component_minus_pixel": 0.1,
                "error_pixel_recall_diff_ci_lo": 0.05,
                "error_pixel_recall_diff_ci_hi": 0.15,
                "error_pixel_recall_p_value": 0.01,
                "f1_gain_upper_bound_diff_component_minus_pixel": -0.01,
                "f1_gain_upper_bound_diff_ci_lo": -0.02,
                "f1_gain_upper_bound_diff_ci_hi": 0.0,
                "f1_gain_upper_bound_p_value": 0.04,
            }
        ]
    )
    out = _build_early_budget_table(wide_df)
    assert out["budget"].tolist() == [0.05]
    assert "error_pixel_recall_diff_component_minus_pixel" in out.columns


def test_paired_referral_stats_reports_mean_diff() -> None:
    component_per_image = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "error_pixel_recall": [0.4, 0.5, 0.6],
            "f1_gain_upper_bound": [0.02, 0.03, 0.04],
            "iou_gain_upper_bound": [0.01, 0.02, 0.03],
        }
    )
    pixel_per_image = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "error_pixel_recall": [0.3, 0.4, 0.5],
            "f1_gain_upper_bound": [0.03, 0.04, 0.05],
            "iou_gain_upper_bound": [0.02, 0.03, 0.04],
        }
    )
    out = _paired_referral_stats(component_per_image, pixel_per_image)
    assert out["error_pixel_recall_diff_component_minus_pixel"] > 0
    assert out["f1_gain_upper_bound_diff_component_minus_pixel"] < 0


def test_component_referral_ablation_primary_budget_table_filters_to_5_and_10() -> None:
    df = pd.DataFrame(
        [
            {"budget": 0.01, "component_score": "score_crs4"},
            {"budget": 0.05, "component_score": "score_entropy"},
            {"budget": 0.10, "component_score": "score_margin"},
        ]
    )
    out = _build_primary_budget_table(df)
    assert out["budget"].tolist() == [0.05, 0.10]

from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.visualization.reports import summarize_components_df, summarize_missed_change
from reliable.components.features import compute_component_features


def test_summarize_components_df_computes_expected_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "BIT",
                "dataset": "LEVIR-CD",
                "image_id": "img1",
                "component_id": 1,
                "is_error_component": True,
                "area": 10,
                "mean_entropy": 0.2,
                "mean_tta": 0.1,
                "mean_shift": 0.05,
            },
            {
                "model_name": "BIT",
                "dataset": "LEVIR-CD",
                "image_id": "img2",
                "component_id": 2,
                "is_error_component": False,
                "area": 20,
                "mean_entropy": 0.4,
                "mean_tta": 0.2,
                "mean_shift": 0.15,
            },
        ]
    )

    out = summarize_components_df(df)

    assert int(out.loc[0, "num_images"]) == 2
    assert int(out.loc[0, "num_components"]) == 2
    assert int(out.loc[0, "num_error_components"]) == 1
    assert float(out.loc[0, "error_component_rate"]) == 0.5


def test_summarize_missed_change_computes_expected_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "BIT",
                "dataset": "LEVIR-CD",
                "image_id": "img1",
                "gt_component_id": 1,
                "is_missed": True,
                "gt_area": 12,
                "pred_overlap_ratio": 0.0,
            },
            {
                "model_name": "BIT",
                "dataset": "LEVIR-CD",
                "image_id": "img1",
                "gt_component_id": 2,
                "is_missed": False,
                "gt_area": 18,
                "pred_overlap_ratio": 0.6,
            },
        ]
    )

    out = summarize_missed_change(df)

    assert int(out.loc[0, "num_images"]) == 1
    assert int(out.loc[0, "num_gt_components"]) == 2
    assert int(out.loc[0, "num_missed_components"]) == 1
    assert float(out.loc[0, "miss_rate"]) == 0.5


def test_compute_component_features_handles_empty_component_case() -> None:
    labeled_pred = np.zeros((4, 4), dtype=int)
    prob_map = np.zeros((4, 4), dtype=float)
    gt_mask = np.zeros((4, 4), dtype=int)

    out = compute_component_features(
        labeled_pred=labeled_pred,
        prob_map=prob_map,
        gt_mask=gt_mask,
        image_id="img0",
        dataset="LEVIR-256",
        model_name="BIT",
    )

    assert list(out.columns) == [
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
    assert out.empty

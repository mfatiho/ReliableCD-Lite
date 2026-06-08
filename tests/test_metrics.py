from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.metrics.calibration_metrics import brier_score, expected_calibration_error
from reliable.metrics.cd_metrics import binary_cd_metrics, binary_dataset_metrics
from reliable.metrics.uncertainty_metrics import component_error_auroc_ap, component_error_metrics_by_image
from scripts.run_threshold_sensitivity import _summarize_threshold_setting


def test_cd_metrics_basic() -> None:
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    metrics = binary_cd_metrics(pred, gt)
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0


def test_cd_metrics_empty_policy_variants() -> None:
    pred = np.zeros((2, 2), dtype=np.uint8)
    gt = np.zeros((2, 2), dtype=np.uint8)
    zero_metrics = binary_cd_metrics(pred, gt, empty_policy="zero")
    one_metrics = binary_cd_metrics(pred, gt, empty_policy="one")
    nan_metrics = binary_cd_metrics(pred, gt, empty_policy="nan")
    assert zero_metrics["f1"] == 0.0
    assert one_metrics["f1"] == 1.0
    assert np.isnan(nan_metrics["f1"])


def test_dataset_level_binary_metrics() -> None:
    pred = np.array([[[1, 0], [1, 0]]], dtype=np.uint8)
    gt = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    metrics = binary_dataset_metrics(pred, gt)
    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["fn"] == 0.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_calibration_metrics_basic() -> None:
    probs = np.array([0.1, 0.9, 0.8, 0.2])
    labels = np.array([0, 1, 1, 0])
    assert expected_calibration_error(probs, labels) >= 0.0
    assert brier_score(probs, labels) >= 0.0


def test_component_error_auroc_ap() -> None:
    df = pd.DataFrame({"is_error_component": [0, 1, 0, 1], "score": [0.1, 0.9, 0.2, 0.8]})
    metrics = component_error_auroc_ap(df, "score")
    assert metrics["error_auroc"] > 0.5


def test_component_error_metrics_by_image() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b"],
            "is_error_component": [0, 1, 0, 1],
            "score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    out = component_error_metrics_by_image(df, "score")
    assert out["image_id"].tolist() == ["a", "b"]
    assert out["error_auroc"].notna().all()


def test_threshold_sensitivity_summary_counts_components() -> None:
    pred_masks = np.array([[[1, 1], [0, 0]]], dtype=np.uint8)
    gt_masks = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    out = _summarize_threshold_setting(
        model_name="BIT",
        dataset="levir-256",
        dataset_canonical="LEVIR-CD",
        threshold=0.5,
        pred_masks=pred_masks,
        gt_masks=gt_masks,
    )
    assert out["num_images"] == 1
    assert out["num_components"] == 0
    assert 0.0 <= out["global_f1"] <= 1.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reliable.referral.analyst_sensitivity import (
    analyst_sensitivity_from_oracle_gain,
    attach_analyst_sensitivity_metadata,
)
from scripts.run_analyst_sensitivity import _build_from_referral_wide


def test_attach_analyst_sensitivity_metadata_prepends_context_columns() -> None:
    base = analyst_sensitivity_from_oracle_gain(0.08, 0.05, alphas=[1.0, 0.5])
    out = attach_analyst_sensitivity_metadata(
        base,
        dataset="WHU-256",
        budget=0.05,
        pixel_source="entropy",
        component_score="score_crs4",
        oracle_f1_gain=0.08,
        oracle_iou_gain=0.05,
    )
    assert out.columns.tolist()[:6] == [
        "dataset",
        "budget",
        "pixel_source",
        "component_score",
        "oracle_f1_gain_upper_bound",
        "oracle_iou_gain_upper_bound",
    ]
    assert out["dataset"].eq("WHU-256").all()


def test_build_from_referral_wide_filters_and_writes_per_setting_files(tmp_path: Path) -> None:
    referral_wide = pd.DataFrame(
        [
            {
                "dataset": "WHU-256",
                "budget": 0.05,
                "pixel_source": "entropy",
                "component_score": "score_crs4",
                "component_f1_gain_upper_bound": 0.08,
                "component_iou_gain_upper_bound": 0.05,
                "pixel_f1_gain_upper_bound": 0.11,
                "pixel_iou_gain_upper_bound": 0.07,
            },
            {
                "dataset": "LEVIR-256",
                "budget": 0.05,
                "pixel_source": "entropy",
                "component_score": "score_crs4",
                "component_f1_gain_upper_bound": 0.02,
                "component_iou_gain_upper_bound": 0.01,
                "pixel_f1_gain_upper_bound": 0.03,
                "pixel_iou_gain_upper_bound": 0.02,
            },
        ]
    )
    wide_path = tmp_path / "referral_wide.csv"
    referral_wide.to_csv(wide_path, index=False)
    out = _build_from_referral_wide(
        referral_wide_path=wide_path,
        datasets=["WHU-256"],
        budgets=[0.05],
        pixel_source="entropy",
        component_score="score_crs4",
        gain_source="component",
        alphas=[1.0, 0.5],
        per_setting_dir=str(tmp_path / "per_setting"),
    )
    assert out["dataset"].unique().tolist() == ["WHU-256"]
    assert out["analyst_correction_accuracy"].tolist() == [1.0, 0.5]
    assert out["expected_f1_gain"].tolist() == [0.08, 0.04]
    assert (tmp_path / "per_setting" / "analyst_sensitivity_whu-256_b050.csv").exists()

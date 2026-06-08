from __future__ import annotations

import pandas as pd


def analyst_sensitivity_from_oracle_gain(
    oracle_f1_gain: float,
    oracle_iou_gain: float,
    alphas: list[float] | None = None,
) -> pd.DataFrame:
    """Simple analytical sensitivity bound for imperfect reviewer correction."""
    alphas = alphas or [1.0, 0.9, 0.7, 0.5]
    return pd.DataFrame(
        [
            {
                "analyst_correction_accuracy": alpha,
                "expected_f1_gain": alpha * oracle_f1_gain,
                "expected_iou_gain": alpha * oracle_iou_gain,
            }
            for alpha in alphas
        ]
    )


def attach_analyst_sensitivity_metadata(
    sensitivity_df: pd.DataFrame,
    *,
    dataset: str,
    budget: float,
    pixel_source: str,
    component_score: str,
    oracle_f1_gain: float,
    oracle_iou_gain: float,
) -> pd.DataFrame:
    out = sensitivity_df.copy()
    out.insert(0, "oracle_iou_gain_upper_bound", oracle_iou_gain)
    out.insert(0, "oracle_f1_gain_upper_bound", oracle_f1_gain)
    out.insert(0, "component_score", component_score)
    out.insert(0, "pixel_source", pixel_source)
    out.insert(0, "budget", budget)
    out.insert(0, "dataset", dataset)
    return out

from __future__ import annotations

import pandas as pd

from reliable.scoring.baselines import MAIN_BASELINES
from reliable.visualization.curves import (
    REVIEW_BUDGET_COMPONENT_COMPARE_COL,
    REVIEW_BUDGET_COMPONENT_PRIMARY_COL,
    REVIEW_BUDGET_PIXEL_COMPARE_COL,
    _cost_accuracy_methods,
    plot_crs_ablation_bars,
    plot_risk_coverage_curves,
    plot_risk_coverage_summary,
)


def test_main_baselines_match_active_scope() -> None:
    assert MAIN_BASELINES == ["random", "margin", "entropy", "crs3", "crs4"]


def test_review_budget_curve_uses_apples_to_apples_error_pixel_recall() -> None:
    assert REVIEW_BUDGET_COMPONENT_COMPARE_COL == "component_error_pixel_recall"
    assert REVIEW_BUDGET_PIXEL_COMPARE_COL == "pixel_error_pixel_recall"
    assert REVIEW_BUDGET_COMPONENT_PRIMARY_COL == "component_error_recall"


def test_crs_ablation_plot_supports_single_lowercase_dataset(tmp_path) -> None:
    ablation_df = pd.DataFrame(
        [
            {"dataset": "levir-256", "method": "Random", "error_recall": 0.20},
            {"dataset": "levir-256", "method": "Entropy", "error_recall": 0.35},
            {"dataset": "levir-256", "method": "Margin", "error_recall": 0.40},
            {"dataset": "levir-256", "method": "CRS-1", "error_recall": 0.42},
            {"dataset": "levir-256", "method": "CRS-2", "error_recall": 0.45},
            {"dataset": "levir-256", "method": "CRS-3", "error_recall": 0.50},
            {"dataset": "levir-256", "method": "CRS-4", "error_recall": 0.55},
        ]
    )
    out_path = tmp_path / "crs_ablation_bars.pdf"
    plot_crs_ablation_bars(ablation_df, out_path, budget_label="5%")
    assert out_path.exists()


def test_cost_accuracy_places_single_map_scores_at_same_cost() -> None:
    runtime_df = pd.DataFrame(
        [
            {"dataset": "levir-256", "mode": "deterministic", "cost_multiplier": 1.0},
            {"dataset": "levir-256", "mode": "fast", "cost_multiplier": 0.79},
            {"dataset": "levir-256", "mode": "full", "cost_multiplier": 14.13},
        ]
    )

    method_costs = {
        label: cost
        for label, _col, cost, _is_default in _cost_accuracy_methods(runtime_df, "levir-256")
    }

    assert method_costs["Entropy"] == 1.0
    assert method_costs["Margin"] == 1.0
    assert method_costs["CRS-3"] == 1.0
    assert method_costs["CRS-4"] == 14.13


def test_risk_coverage_summary_plot_supports_per_dataset_input(tmp_path) -> None:
    curves_input = {
        "levir-256": {
            "Random": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.50, 0.48, 0.46]}),
            "Margin": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.40, 0.28, 0.16]}),
            "Entropy": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.42, 0.30, 0.18]}),
            "CRS-3": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.36, 0.24, 0.12]}),
            "CRS-4": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.34, 0.22, 0.10]}),
        },
        "whu-256": {
            "Random": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.90, 0.89, 0.88]}),
            "Margin": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.82, 0.72, 0.63]}),
            "Entropy": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.83, 0.73, 0.64]}),
            "CRS-3": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.80, 0.70, 0.60]}),
            "CRS-4": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.79, 0.69, 0.59]}),
        },
    }
    out_path = tmp_path / "risk_coverage_curves.pdf"
    plot_risk_coverage_curves(curves_input, out_path, model_name="BIT")
    assert out_path.exists()


def test_aurc_summary_plot_supports_per_dataset_input(tmp_path) -> None:
    curves_input = {
        "levir-256": {
            "Random": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.50, 0.48, 0.46]}),
            "Margin": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.40, 0.28, 0.16]}),
            "Entropy": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.42, 0.30, 0.18]}),
            "CRS-3": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.36, 0.24, 0.12]}),
            "CRS-4": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.34, 0.22, 0.10]}),
        },
        "whu-256": {
            "Random": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.90, 0.89, 0.88]}),
            "Margin": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.82, 0.72, 0.63]}),
            "Entropy": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.83, 0.73, 0.64]}),
            "CRS-3": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.80, 0.70, 0.60]}),
            "CRS-4": pd.DataFrame({"coverage": [0.0, 0.5, 1.0], "residual_risk": [0.79, 0.69, 0.59]}),
        },
    }
    out_path = tmp_path / "figure4_aurc_summary.pdf"
    plot_risk_coverage_summary(curves_input, out_path, model_name="BIT")
    assert out_path.exists()

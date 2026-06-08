from __future__ import annotations

import pandas as pd

from reliable.visualization.heatmaps import _prepare_error_corr_df
from reliable.visualization.panels import select_focus_budgets, summarize_budget_labels
from scripts.make_figures import _filter_components_by_model_name, _infer_model_name


def test_infer_model_name_prefers_single_referral_model_over_mixed_components() -> None:
    components_df = pd.DataFrame(
        [
            {"model_name": "BIT", "dataset": "levir-256"},
            {"model_name": "ChangeFormer", "dataset": "levir-256"},
        ]
    )
    referral_df = pd.DataFrame(
        [
            {"model_name": "BIT", "dataset": "levir-256", "budget": 0.05},
            {"model_name": "BIT", "dataset": "whu-256", "budget": 0.05},
        ]
    )

    assert _infer_model_name(components_df, referral_df) == "BIT"


def test_filter_components_by_model_name_restricts_mixed_directory() -> None:
    components_df = pd.DataFrame(
        [
            {"model_name": "BIT", "dataset": "levir-256", "value": 1},
            {"model_name": "ChangeFormer", "dataset": "levir-256", "value": 2},
            {"model_name": "BIT", "dataset": "whu-256", "value": 3},
        ]
    )

    filtered = _filter_components_by_model_name(components_df, "BIT")

    assert set(filtered["model_name"]) == {"BIT"}
    assert filtered["value"].tolist() == [1, 3]


def test_prepare_error_corr_df_orders_by_global_strength() -> None:
    corr_df = pd.DataFrame(
        [
            {"feature": "compactness", "spearman_rho_with_error": -0.10},
            {"feature": "mean_shift", "spearman_rho_with_error": 0.31},
            {"feature": "mean_entropy", "spearman_rho_with_error": 0.42},
            {"feature": "boundary_uncertainty", "spearman_rho_with_error": 0.45},
            {"feature": "probability_margin", "spearman_rho_with_error": -0.43},
        ]
    )

    out = _prepare_error_corr_df(corr_df)

    assert out["feature"].tolist() == [
        "boundary_uncertainty",
        "probability_margin",
        "mean_entropy",
        "mean_shift",
        "compactness",
    ]


def test_summarize_budget_labels_formats_sorted_unique_percentages() -> None:
    labels = summarize_budget_labels([0.10, 0.05, 0.05, 0.20])
    assert labels == "5% / 10% / 20%"


def test_select_focus_budgets_prefers_5_and_10_when_available() -> None:
    focus = select_focus_budgets([0.01, 0.03, 0.05, 0.10, 0.20])
    assert focus == [0.05, 0.10]

from __future__ import annotations

import pandas as pd

from reliable.stats.bootstrap import bootstrap_ci_per_image, bootstrap_ratio_by_image
from reliable.stats.correlation import spearman_feature_error_correlation
from reliable.stats.significance import paired_wilcoxon


def test_bootstrap_ci_returns_ordered_bounds() -> None:
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "value": [0.2, 0.4, 0.6]})
    lo, hi = bootstrap_ci_per_image(df, "value", n_boot=100)
    assert lo <= hi


def test_bootstrap_ratio_by_image_returns_ordered_bounds() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "reviewed": [2, 3, 5],
            "total": [10, 10, 10],
        }
    )
    lo, hi = bootstrap_ratio_by_image(df, "reviewed", "total", n_boot=100)
    assert lo <= hi


def test_wilcoxon_wrapper_works() -> None:
    df = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [0.2, 0.3, 0.4, 0.5]})
    out = paired_wilcoxon(df, "a", "b")
    assert "p_value" in out


def test_wilcoxon_wrapper_handles_identical_pairs() -> None:
    df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.1, 0.2, 0.3]})
    out = paired_wilcoxon(df, "a", "b")
    assert out["p_value"] == 1.0
    assert out["n_pairs"] == 3


def test_spearman_feature_error_correlation_returns_ci_columns() -> None:
    df = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "b", "c", "c"],
            "f1": [0.1, 0.2, 0.2, 0.5, 0.6, 0.8],
            "f2": [0.9, 0.8, 0.8, 0.5, 0.4, 0.2],
            "is_error_component": [0, 0, 0, 1, 1, 1],
        }
    )

    out = spearman_feature_error_correlation(df, ["f1", "f2"], n_boot=50, seed=7)

    assert list(out["feature"]) == ["f1", "f2"]
    assert {"rho_ci_lo", "rho_ci_hi"}.issubset(out.columns)
    assert (out["rho_ci_lo"] <= out["rho_ci_hi"]).all()

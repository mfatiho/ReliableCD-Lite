from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.scoring.baselines import prepare_component_scores
from reliable.scoring.crs import add_mean_mi_or_entropy, compute_crs_variants
from reliable.scoring.zscore import CRSNormalizer


def make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": [10, 20, 30],
            "mean_prob": [0.4, 0.6, 0.8],
            "mean_entropy": [0.2, 0.5, 0.7],
            "mean_mi": [0.1, np.nan, 0.8],
            "boundary_uncertainty": [0.3, 0.4, 0.2],
            "probability_margin": [0.2, 0.1, 0.3],
            "mean_tta": [0.1, 0.2, 0.3],
            "mean_shift": [0.2, 0.3, 0.4],
        }
    )


def test_zscore_columns_created() -> None:
    df = add_mean_mi_or_entropy(make_df())
    normalizer = CRSNormalizer()
    normalizer.fit(
        df,
        ["mean_mi_or_entropy", "boundary_uncertainty", "probability_margin", "mean_tta", "mean_shift"],
    )
    out = normalizer.transform(df)
    assert "z_mean_mi_or_entropy" in out.columns


def test_crs_columns_created_and_entropy_fallback_used() -> None:
    df = add_mean_mi_or_entropy(make_df())
    normalizer = CRSNormalizer()
    columns = ["mean_mi_or_entropy", "boundary_uncertainty", "probability_margin", "mean_tta", "mean_shift"]
    normalizer.fit(df, columns)
    out = compute_crs_variants(normalizer.transform(df))
    assert all(col in out.columns for col in ["crs1", "crs2", "crs3", "crs4"])
    assert float(df.loc[1, "mean_mi_or_entropy"]) == float(df.loc[1, "mean_entropy"])


def test_prepare_component_scores_adds_crs_and_score_columns() -> None:
    df = make_df()
    out, normalizer = prepare_component_scores(df, seed=123)
    assert normalizer.columns == [
        "mean_mi_or_entropy",
        "boundary_uncertainty",
        "probability_margin",
        "mean_tta",
        "mean_shift",
    ]
    expected_columns = {
        "mean_mi_or_entropy",
        "z_mean_mi_or_entropy",
        "z_boundary_uncertainty",
        "z_probability_margin",
        "z_mean_tta",
        "z_mean_shift",
        "crs1",
        "crs2",
        "crs3",
        "crs4",
        "score_random",
        "score_area",
        "score_low_confidence",
        "score_margin",
        "score_entropy",
        "score_mi",
        "score_tta",
        "score_shift",
        "score_crs3",
        "score_crs4",
    }
    assert expected_columns.issubset(out.columns)
    assert out["score_random"].between(0.0, 1.0).all()


def test_prepare_component_scores_can_reuse_fixed_normalizer() -> None:
    reference = make_df()
    target = make_df().assign(mean_entropy=[1.0, 1.1, 1.2])

    _, normalizer = prepare_component_scores(reference, seed=123)
    out, reused = prepare_component_scores(target, seed=123, normalizer=normalizer)

    assert reused is normalizer
    expected = (target.loc[0, "boundary_uncertainty"] - normalizer.means["boundary_uncertainty"]) / normalizer.stds["boundary_uncertainty"]
    assert np.isclose(out.loc[0, "z_boundary_uncertainty"], expected)

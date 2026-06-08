from __future__ import annotations

import numpy as np
import pandas as pd

from reliable.scoring.crs import add_mean_mi_or_entropy, compute_crs_variants
from reliable.scoring.zscore import CRSNormalizer


MAIN_BASELINES = ["random", "margin", "entropy", "crs3", "crs4"]

SUPPLEMENTARY_BASELINES = [
    "random",
    "area",
    "low_confidence",
    "margin",
    "entropy",
    "mi",
    "tta",
    "shift",
    "crs3",
    "crs4",
]


def add_ranking_scores(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    out["score_random"] = rng.random(len(out))
    out["score_area"] = out["area"]
    out["score_low_confidence"] = -out["mean_prob"]
    out["score_margin"] = -out["probability_margin"]
    out["score_entropy"] = out["mean_entropy"]
    mean_mi = pd.to_numeric(out["mean_mi"], errors="coerce")
    mean_entropy = pd.to_numeric(out["mean_entropy"], errors="coerce")
    out["score_mi"] = mean_mi.fillna(mean_entropy)
    out["score_tta"] = out["mean_tta"]
    out["score_shift"] = out["mean_shift"]
    out["score_crs3"] = out["crs3"]
    out["score_crs4"] = out["crs4"]
    return out


def prepare_component_scores(
    df: pd.DataFrame,
    seed: int = 42,
    normalizer: CRSNormalizer | None = None,
) -> tuple[pd.DataFrame, CRSNormalizer]:
    out = add_mean_mi_or_entropy(df)
    zscore_columns = [
        "mean_mi_or_entropy",
        "boundary_uncertainty",
        "probability_margin",
        "mean_tta",
        "mean_shift",
    ]
    missing = [col for col in zscore_columns if col not in out.columns]
    if missing:
        raise KeyError(f"Missing component feature columns required for CRS scoring: {missing}")

    if normalizer is None:
        normalizer = CRSNormalizer()
        normalizer.fit(out, zscore_columns)
    else:
        missing_normalizer_cols = [col for col in zscore_columns if col not in normalizer.columns]
        if missing_normalizer_cols:
            raise KeyError(
                "CRS normalizer is missing required feature columns: "
                f"{missing_normalizer_cols}"
            )
    out = normalizer.transform(out)
    out = compute_crs_variants(out)
    out = add_ranking_scores(out, seed=seed)
    return out, normalizer

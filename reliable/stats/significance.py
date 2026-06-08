from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(
    df: pd.DataFrame,
    method_a_col: str,
    method_b_col: str,
) -> dict[str, float]:
    """Paired method comparison using Wilcoxon signed-rank."""
    paired = df[[method_a_col, method_b_col]].dropna()
    if paired.empty:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_pairs": 0}
    diffs = paired[method_a_col] - paired[method_b_col]
    if np.allclose(diffs.to_numpy(dtype=float), 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": int(len(paired))}
    stat, p = wilcoxon(paired[method_a_col], paired[method_b_col])
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": int(len(paired))}

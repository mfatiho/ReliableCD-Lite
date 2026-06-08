from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_feature_correlation(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Feature-feature Spearman correlation matrix."""
    return df[columns].corr(method="spearman")


def spearman_feature_error_correlation(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    image_col: str = "image_id",
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Feature-error Spearman correlations with optional cluster-bootstrap CIs."""
    rows = []
    for col in feature_columns:
        sub_df = df[[image_col, col, "is_error_component"]].dropna().reset_index(drop=True)
        x = sub_df[col].to_numpy(dtype=float)
        y = sub_df["is_error_component"].to_numpy(dtype=float)
        rho = float("nan") if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y).statistic)

        ci_lo = float("nan")
        ci_hi = float("nan")
        image_ids = sub_df[image_col].unique()
        if n_boot > 0 and len(image_ids) >= 2:
            grouped_indices = [
                sub_df.index[sub_df[image_col] == iid].to_numpy(dtype=int)
                for iid in image_ids
            ]
            rng = np.random.default_rng(seed)
            samples: list[float] = []
            for _ in range(n_boot):
                sampled = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
                boot_idx = np.concatenate([grouped_indices[i] for i in sampled])
                if np.ptp(x[boot_idx]) == 0 or np.ptp(y[boot_idx]) == 0:
                    continue
                stat = float(spearmanr(x[boot_idx], y[boot_idx]).statistic)
                if not np.isnan(stat):
                    samples.append(stat)
            if samples:
                alpha = (1.0 - ci) / 2.0
                ci_lo = float(np.quantile(samples, alpha))
                ci_hi = float(np.quantile(samples, 1.0 - alpha))

        rows.append(
            {
                "feature": col,
                "spearman_rho_with_error": rho,
                "rho_ci_lo": ci_lo,
                "rho_ci_hi": ci_hi,
            }
        )
    return pd.DataFrame(rows)

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci_metric_by_image(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    image_col: str = "image_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Cluster bootstrap CI for any scalar metric computed over a component DataFrame.

    Resamples image_ids with replacement; all components from sampled images form the
    bootstrap replicate. This preserves within-image component correlations.
    """
    image_ids = df[image_col].unique()
    if len(image_ids) < 2:
        return (float("nan"), float("nan"))
    # Pre-group for efficiency
    groups: dict[str, pd.DataFrame] = {iid: df[df[image_col] == iid] for iid in image_ids}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
        boot_df = pd.concat([groups[iid] for iid in sampled_ids], ignore_index=True)
        val = metric_fn(boot_df)
        if not np.isnan(val):
            samples.append(val)
    if not samples:
        return (float("nan"), float("nan"))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def bootstrap_ci_per_image(
    per_image_values: pd.DataFrame,
    value_col: str,
    image_col: str = "image_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Image-level bootstrap CI."""
    grouped = per_image_values.groupby(image_col)[value_col].mean()
    values = grouped.to_numpy(dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        draw = rng.choice(values, size=len(values), replace=True)
        samples.append(float(draw.mean()))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def bootstrap_ratio_by_image(
    per_image_values: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a dataset-level ratio from per-image numerator/denominator columns."""
    if per_image_values.empty:
        return (float("nan"), float("nan"))
    grouped = per_image_values.groupby("image_id")[[numerator_col, denominator_col]].sum()
    numerators = grouped[numerator_col].to_numpy(dtype=float)
    denominators = grouped[denominator_col].to_numpy(dtype=float)
    if len(numerators) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(numerators), size=len(numerators))
        denom = float(denominators[idx].sum())
        if denom <= 0:
            samples.append(0.0)
            continue
        samples.append(float(numerators[idx].sum() / denom))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))

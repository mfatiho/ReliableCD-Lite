from __future__ import annotations

import numpy as np
import pandas as pd


def component_error_auroc_ap(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Error AUROC/AUPRC for component risk score."""
    y_true = df["is_error_component"].astype(int).to_numpy()
    scores = df[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"error_auroc": float("nan"), "error_auprc": float("nan")}
    auroc, auprc = _binary_auroc_and_ap(y_true, scores)
    return {"error_auroc": auroc, "error_auprc": auprc}


def component_error_metrics_with_ci(
    df: pd.DataFrame,
    score_col: str,
    image_col: str = "image_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Error AUROC/AUPRC with 95% cluster bootstrap CIs and random AUPRC baseline.

    Single-pass bootstrap: AUROC and AUPRC are computed together per replicate.
    Pre-extracts numpy arrays per image to avoid DataFrame overhead in the hot loop.
    """
    base = component_error_auroc_ap(df, score_col)

    n_total = len(df)
    n_errors = int(df["is_error_component"].astype(int).sum())
    random_auprc = n_errors / n_total if n_total > 0 else float("nan")

    has_image_col = image_col in df.columns and df[image_col].nunique() >= 4
    if has_image_col:
        (auroc_lo, auroc_hi), (auprc_lo, auprc_hi) = _bootstrap_auroc_auprc_ci(
            df, score_col, image_col=image_col, n_boot=n_boot, ci=ci, seed=seed,
        )
    else:
        auroc_lo = auroc_hi = auprc_lo = auprc_hi = float("nan")

    return {
        **base,
        "auroc_ci_lo": auroc_lo,
        "auroc_ci_hi": auroc_hi,
        "auprc_ci_lo": auprc_lo,
        "auprc_ci_hi": auprc_hi,
        "random_auprc": random_auprc,
    }


def component_error_metrics_by_image(
    df: pd.DataFrame,
    score_col: str,
    image_col: str = "image_id",
) -> pd.DataFrame:
    """Per-image error AUROC/AUPRC for paired dataset-level significance tests.

    Images with only one class are retained with NaN metrics so downstream callers
    can drop them consistently per metric.
    """
    rows: list[dict[str, float | str | int]] = []
    for image_id, group in df.groupby(image_col, dropna=False, sort=False):
        y_true = group["is_error_component"].astype(int).to_numpy()
        scores = group[score_col].to_numpy(dtype=float)
        metrics = {"error_auroc": float("nan"), "error_auprc": float("nan")}
        if len(np.unique(y_true)) >= 2:
            auroc, auprc = _binary_auroc_and_ap(y_true, scores)
            metrics = {"error_auroc": auroc, "error_auprc": auprc}
        rows.append(
            {
                image_col: str(image_id),
                "num_components": int(len(group)),
                "error_prevalence": float(group["is_error_component"].astype(int).mean()) if len(group) else float("nan"),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fast vectorized metric kernels
# ---------------------------------------------------------------------------

def _binary_auroc_and_ap(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Compute AUROC and AUPRC together in one combined pass."""
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # AUROC via Mann-Whitney / searchsorted — O(N log N), no Python loop
    pos_scores = scores[pos_mask]
    neg_sorted = np.sort(scores[~pos_mask])
    lo = np.searchsorted(neg_sorted, pos_scores, side="left")   # count neg < pos
    hi = np.searchsorted(neg_sorted, pos_scores, side="right")  # count neg <= pos
    auroc = float((lo.sum() + 0.5 * (hi - lo).sum()) / (n_pos * n_neg))

    # AUPRC via cumsum — O(N log N), no Python loop
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    cumtp = np.cumsum(y_sorted)
    auprc = float((cumtp / np.arange(1, len(y_sorted) + 1))[y_sorted.astype(bool)].mean())

    return auroc, auprc


# Keep scalar wrappers for external callers that use them directly.
def _binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return _binary_auroc_and_ap(y_true, scores)[0]


def _binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    return _binary_auroc_and_ap(y_true, scores)[1]


# ---------------------------------------------------------------------------
# Fast combined cluster bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_auroc_auprc_ci(
    df: pd.DataFrame,
    score_col: str,
    image_col: str = "image_id",
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Single-pass cluster bootstrap for AUROC and AUPRC CIs.

    Pre-extracts one numpy array per image so the hot loop uses np.concatenate
    instead of pd.concat — roughly 10-50x faster for large component tables.
    """
    image_ids = df[image_col].unique()
    # Pre-extract once; avoids all DataFrame overhead inside the loop
    labels_by_id: dict[object, np.ndarray] = {
        iid: df.loc[df[image_col] == iid, "is_error_component"].to_numpy(dtype=np.int8)
        for iid in image_ids
    }
    scores_by_id: dict[object, np.ndarray] = {
        iid: df.loc[df[image_col] == iid, score_col].to_numpy(dtype=np.float64)
        for iid in image_ids
    }

    rng = np.random.default_rng(seed)
    auroc_buf = np.empty(n_boot, dtype=np.float64)
    auprc_buf = np.empty(n_boot, dtype=np.float64)
    n_valid = 0

    for _ in range(n_boot):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        y = np.concatenate([labels_by_id[iid] for iid in sampled])
        s = np.concatenate([scores_by_id[iid] for iid in sampled])
        if len(np.unique(y)) < 2:
            continue
        auroc_buf[n_valid], auprc_buf[n_valid] = _binary_auroc_and_ap(y, s)
        n_valid += 1

    if n_valid < 2:
        nan_pair: tuple[float, float] = (float("nan"), float("nan"))
        return nan_pair, nan_pair

    alpha = (1.0 - ci) / 2.0
    auroc_ci = (
        float(np.quantile(auroc_buf[:n_valid], alpha)),
        float(np.quantile(auroc_buf[:n_valid], 1.0 - alpha)),
    )
    auprc_ci = (
        float(np.quantile(auprc_buf[:n_valid], alpha)),
        float(np.quantile(auprc_buf[:n_valid], 1.0 - alpha)),
    )
    return auroc_ci, auprc_ci

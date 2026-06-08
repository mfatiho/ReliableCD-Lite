from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(errors: np.ndarray, confidence_or_risk: np.ndarray) -> pd.DataFrame:
    """Selective prediction curve data sorted from high risk to low risk."""
    errors = np.asarray(errors).astype(float).reshape(-1)
    risk = np.asarray(confidence_or_risk, dtype=float).reshape(-1)
    order = np.argsort(risk)[::-1]
    sorted_errors = errors[order]
    n = len(sorted_errors)
    rows = []
    for reviewed in range(1, n + 1):
        remaining = sorted_errors[reviewed:]
        coverage = 1.0 - (reviewed / n)
        residual_risk = float(remaining.mean()) if len(remaining) else 0.0
        rows.append({"reviewed_fraction": reviewed / n, "coverage": coverage, "residual_risk": residual_risk})
    return pd.DataFrame(rows)

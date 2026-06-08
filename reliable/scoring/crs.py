from __future__ import annotations

import pandas as pd


def add_mean_mi_or_entropy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean_mi = pd.to_numeric(out["mean_mi"], errors="coerce")
    mean_entropy = pd.to_numeric(out["mean_entropy"], errors="coerce")
    out["mean_mi_or_entropy"] = mean_mi.fillna(mean_entropy)
    return out


def compute_crs_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CRS-1/2/3/4 using equal weights."""
    out = add_mean_mi_or_entropy(df)
    required = [
        "z_mean_mi_or_entropy",
        "z_boundary_uncertainty",
        "z_probability_margin",
        "z_mean_tta",
        "z_mean_shift",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"Missing normalized CRS columns: {missing}")

    out["crs1"] = out["z_mean_mi_or_entropy"]
    out["crs2"] = out["z_mean_mi_or_entropy"] + out["z_boundary_uncertainty"]
    out["crs3"] = out["z_mean_mi_or_entropy"] + out["z_boundary_uncertainty"] - out["z_probability_margin"]
    out["crs4"] = (
        out["z_mean_mi_or_entropy"]
        + out["z_mean_tta"]
        + out["z_mean_shift"]
        + out["z_boundary_uncertainty"]
        - out["z_probability_margin"]
    )
    return out

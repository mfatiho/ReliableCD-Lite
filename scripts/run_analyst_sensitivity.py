from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reliable.referral.analyst_sensitivity import (
    analyst_sensitivity_from_oracle_gain,
    attach_analyst_sensitivity_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-f1-gain", type=float, default=None)
    parser.add_argument("--oracle-iou-gain", type=float, default=None)
    parser.add_argument("--referral-wide", default=None, help="Referral wide CSV from scripts/run_referral.py")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=float, default=None)
    parser.add_argument("--pixel-source", default=None)
    parser.add_argument("--component-score", default=None)
    parser.add_argument("--gain-source", choices=["component", "pixel"], default="component")
    parser.add_argument("--alphas", nargs="*", type=float, default=[1.0, 0.9, 0.7, 0.5])
    parser.add_argument("--per-setting-dir", default=None, help="Optional directory for per-dataset/budget CSV files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.referral_wide:
        table = _build_from_referral_wide(
            referral_wide_path=args.referral_wide,
            datasets=args.datasets,
            budgets=args.budgets,
            pixel_source=args.pixel_source,
            component_score=args.component_score,
            gain_source=args.gain_source,
            alphas=args.alphas,
            per_setting_dir=args.per_setting_dir,
        )
    else:
        if args.oracle_f1_gain is None or args.oracle_iou_gain is None:
            parser.error("--oracle-f1-gain and --oracle-iou-gain are required unless --referral-wide is provided.")
        table = analyst_sensitivity_from_oracle_gain(
            oracle_f1_gain=args.oracle_f1_gain,
            oracle_iou_gain=args.oracle_iou_gain,
            alphas=args.alphas,
        )

    table.to_csv(out_path, index=False)


def _build_from_referral_wide(
    *,
    referral_wide_path: str | Path,
    datasets: list[str] | None,
    budgets: list[float] | None,
    pixel_source: str | None,
    component_score: str | None,
    gain_source: str,
    alphas: list[float],
    per_setting_dir: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(referral_wide_path)
    filtered = df.copy()
    if datasets:
        filtered = filtered[filtered["dataset"].isin(datasets)]
    if budgets:
        filtered = filtered[filtered["budget"].isin(budgets)]
    if pixel_source is not None:
        filtered = filtered[filtered["pixel_source"] == pixel_source]
    if component_score is not None:
        filtered = filtered[filtered["component_score"] == component_score]
    if filtered.empty:
        raise ValueError("No referral rows matched the requested analyst sensitivity filters.")

    per_setting_path = None if per_setting_dir is None else Path(per_setting_dir)
    if per_setting_path is not None:
        per_setting_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    ordered = filtered.sort_values(["dataset", "budget", "pixel_source", "component_score"]).reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        oracle_f1_gain = float(getattr(row, f"{gain_source}_f1_gain_upper_bound"))
        oracle_iou_gain = float(getattr(row, f"{gain_source}_iou_gain_upper_bound"))
        sensitivity_df = analyst_sensitivity_from_oracle_gain(
            oracle_f1_gain=oracle_f1_gain,
            oracle_iou_gain=oracle_iou_gain,
            alphas=alphas,
        )
        sensitivity_df = attach_analyst_sensitivity_metadata(
            sensitivity_df,
            dataset=str(row.dataset),
            budget=float(row.budget),
            pixel_source=str(row.pixel_source),
            component_score=str(row.component_score),
            oracle_f1_gain=oracle_f1_gain,
            oracle_iou_gain=oracle_iou_gain,
        )
        if per_setting_path is not None:
            per_file = per_setting_path / f"analyst_sensitivity_{_dataset_slug(str(row.dataset))}_b{_budget_slug(float(row.budget))}.csv"
            sensitivity_df.to_csv(per_file, index=False)
        frames.append(sensitivity_df)
    return pd.concat(frames, ignore_index=True)


def _dataset_slug(dataset: str) -> str:
    return dataset.lower().replace(" ", "-")


def _budget_slug(budget: float) -> str:
    return f"{int(round(budget * 1000)):03d}"


if __name__ == "__main__":
    main()

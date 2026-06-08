from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running as `python scripts/run_referral.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reliable.components.extraction import extract_components
from reliable.data import canonical_dataset_name, dataset_instance_slug
from reliable.inference.save_maps import load_prediction_bundle
from reliable.referral.component_referral import component_level_referral
from reliable.referral.pixel_referral import dataset_pixel_referral
from reliable.stats.bootstrap import bootstrap_ci_per_image
from reliable.stats.significance import paired_wilcoxon
from reliable.utils.manifest import write_json_manifest
from reliable.utils.progress import make_progress
from reliable.visualization.curves import plot_review_budget_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--budgets", nargs="+", required=True, type=float)
    parser.add_argument("--component-score", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--uq-dir", default="results/uq")
    parser.add_argument("--components-dir", default="results/components")
    parser.add_argument("--components", default=None, help="Optional explicit parquet path for a single dataset.")
    parser.add_argument("--pixel-map-source", default="mi", choices=["mi", "entropy", "tta", "shift", "margin", "low_confidence"])
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations for 95%% CI and paired diffs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling.")
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir)
    uq_dir = Path(args.uq_dir)
    components_dir = Path(args.components_dir)
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    dataset_progress = make_progress(args.datasets, total=len(args.datasets), desc="Referral datasets", unit="dataset")

    for dataset_name in dataset_progress:
        canonical = canonical_dataset_name(dataset_name)
        slug = dataset_instance_slug(dataset_name)
        model_slug = args.model.lower()
        bundle = load_prediction_bundle(baseline_dir / f"{model_slug}_{slug}_predictions.npz")
        image_ids = [str(x) for x in bundle["image_ids"]]
        pred_masks = {image_id: bundle["pred_masks"][idx].astype(np.uint8) for idx, image_id in enumerate(image_ids)}
        gt_masks = {image_id: bundle["gt_masks"][idx].astype(np.uint8) for idx, image_id in enumerate(image_ids)}
        image_area_by_id = {image_id: int(bundle["pred_masks"][idx].size) for idx, image_id in enumerate(image_ids)}

        components_path = _resolve_components_path(args.components, components_dir, model_slug, slug)
        components_df = pd.read_parquet(components_path)
        labeled_masks = _build_labeled_masks(pred_masks, components_df)
        uncertainty_maps = _load_pixel_source(args.pixel_map_source, uq_dir, model_slug, slug, bundle)

        budget_progress = make_progress(
            args.budgets,
            total=len(args.budgets),
            desc=f"{args.model}/{dataset_name} budgets",
            unit="budget",
            leave=False,
            position=1,
        )
        for budget in budget_progress:
            pixel_metrics, pixel_per_image = dataset_pixel_referral(
                uncertainty_maps=uncertainty_maps,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                review_budget=budget,
                return_per_image=True,
            )
            component_metrics, component_per_image = component_level_referral(
                components_df=components_df,
                score_col=args.component_score,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
                labeled_masks=labeled_masks,
                image_area_by_id=image_area_by_id,
                review_budget=budget,
                return_per_image=True,
            )
            pixel_ci = _pixel_referral_cis(pixel_per_image, n_boot=args.n_boot, seed=args.seed)
            component_ci = _component_referral_cis(component_per_image, n_boot=args.n_boot, seed=args.seed)
            paired_stats = _paired_referral_stats(component_per_image, pixel_per_image, n_boot=args.n_boot, seed=args.seed)
            pixel_per_image = pixel_per_image.assign(
                model_name=args.model,
                dataset=dataset_name,
                dataset_canonical=canonical,
                method="pixel",
                pixel_source=args.pixel_map_source,
                component_score=args.component_score,
                budget=budget,
            )
            component_per_image = component_per_image.assign(
                model_name=args.model,
                dataset=dataset_name,
                dataset_canonical=canonical,
                method="component",
                pixel_source=args.pixel_map_source,
                component_score=args.component_score,
                budget=budget,
            )
            per_image_rows.extend(pixel_per_image.to_dict(orient="records"))
            per_image_rows.extend(component_per_image.to_dict(orient="records"))

            long_rows.append(
                {
                    "model_name": args.model,
                    "dataset": dataset_name,
                    "dataset_canonical": canonical,
                    "method": "pixel",
                    "pixel_source": args.pixel_map_source,
                    "component_score": args.component_score,
                    "budget": budget,
                    "error_pixel_recall_ci_lo": pixel_ci["error_pixel_recall"][0],
                    "error_pixel_recall_ci_hi": pixel_ci["error_pixel_recall"][1],
                    "reviewed_area_pct_ci_lo": pixel_ci["reviewed_area_pct"][0],
                    "reviewed_area_pct_ci_hi": pixel_ci["reviewed_area_pct"][1],
                    "error_pixel_recall_pooled": pixel_metrics["error_pixel_recall_pooled"],
                    "reviewed_area_pct_pooled": pixel_metrics["reviewed_area_pct_pooled"],
                    "f1_gain_upper_bound_ci_lo": pixel_ci["f1_gain_upper_bound"][0],
                    "f1_gain_upper_bound_ci_hi": pixel_ci["f1_gain_upper_bound"][1],
                    "iou_gain_upper_bound_ci_lo": pixel_ci["iou_gain_upper_bound"][0],
                    "iou_gain_upper_bound_ci_hi": pixel_ci["iou_gain_upper_bound"][1],
                    **pixel_metrics,
                }
            )
            long_rows.append(
                {
                    "model_name": args.model,
                    "dataset": dataset_name,
                    "dataset_canonical": canonical,
                    "method": "component",
                    "pixel_source": args.pixel_map_source,
                    "component_score": args.component_score,
                    "budget": budget,
                    "error_recall_ci_lo": component_ci["error_recall"][0],
                    "error_recall_ci_hi": component_ci["error_recall"][1],
                    "error_pixel_recall_ci_lo": component_ci["error_pixel_recall"][0],
                    "error_pixel_recall_ci_hi": component_ci["error_pixel_recall"][1],
                    "reviewed_area_pct_ci_lo": component_ci["reviewed_area_pct"][0],
                    "reviewed_area_pct_ci_hi": component_ci["reviewed_area_pct"][1],
                    "error_recall_pooled": component_metrics["error_recall_pooled"],
                    "error_pixel_recall_pooled": component_metrics["error_pixel_recall_pooled"],
                    "reviewed_area_pct_pooled": component_metrics["reviewed_area_pct_pooled"],
                    "f1_gain_upper_bound_ci_lo": component_ci["f1_gain_upper_bound"][0],
                    "f1_gain_upper_bound_ci_hi": component_ci["f1_gain_upper_bound"][1],
                    "iou_gain_upper_bound_ci_lo": component_ci["iou_gain_upper_bound"][0],
                    "iou_gain_upper_bound_ci_hi": component_ci["iou_gain_upper_bound"][1],
                    **component_metrics,
                }
            )
            wide_rows.append(
                {
                    "model_name": args.model,
                    "dataset": dataset_name,
                    "dataset_canonical": canonical,
                    "budget": budget,
                    "pixel_source": args.pixel_map_source,
                    "component_score": args.component_score,
                    "pixel_error_pixel_recall": pixel_metrics["error_pixel_recall"],
                    "pixel_error_pixel_recall_ci_lo": pixel_ci["error_pixel_recall"][0],
                    "pixel_error_pixel_recall_ci_hi": pixel_ci["error_pixel_recall"][1],
                    "pixel_error_pixel_recall_pooled": pixel_metrics["error_pixel_recall_pooled"],
                    "component_error_recall": component_metrics["error_recall"],
                    "component_error_recall_ci_lo": component_ci["error_recall"][0],
                    "component_error_recall_ci_hi": component_ci["error_recall"][1],
                    "component_error_recall_pooled": component_metrics["error_recall_pooled"],
                    "component_error_pixel_recall": component_metrics["error_pixel_recall"],
                    "component_error_pixel_recall_ci_lo": component_ci["error_pixel_recall"][0],
                    "component_error_pixel_recall_ci_hi": component_ci["error_pixel_recall"][1],
                    "component_error_pixel_recall_pooled": component_metrics["error_pixel_recall_pooled"],
                    "pixel_f1_gain_upper_bound": pixel_metrics["f1_gain_upper_bound"],
                    "pixel_f1_gain_upper_bound_ci_lo": pixel_ci["f1_gain_upper_bound"][0],
                    "pixel_f1_gain_upper_bound_ci_hi": pixel_ci["f1_gain_upper_bound"][1],
                    "component_f1_gain_upper_bound": component_metrics["f1_gain_upper_bound"],
                    "component_f1_gain_upper_bound_ci_lo": component_ci["f1_gain_upper_bound"][0],
                    "component_f1_gain_upper_bound_ci_hi": component_ci["f1_gain_upper_bound"][1],
                    "pixel_iou_gain_upper_bound": pixel_metrics["iou_gain_upper_bound"],
                    "pixel_iou_gain_upper_bound_ci_lo": pixel_ci["iou_gain_upper_bound"][0],
                    "pixel_iou_gain_upper_bound_ci_hi": pixel_ci["iou_gain_upper_bound"][1],
                    "component_iou_gain_upper_bound": component_metrics["iou_gain_upper_bound"],
                    "component_iou_gain_upper_bound_ci_lo": component_ci["iou_gain_upper_bound"][0],
                    "component_iou_gain_upper_bound_ci_hi": component_ci["iou_gain_upper_bound"][1],
                    "pixel_reviewed_area_pct": pixel_metrics["reviewed_area_pct"],
                    "pixel_reviewed_area_pct_ci_lo": pixel_ci["reviewed_area_pct"][0],
                    "pixel_reviewed_area_pct_ci_hi": pixel_ci["reviewed_area_pct"][1],
                    "pixel_reviewed_area_pct_pooled": pixel_metrics["reviewed_area_pct_pooled"],
                    "component_reviewed_area_pct": component_metrics["reviewed_area_pct"],
                    "component_reviewed_area_pct_ci_lo": component_ci["reviewed_area_pct"][0],
                    "component_reviewed_area_pct_ci_hi": component_ci["reviewed_area_pct"][1],
                    "component_reviewed_area_pct_pooled": component_metrics["reviewed_area_pct_pooled"],
                    "component_budget_overshoot_pct": component_metrics["budget_overshoot_pct"],
                    "component_reviewed_component_count": component_metrics["reviewed_component_count"],
                    **paired_stats,
                }
            )
            significance_rows.extend(
                _paired_referral_stats_rows(
                    model_name=args.model,
                    dataset_name=dataset_name,
                    canonical=canonical,
                    budget=budget,
                    pixel_source=args.pixel_map_source,
                    component_score=args.component_score,
                    paired_stats=paired_stats,
                )
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(long_rows).to_csv(out_path, index=False)
    wide_path = out_path.with_name(f"{out_path.stem}_wide.csv")
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(wide_path, index=False)
    significance_path = out_path.with_name(f"{out_path.stem}_significance.csv")
    pd.DataFrame(significance_rows).to_csv(significance_path, index=False)
    per_image_path = out_path.with_name(f"{out_path.stem}_per_image.csv")
    pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False)
    early_budget_path = out_path.with_name(f"{out_path.stem}_early_budget_table.csv")
    early_budget_df = _build_early_budget_table(wide_df)
    early_budget_df.to_csv(early_budget_path, index=False)
    whu_main_path = out_path.with_name(f"{out_path.stem}_whu_main_operational_table.csv")
    whu_main_df = _build_whu_main_operational_table(wide_df)
    whu_main_df.to_csv(whu_main_path, index=False)
    curve_path = plot_review_budget_curves(wide_df, out_path.with_name(f"{out_path.stem}_review_budget_curves.pdf"))
    write_json_manifest(
        out_path.with_suffix(".json"),
        {
            "model_name": args.model,
            "datasets": args.datasets,
            "budgets": args.budgets,
            "component_score": args.component_score,
            "pixel_map_source": args.pixel_map_source,
            "outputs": {
                "long_csv": str(out_path),
                "wide_csv": str(wide_path),
                "significance_csv": str(significance_path),
                "per_image_csv": str(per_image_path),
                "early_budget_table_csv": str(early_budget_path),
                "whu_main_operational_table_csv": str(whu_main_path),
                "review_budget_curves_pdf": str(curve_path),
            },
        },
    )


def _resolve_components_path(explicit_path: str | None, components_dir: Path, model_slug: str, slug: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    expected = components_dir / f"{model_slug}_{slug}_components.parquet"
    if expected.exists():
        return expected
    matches = sorted(components_dir.glob(f"{model_slug}_{slug}*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No components parquet found for {model_slug}_{slug} under {components_dir}.")
    if len(matches) == 1:
        return matches[0]
    raise FileExistsError(f"Multiple candidate component parquets found for {model_slug}_{slug}: {matches}")


def _build_labeled_masks(pred_masks: dict[str, np.ndarray], components_df: pd.DataFrame) -> dict[str, np.ndarray]:
    progress = make_progress(pred_masks.items(), total=len(pred_masks), desc="Labeled component masks", unit="image", leave=False, position=2)
    return {image_id: extract_components(mask, min_area=1)[0] for image_id, mask in progress}


def _load_pixel_source(source: str, uq_dir: Path, model_slug: str, slug: str, bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image_ids = [str(x) for x in bundle["image_ids"]]
    if source == "margin":
        maps = np.abs(bundle["prob_maps"] - 0.5)
        return {image_id: maps[idx] for idx, image_id in enumerate(image_ids)}
    if source == "low_confidence":
        maps = 1.0 - bundle["prob_maps"]
        return {image_id: maps[idx] for idx, image_id in enumerate(image_ids)}
    file_map = {
        "entropy": (uq_dir / f"{model_slug}_{slug}_entropy_maps.npz", "entropy_maps"),
        "mi": (uq_dir / f"{model_slug}_{slug}_mi_maps.npz", "mi_maps"),
        "tta": (uq_dir / f"{model_slug}_{slug}_tta_maps.npz", "tta_maps"),
        "shift": (uq_dir / f"{model_slug}_{slug}_shift_maps.npz", "shift_maps"),
    }
    if source not in file_map:
        raise KeyError(f"Unsupported pixel map source: {source}")
    path, key = file_map[source]
    data = np.load(path, allow_pickle=False)
    maps = data[key]
    return {image_id: maps[idx] for idx, image_id in enumerate(image_ids)}


def _pixel_referral_cis(
    per_image_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    return {
        "error_pixel_recall": bootstrap_ci_per_image(per_image_df, value_col="error_pixel_recall", n_boot=n_boot, seed=seed),
        "reviewed_area_pct": bootstrap_ci_per_image(per_image_df, value_col="reviewed_area_pct", n_boot=n_boot, seed=seed),
        "f1_gain_upper_bound": bootstrap_ci_per_image(per_image_df, value_col="f1_gain_upper_bound", n_boot=n_boot, seed=seed),
        "iou_gain_upper_bound": bootstrap_ci_per_image(per_image_df, value_col="iou_gain_upper_bound", n_boot=n_boot, seed=seed),
    }


def _component_referral_cis(
    per_image_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    return {
        "error_recall": bootstrap_ci_per_image(per_image_df, value_col="component_error_recall", n_boot=n_boot, seed=seed),
        "error_pixel_recall": bootstrap_ci_per_image(per_image_df, value_col="error_pixel_recall", n_boot=n_boot, seed=seed),
        "reviewed_area_pct": bootstrap_ci_per_image(per_image_df, value_col="reviewed_area_pct", n_boot=n_boot, seed=seed),
        "f1_gain_upper_bound": bootstrap_ci_per_image(per_image_df, value_col="f1_gain_upper_bound", n_boot=n_boot, seed=seed),
        "iou_gain_upper_bound": bootstrap_ci_per_image(per_image_df, value_col="iou_gain_upper_bound", n_boot=n_boot, seed=seed),
    }


def _paired_referral_stats(
    component_per_image: pd.DataFrame,
    pixel_per_image: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    merged = component_per_image.merge(
        pixel_per_image[["image_id", "error_pixel_recall", "f1_gain_upper_bound", "iou_gain_upper_bound"]],
        on="image_id",
        how="inner",
        suffixes=("_component", "_pixel"),
    )
    stats: dict[str, float] = {}
    for metric in ("error_pixel_recall", "f1_gain_upper_bound", "iou_gain_upper_bound"):
        comp_col = f"{metric}_component"
        pix_col = f"{metric}_pixel"
        valid = merged[["image_id", comp_col, pix_col]].dropna().copy()
        if valid.empty:
            stats[f"{metric}_diff_component_minus_pixel"] = float("nan")
            stats[f"{metric}_diff_ci_lo"] = float("nan")
            stats[f"{metric}_diff_ci_hi"] = float("nan")
            stats[f"{metric}_p_value"] = float("nan")
            stats[f"{metric}_n_pairs"] = 0
            continue
        valid["diff"] = valid[comp_col] - valid[pix_col]
        ci_lo, ci_hi = bootstrap_ci_per_image(valid, value_col="diff", n_boot=n_boot, seed=seed)
        test = paired_wilcoxon(valid, comp_col, pix_col)
        stats[f"{metric}_diff_component_minus_pixel"] = float(valid["diff"].mean())
        stats[f"{metric}_diff_ci_lo"] = ci_lo
        stats[f"{metric}_diff_ci_hi"] = ci_hi
        stats[f"{metric}_p_value"] = test["p_value"]
        stats[f"{metric}_n_pairs"] = test["n_pairs"]
    return stats


def _paired_referral_stats_rows(
    *,
    model_name: str,
    dataset_name: str,
    canonical: str,
    budget: float,
    pixel_source: str,
    component_score: str,
    paired_stats: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in ("error_pixel_recall", "f1_gain_upper_bound", "iou_gain_upper_bound"):
        rows.append(
            {
                "model_name": model_name,
                "dataset": dataset_name,
                "dataset_canonical": canonical,
                "budget": budget,
                "pixel_source": pixel_source,
                "component_score": component_score,
                "metric": metric,
                "component_minus_pixel_mean_diff": paired_stats[f"{metric}_diff_component_minus_pixel"],
                "diff_ci_lo": paired_stats[f"{metric}_diff_ci_lo"],
                "diff_ci_hi": paired_stats[f"{metric}_diff_ci_hi"],
                "p_value": paired_stats[f"{metric}_p_value"],
                "n_pairs": paired_stats[f"{metric}_n_pairs"],
            }
        )
    return rows


def _build_early_budget_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    early = wide_df[np.isclose(wide_df["budget"], 0.01) | np.isclose(wide_df["budget"], 0.03) | np.isclose(wide_df["budget"], 0.05)].copy()
    early = early.sort_values(["dataset", "budget"]).reset_index(drop=True)
    columns = [
        "model_name",
        "dataset",
        "dataset_canonical",
        "budget",
        "pixel_source",
        "component_score",
        "pixel_error_pixel_recall",
        "pixel_error_pixel_recall_ci_lo",
        "pixel_error_pixel_recall_ci_hi",
        "component_error_pixel_recall",
        "component_error_pixel_recall_ci_lo",
        "component_error_pixel_recall_ci_hi",
        "component_error_recall",
        "component_error_recall_ci_lo",
        "component_error_recall_ci_hi",
        "pixel_f1_gain_upper_bound",
        "pixel_f1_gain_upper_bound_ci_lo",
        "pixel_f1_gain_upper_bound_ci_hi",
        "component_f1_gain_upper_bound",
        "component_f1_gain_upper_bound_ci_lo",
        "component_f1_gain_upper_bound_ci_hi",
        "error_pixel_recall_diff_component_minus_pixel",
        "error_pixel_recall_diff_ci_lo",
        "error_pixel_recall_diff_ci_hi",
        "error_pixel_recall_p_value",
        "f1_gain_upper_bound_diff_component_minus_pixel",
        "f1_gain_upper_bound_diff_ci_lo",
        "f1_gain_upper_bound_diff_ci_hi",
        "f1_gain_upper_bound_p_value",
    ]
    return early.reindex(columns=columns)


def _build_whu_main_operational_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    whu_aliases = {"whu-256", "WHU-256"}
    whu = wide_df[wide_df["dataset"].isin(whu_aliases) & (np.isclose(wide_df["budget"], 0.05) | np.isclose(wide_df["budget"], 0.10))].copy()
    whu = whu.sort_values(["dataset", "budget"]).reset_index(drop=True)
    columns = [
        "model_name",
        "dataset",
        "dataset_canonical",
        "budget",
        "pixel_source",
        "component_score",
        "pixel_error_pixel_recall",
        "pixel_error_pixel_recall_ci_lo",
        "pixel_error_pixel_recall_ci_hi",
        "component_error_pixel_recall",
        "component_error_pixel_recall_ci_lo",
        "component_error_pixel_recall_ci_hi",
        "component_error_recall",
        "component_error_recall_ci_lo",
        "component_error_recall_ci_hi",
        "pixel_f1_gain_upper_bound",
        "pixel_f1_gain_upper_bound_ci_lo",
        "pixel_f1_gain_upper_bound_ci_hi",
        "component_f1_gain_upper_bound",
        "component_f1_gain_upper_bound_ci_lo",
        "component_f1_gain_upper_bound_ci_hi",
        "error_pixel_recall_diff_component_minus_pixel",
        "error_pixel_recall_diff_ci_lo",
        "error_pixel_recall_diff_ci_hi",
        "error_pixel_recall_p_value",
        "f1_gain_upper_bound_diff_component_minus_pixel",
        "f1_gain_upper_bound_diff_ci_lo",
        "f1_gain_upper_bound_diff_ci_hi",
        "f1_gain_upper_bound_p_value",
    ]
    return whu.reindex(columns=columns)


if __name__ == "__main__":
    main()

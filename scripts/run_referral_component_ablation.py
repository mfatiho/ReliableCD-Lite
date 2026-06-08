from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running as `python scripts/run_referral_component_ablation.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reliable.components.extraction import extract_components
from reliable.data import canonical_dataset_name, dataset_instance_slug
from reliable.inference.save_maps import load_prediction_bundle
from reliable.referral.component_referral import component_level_referral
from reliable.stats.bootstrap import bootstrap_ci_per_image
from reliable.stats.significance import paired_wilcoxon
from reliable.utils.manifest import write_json_manifest
from reliable.utils.progress import make_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10])
    parser.add_argument("--component-scores", nargs="+", default=["score_crs4", "score_margin", "score_entropy"])
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--components-dir", default="results/components")
    parser.add_argument("--components", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    components_dir = Path(args.components_dir)
    model_slug = args.model.lower()
    slug = dataset_instance_slug(args.dataset)
    canonical = canonical_dataset_name(args.dataset)

    bundle = load_prediction_bundle(baseline_dir / f"{model_slug}_{slug}_predictions.npz")
    image_ids = [str(x) for x in bundle["image_ids"]]
    pred_masks = {image_id: bundle["pred_masks"][idx].astype(np.uint8) for idx, image_id in enumerate(image_ids)}
    gt_masks = {image_id: bundle["gt_masks"][idx].astype(np.uint8) for idx, image_id in enumerate(image_ids)}
    image_area_by_id = {image_id: int(bundle["pred_masks"][idx].size) for idx, image_id in enumerate(image_ids)}

    components_path = _resolve_components_path(args.components, components_dir, model_slug, slug)
    components_df = pd.read_parquet(components_path)
    labeled_masks = _build_labeled_masks(pred_masks)

    rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    cached_per_image: dict[tuple[str, float], pd.DataFrame] = {}

    work_items = [(score_col, budget) for score_col in args.component_scores for budget in args.budgets]
    progress = make_progress(work_items, total=len(work_items), desc=f"{args.model}/{args.dataset} referral ablation", unit="setting")
    for score_col, budget in progress:
        metrics, per_image_df = component_level_referral(
            components_df=components_df,
            score_col=score_col,
            pred_masks=pred_masks,
            gt_masks=gt_masks,
            labeled_masks=labeled_masks,
            image_area_by_id=image_area_by_id,
            review_budget=budget,
            return_per_image=True,
        )
        ci = _component_referral_cis(per_image_df, n_boot=args.n_boot, seed=args.seed)
        cached_per_image[(score_col, budget)] = per_image_df.copy()
        per_image_rows.extend(
            per_image_df.assign(
                model_name=args.model,
                dataset=args.dataset,
                dataset_canonical=canonical,
                component_score=score_col,
                budget=budget,
            ).to_dict(orient="records")
        )
        rows.append(
            {
                "model_name": args.model,
                "dataset": args.dataset,
                "dataset_canonical": canonical,
                "budget": budget,
                "component_score": score_col,
                "error_recall": metrics["error_recall"],
                "error_recall_ci_lo": ci["error_recall"][0],
                "error_recall_ci_hi": ci["error_recall"][1],
                "error_recall_pooled": metrics["error_recall_pooled"],
                "error_pixel_recall": metrics["error_pixel_recall"],
                "error_pixel_recall_ci_lo": ci["error_pixel_recall"][0],
                "error_pixel_recall_ci_hi": ci["error_pixel_recall"][1],
                "error_pixel_recall_pooled": metrics["error_pixel_recall_pooled"],
                "reviewed_area_pct": metrics["reviewed_area_pct"],
                "reviewed_area_pct_ci_lo": ci["reviewed_area_pct"][0],
                "reviewed_area_pct_ci_hi": ci["reviewed_area_pct"][1],
                "reviewed_area_pct_pooled": metrics["reviewed_area_pct_pooled"],
                "f1_gain_upper_bound": metrics["f1_gain_upper_bound"],
                "f1_gain_upper_bound_ci_lo": ci["f1_gain_upper_bound"][0],
                "f1_gain_upper_bound_ci_hi": ci["f1_gain_upper_bound"][1],
                "iou_gain_upper_bound": metrics["iou_gain_upper_bound"],
                "iou_gain_upper_bound_ci_lo": ci["iou_gain_upper_bound"][0],
                "iou_gain_upper_bound_ci_hi": ci["iou_gain_upper_bound"][1],
                "component_budget_overshoot_pct": metrics["budget_overshoot_pct"],
                "reviewed_component_count": metrics["reviewed_component_count"],
            }
        )
    progress.close()

    primary_score = args.component_scores[0]
    for budget in args.budgets:
        primary_df = cached_per_image[(primary_score, budget)]
        for score_col in args.component_scores[1:]:
            compare_df = cached_per_image[(score_col, budget)]
            significance_rows.extend(
                _paired_score_rows(
                    primary_df=primary_df,
                    compare_df=compare_df,
                    model_name=args.model,
                    dataset=args.dataset,
                    canonical=canonical,
                    budget=budget,
                    primary_score=primary_score,
                    compare_score=score_col,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    main_df = pd.DataFrame(rows).sort_values(["budget", "component_score"]).reset_index(drop=True)
    main_df.to_csv(out_path, index=False)
    per_image_path = out_path.with_name(f"{out_path.stem}_per_image.csv")
    pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False)
    significance_path = out_path.with_name(f"{out_path.stem}_significance.csv")
    pd.DataFrame(significance_rows).to_csv(significance_path, index=False)
    primary_table_path = out_path.with_name(f"{out_path.stem}_primary_budgets.csv")
    _build_primary_budget_table(main_df).to_csv(primary_table_path, index=False)
    write_json_manifest(
        out_path.with_suffix(".json"),
        {
            "model_name": args.model,
            "dataset": args.dataset,
            "component_scores": args.component_scores,
            "budgets": args.budgets,
            "outputs": {
                "main_csv": str(out_path),
                "per_image_csv": str(per_image_path),
                "significance_csv": str(significance_path),
                "primary_budgets_csv": str(primary_table_path),
            },
        },
    )


def _build_labeled_masks(pred_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    progress = make_progress(pred_masks.items(), total=len(pred_masks), desc="Ablation labeled masks", unit="image", leave=False, position=1)
    masks = {image_id: extract_components(mask, min_area=1)[0] for image_id, mask in progress}
    progress.close()
    return masks


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


def _paired_score_rows(
    *,
    primary_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    model_name: str,
    dataset: str,
    canonical: str,
    budget: float,
    primary_score: str,
    compare_score: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    merged = primary_df.merge(
        compare_df[["image_id", "component_error_recall", "error_pixel_recall", "f1_gain_upper_bound", "iou_gain_upper_bound"]],
        on="image_id",
        how="inner",
        suffixes=("_primary", "_compare"),
    )
    rows: list[dict[str, object]] = []
    metric_map = {
        "component_error_recall": "component_error_recall",
        "error_pixel_recall": "error_pixel_recall",
        "f1_gain_upper_bound": "f1_gain_upper_bound",
        "iou_gain_upper_bound": "iou_gain_upper_bound",
    }
    for metric_name, base_col in metric_map.items():
        primary_col = f"{base_col}_primary"
        compare_col = f"{base_col}_compare"
        valid = merged[["image_id", primary_col, compare_col]].dropna().copy()
        if valid.empty:
            rows.append(
                {
                    "model_name": model_name,
                    "dataset": dataset,
                    "dataset_canonical": canonical,
                    "budget": budget,
                    "metric": metric_name,
                    "primary_score": primary_score,
                    "compare_score": compare_score,
                    "mean_diff_primary_minus_compare": float("nan"),
                    "diff_ci_lo": float("nan"),
                    "diff_ci_hi": float("nan"),
                    "p_value": float("nan"),
                    "n_pairs": 0,
                }
            )
            continue
        valid["diff"] = valid[primary_col] - valid[compare_col]
        diff_ci_lo, diff_ci_hi = bootstrap_ci_per_image(valid, value_col="diff", n_boot=n_boot, seed=seed)
        test = paired_wilcoxon(valid, primary_col, compare_col)
        rows.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "dataset_canonical": canonical,
                "budget": budget,
                "metric": metric_name,
                "primary_score": primary_score,
                "compare_score": compare_score,
                "mean_diff_primary_minus_compare": float(valid["diff"].mean()),
                "diff_ci_lo": diff_ci_lo,
                "diff_ci_hi": diff_ci_hi,
                "p_value": test["p_value"],
                "n_pairs": test["n_pairs"],
            }
        )
    return rows


def _build_primary_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    focus = df[df["budget"].isin([0.05, 0.10])].copy()
    return focus.sort_values(["budget", "component_score"]).reset_index(drop=True)


if __name__ == "__main__":
    main()

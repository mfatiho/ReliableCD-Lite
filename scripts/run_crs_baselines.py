from __future__ import annotations

import sys
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running as `python scripts/run_crs_baselines.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from reliable.metrics.uncertainty_metrics import component_error_metrics_by_image, component_error_metrics_with_ci
from reliable.scoring.baselines import MAIN_BASELINES, SUPPLEMENTARY_BASELINES, prepare_component_scores
from reliable.scoring.crs import add_mean_mi_or_entropy
from reliable.scoring.zscore import CRSNormalizer
from reliable.stats.bootstrap import bootstrap_ci_per_image
from reliable.stats.significance import paired_wilcoxon
from reliable.utils.progress import make_progress
from reliable.visualization.reports import plot_baseline_metrics


def _compute_one_baseline(
    ds_df: pd.DataFrame,
    dataset: str,
    baseline: str,
    n_boot: int,
    seed: int,
) -> dict | None:
    score_col = f"score_{baseline.lower().replace('-', '')}"
    if score_col not in ds_df.columns:
        return None
    metrics = component_error_metrics_with_ci(ds_df, score_col, n_boot=n_boot, seed=seed)
    metrics["score"] = baseline
    metrics["dataset"] = dataset
    return metrics


def _compute_metrics_per_dataset(
    df: pd.DataFrame,
    baselines: list[str],
    seed: int = 42,
    n_boot: int = 1000,
    desc: str = "CRS baselines",
    n_jobs: int = 1,
) -> list[dict]:
    """Compute per-dataset error metrics with cluster bootstrap CIs for each baseline.

    n_jobs=1  → sequential (default)
    n_jobs=-1 → one thread per CPU (numpy releases the GIL; effective for bootstrap)
    n_jobs=N  → N threads
    """
    datasets = sorted(df["dataset"].dropna().unique()) if "dataset" in df.columns else ["all"]
    ds_dfs = {
        dataset: (df[df["dataset"] == dataset] if dataset != "all" else df)
        for dataset in datasets
    }
    work_items = [(dataset, baseline) for dataset in datasets for baseline in baselines]
    bar = make_progress(total=len(work_items), desc=desc, unit="score")
    rows: list[dict] = []

    if n_jobs != 1:
        max_workers = None if n_jobs == -1 else n_jobs
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_compute_one_baseline, ds_dfs[dataset], dataset, baseline, n_boot, seed): None
                for dataset, baseline in work_items
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    rows.append(result)
                bar.update(1)
    else:
        for dataset, baseline in work_items:
            result = _compute_one_baseline(ds_dfs[dataset], dataset, baseline, n_boot, seed)
            if result is not None:
                rows.append(result)
            bar.update(1)

    bar.close()
    return rows


def _score_column_for_baseline(baseline: str) -> str:
    return f"score_{baseline.lower().replace('-', '')}"


def _fit_normalizer_from_components(path: Path) -> CRSNormalizer:
    reference_df = add_mean_mi_or_entropy(pd.read_parquet(path))
    zscore_columns = [
        "mean_mi_or_entropy",
        "boundary_uncertainty",
        "probability_margin",
        "mean_tta",
        "mean_shift",
    ]
    missing = [col for col in zscore_columns if col not in reference_df.columns]
    if missing:
        raise KeyError(f"Missing component feature columns required to fit CRS normalizer: {missing}")
    normalizer = CRSNormalizer()
    normalizer.fit(reference_df, zscore_columns)
    return normalizer


def _compute_main_significance_per_dataset(
    df: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets = sorted(df["dataset"].dropna().unique()) if "dataset" in df.columns else ["all"]
    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset] if dataset != "all" else df
        for method_a, method_b in comparisons:
            score_a = _score_column_for_baseline(method_a)
            score_b = _score_column_for_baseline(method_b)
            if score_a not in ds_df.columns or score_b not in ds_df.columns:
                continue
            per_image_a = component_error_metrics_by_image(ds_df, score_a).rename(
                columns={"error_auroc": "error_auroc_a", "error_auprc": "error_auprc_a"}
            )
            per_image_b = component_error_metrics_by_image(ds_df, score_b).rename(
                columns={"error_auroc": "error_auroc_b", "error_auprc": "error_auprc_b"}
            )
            merged = per_image_a.merge(per_image_b[["image_id", "error_auroc_b", "error_auprc_b"]], on="image_id", how="inner")
            for metric in ("error_auroc", "error_auprc"):
                col_a = f"{metric}_a"
                col_b = f"{metric}_b"
                valid = merged[["image_id", col_a, col_b]].dropna().copy()
                if valid.empty:
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "method_a": method_a,
                            "method_b": method_b,
                            "mean_diff": float("nan"),
                            "diff_ci_lo": float("nan"),
                            "diff_ci_hi": float("nan"),
                            "statistic": float("nan"),
                            "p_value": float("nan"),
                            "n_pairs": 0,
                        }
                    )
                    continue
                valid["diff"] = valid[col_a] - valid[col_b]
                diff_ci_lo, diff_ci_hi = bootstrap_ci_per_image(valid, value_col="diff", n_boot=n_boot, seed=seed)
                test = paired_wilcoxon(valid, col_a, col_b)
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "mean_diff": float(valid["diff"].mean()),
                        "diff_ci_lo": diff_ci_lo,
                        "diff_ci_hi": diff_ci_hi,
                        "statistic": test["statistic"],
                        "p_value": test["p_value"],
                        "n_pairs": test["n_pairs"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--main-baselines", nargs="*", default=MAIN_BASELINES)
    parser.add_argument("--supplementary-all", action="store_true")
    parser.add_argument(
        "--normalizer-in",
        help="Load a fixed CRS z-score normalizer JSON and apply it instead of fitting on --components.",
    )
    parser.add_argument(
        "--fit-normalizer-from",
        help="Fit the CRS z-score normalizer from this reference component parquet, then apply it to --components.",
    )
    parser.add_argument(
        "--normalizer-out",
        help="Where to save the CRS z-score normalizer JSON. Defaults to <out-dir>/crs_zscore_normalizer.json.",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations for 95%% CI (default 1000)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel threads for baseline computation. -1 = all CPUs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.normalizer_in and args.fit_normalizer_from:
        parser.error("--normalizer-in and --fit-normalizer-from are mutually exclusive.")

    components_path = Path(args.components)
    df = pd.read_parquet(components_path)
    if args.normalizer_in:
        normalizer = CRSNormalizer.load(args.normalizer_in)
    elif args.fit_normalizer_from:
        normalizer = _fit_normalizer_from_components(Path(args.fit_normalizer_from))
    else:
        normalizer = None
    df, normalizer = prepare_component_scores(df, seed=args.seed, normalizer=normalizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(components_path, index=False)
    normalizer_path = Path(args.normalizer_out) if args.normalizer_out else out_dir / "crs_zscore_normalizer.json"
    normalizer.save(normalizer_path)

    col_order = ["score", "dataset", "error_auroc", "auroc_ci_lo", "auroc_ci_hi",
                 "error_auprc", "auprc_ci_lo", "auprc_ci_hi", "random_auprc"]

    main_rows = _compute_metrics_per_dataset(
        df, args.main_baselines, seed=args.seed, n_boot=args.n_boot,
        desc="Main baselines", n_jobs=args.n_jobs,
    )
    main_df = pd.DataFrame(main_rows).reindex(columns=[c for c in col_order if c in pd.DataFrame(main_rows).columns])
    main_df.to_csv(out_dir / "main_baselines_table2.csv", index=False)
    significance_df = _compute_main_significance_per_dataset(
        df,
        comparisons=[("crs4", "margin"), ("crs4", "entropy")],
        n_boot=args.n_boot,
        seed=args.seed,
    )
    significance_df.to_csv(out_dir / "main_baselines_significance.csv", index=False)
    try:
        plot_baseline_metrics(main_df, out_dir / "main_baselines_table2.pdf")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        warnings.warn("matplotlib is not installed; skipped main_baselines_table2.pdf generation.", stacklevel=1)

    if args.supplementary_all:
        supp_rows = _compute_metrics_per_dataset(
            df, SUPPLEMENTARY_BASELINES, seed=args.seed, n_boot=args.n_boot,
            desc="Supplementary baselines", n_jobs=args.n_jobs,
        )
        supp_df = pd.DataFrame(supp_rows).reindex(columns=[c for c in col_order if c in pd.DataFrame(supp_rows).columns])
        supp_df.to_csv(out_dir / "supplementary_full_baselines.csv", index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path
import subprocess
import warnings
from shutil import copy2, which

# Allow running as `python scripts/make_figures.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

from reliable.scoring.baselines import add_ranking_scores
from reliable.stats.correlation import spearman_feature_correlation, spearman_feature_error_correlation
from reliable.utils.progress import make_progress
from reliable.visualization.curves import plot_review_budget_curves
from reliable.visualization.heatmaps import plot_error_correlation_forest, plot_spearman_heatmap
from reliable.visualization.panels import (
    display_dataset_name,
    plot_framework_diagram,
    summarize_budget_labels,
    select_review_case_images,
    select_failure_mode_cases,
    plot_score_separation_panel,
)


FEATURE_COLUMNS = [
    "mean_entropy",
    "mean_tta",
    "mean_shift",
    "boundary_uncertainty",
    "probability_margin",
    "area",
    "compactness",
    "eccentricity",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures for ReliableCD-Lite.")
    parser.add_argument("--components", required=True, help="Parquet file or directory of component parquet files")
    parser.add_argument("--referral-wide", required=True, help="Wide referral CSV from scripts/run_referral.py")
    parser.add_argument("--referral-per-image", default=None, help="Optional per-image referral CSV from scripts/run_referral.py")
    parser.add_argument("--missed-change-dir", default="results/missed_change", help="Directory containing missed-change CSV artifacts")
    parser.add_argument("--feature-correlation", default=None, help="Optional precomputed Spearman correlation CSV")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--model-name", default=None, help="Override model name label used in figure titles (inferred from data if omitted)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components_df = _load_components(args.components)
    referral_df   = pd.read_csv(args.referral_wide)
    model_name = args.model_name or _infer_model_name(components_df, referral_df)
    if str(model_name).lower() != "bit":
        raise ValueError(
            "Paper figure generation is restricted to BIT. "
            "ChangeFormer is reported as a LEVIR-CD table-only sanity check."
        )
    components_df = _filter_components_by_model_name(components_df, model_name)
    components_df = _ensure_score_columns(components_df)
    dataset_labels = [display_dataset_name(str(dataset)) for dataset in components_df["dataset"].dropna().unique()]
    budget_labels = summarize_budget_labels(referral_df.get("budget", pd.Series(dtype=float)))
    referral_per_image_df = _load_referral_per_image(args.referral_per_image, args.referral_wide)
    missed_change_df = _load_missed_change_cases(
        model_name=model_name,
        datasets=[str(dataset) for dataset in components_df["dataset"].dropna().unique()],
        missed_change_dir=args.missed_change_dir,
    )

    model_key = str(model_name).lower()
    pred_masks_path = Path("results/baseline") / f"{model_key}_whu-256_pred_masks.npz"
    prob_maps_path = Path("results/baseline") / f"{model_key}_whu-256_prob_maps.npz"
    runtime_path = Path("results/runtime") / f"{model_key}_levir-256_runtime.csv"
    optional_figure_count = int(pred_masks_path.exists() and prob_maps_path.exists()) + int(runtime_path.exists())

    # Paper figures generated here: framework, budget curves, Spearman
    # heatmap, feature-error correlation, score separation, and optional
    # protocol/cost figures.
    steps = make_progress(total=5 + optional_figure_count, desc="Figure generation", unit="figure")

    # Figure 1 — Framework overview diagram
    framework_path = _render_figure_with_png(
        lambda path: plot_framework_diagram(
            path,
            model_name=model_name,
            dataset_labels=dataset_labels,
            budget_labels=budget_labels,
        ),
        out_dir / "figure1_framework.pdf",
        out_dir,
    )
    steps.update(1)

    # Candidate cases consumed by scripts/make_figure2_panel.py.
    review_case_df = select_review_case_images(referral_per_image_df, missed_change_df)
    candidate_df = pd.concat(
        [
            select_failure_mode_cases(components_df).assign(candidate_source="component_case"),
            review_case_df.assign(candidate_source="review_case"),
        ],
        ignore_index=True,
        sort=False,
    )
    if not candidate_df.empty and "dataset" in candidate_df.columns:
        candidate_df["dataset_display"] = candidate_df["dataset"].astype(str).map(display_dataset_name)
    candidate_df.to_csv(out_dir / "figure2_qualitative_case_candidates.csv", index=False)

    # Figure 3 — Error Recall and F1 Gain vs. Review Budget curves
    review_budget_path = _render_figure_with_png(
        lambda path: plot_review_budget_curves(referral_df, path),
        out_dir / "figure3_review_budget_curves.pdf",
        out_dir,
    )
    steps.update(1)

    # Figure 2 correlation panels (5b/5c) are computed on LEVIR-CD components,
    # the source/validation domain the methodology text describes.
    levir_components = components_df[
        components_df["dataset"].astype(str).isin(["levir-256", "LEVIR-CD"])
    ].copy()
    if levir_components.empty:
        levir_components = components_df

    # Figure 5b — Spearman correlation heatmap among the CRS input signals
    # plus component area (whose negative correlation with shift sensitivity
    # the text uses); morphological features (compactness, eccentricity) are
    # excluded because they are never discussed.
    HEATMAP_FEATURE_COLUMNS = [
        "mean_entropy", "mean_tta", "mean_shift",
        "boundary_uncertainty", "probability_margin", "area",
    ]
    if args.feature_correlation:
        corr_df = pd.read_csv(args.feature_correlation, index_col=0)
    else:
        heatmap_features = [col for col in HEATMAP_FEATURE_COLUMNS if col in levir_components.columns]
        corr_df = spearman_feature_correlation(levir_components.fillna(0.0), heatmap_features)
    keep = [col for col in HEATMAP_FEATURE_COLUMNS if col in corr_df.columns]
    if keep:
        corr_df = corr_df.loc[keep, keep]
    spearman_path = _render_figure_with_png(
        lambda path: plot_spearman_heatmap(corr_df, path),
        out_dir / "figure5b_spearman_supplementary.pdf",
        out_dir,
    )
    steps.update(1)

    # Figure 5c — feature-error correlation forest plot. All candidate
    # features are shown so the exclusion of the morphological ones is visible.
    avail_features = [col for col in FEATURE_COLUMNS if col in levir_components.columns]
    err_corr_df = spearman_feature_error_correlation(
        levir_components.fillna(0.0),
        avail_features,
        n_boot=200,
        seed=42,
    )
    error_corr_path = _render_figure_with_png(
        lambda path: plot_error_correlation_forest(err_corr_df, path),
        out_dir / "figure5c_feature_error_correlation.pdf",
        out_dir,
    )
    steps.update(1)

    # Figure 7 — Score separation (violin/strip) with AUROC per panel
    separation_path = _render_figure_with_png(
        lambda path: plot_score_separation_panel(components_df, path, model_name=model_name),
        out_dir / "figure7_score_separation.pdf",
        out_dir,
    )
    steps.update(1)

    # New figures: protocol walkthrough (one real image) + cost vs. accuracy
    from reliable.visualization.curves import plot_cost_accuracy
    from reliable.visualization.panels import plot_protocol_walkthrough

    if pred_masks_path.exists() and prob_maps_path.exists():
        _render_figure_with_png(
            lambda path: plot_protocol_walkthrough(
                components_df, pred_masks_path, prob_maps_path, path,
                dataset="whu-256", model_name=model_name),
            out_dir / "figure_protocol_walkthrough.pdf",
            out_dir,
        )
        steps.update(1)
    else:
        warnings.warn(
            f"skipped protocol walkthrough: missing {pred_masks_path}", stacklevel=1)

    if runtime_path.exists():
        _render_figure_with_png(
            lambda path: plot_cost_accuracy(
                components_df, pd.read_csv(runtime_path), path,
                model_name=model_name, dataset="levir-256"),
            out_dir / "figure_cost_accuracy.pdf",
            out_dir,
        )
        steps.update(1)
    else:
        warnings.warn(
            f"skipped cost-accuracy figure: missing {runtime_path}", stacklevel=1)

    steps.close()
    print(f"\nAll figures saved to: {out_dir.resolve()}")


def _load_components(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.is_file():
        return pd.read_parquet(target)
    parquet_files = sorted(target.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {target}")
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


def _ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "score_crs4" not in out.columns and "crs4" in out.columns:
        out["score_crs4"] = out["crs4"]
    if "score_crs3" not in out.columns and "crs3" in out.columns:
        out["score_crs3"] = out["crs3"]
    if "score_margin" not in out.columns:
        out["score_margin"] = -out["probability_margin"]
    required = {"score_crs4", "score_margin", "is_error_component"}
    if not required.issubset(out.columns):
        if {"area", "mean_prob", "mean_entropy", "mean_tta", "mean_shift",
                "crs3", "crs4", "probability_margin"}.issubset(out.columns):
            out = add_ranking_scores(out)
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Missing required score columns for figure generation: {sorted(missing)}")
    return out


def _filter_components_by_model_name(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if "model_name" not in df.columns:
        return df
    filtered = df[df["model_name"].astype(str) == str(model_name)].copy()
    if filtered.empty:
        return df
    return filtered


def _infer_model_name(components_df: pd.DataFrame, referral_df: pd.DataFrame) -> str:
    component_names = (
        {str(name) for name in components_df["model_name"].dropna().unique()}
        if "model_name" in components_df.columns
        else set()
    )
    referral_names = (
        {str(name) for name in referral_df["model_name"].dropna().unique()}
        if "model_name" in referral_df.columns
        else set()
    )

    # Prefer the model identity carried by the referral artifact. This avoids
    # mixed titles when a components directory contains multiple model runs but
    # the current figure command targets one referral CSV (for example BIT only).
    if len(referral_names) == 1:
        return next(iter(referral_names))

    overlap = component_names & referral_names
    if len(overlap) == 1:
        return next(iter(overlap))

    if len(component_names) == 1:
        return next(iter(component_names))

    names = referral_names or component_names
    if not names:
        return "BIT"
    return " + ".join(sorted(names))


def _asset_png_path_for_pdf(pdf_path: str | Path, out_dir: str | Path) -> Path:
    pdf_path = _pdf_path_for_target(pdf_path)
    out_dir = Path(out_dir)
    figures_root = Path("results") / "figures"
    try:
        rel_dir = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_dir = Path(out_dir.name) if out_dir.name else Path()
    if rel_dir.parts and rel_dir.parts[-1] == "pdf":
        rel_dir = Path(*rel_dir.parts[:-1]) if len(rel_dir.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_dir / "png" / f"{pdf_path.stem}.png"


def _asset_pdf_path_for_pdf(pdf_path: str | Path, out_dir: str | Path) -> Path:
    pdf_path = _pdf_path_for_target(pdf_path)
    out_dir = Path(out_dir)
    figures_root = Path("results") / "figures"
    try:
        rel_dir = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_dir = Path(out_dir.name) if out_dir.name else Path()
    if rel_dir.parts and rel_dir.parts[-1] == "pdf":
        rel_dir = Path(*rel_dir.parts[:-1]) if len(rel_dir.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_dir / "pdf" / f"{pdf_path.stem}.pdf"


def _asset_pgf_path_for_pdf(pdf_path: str | Path, out_dir: str | Path) -> Path:
    pdf_path = _pdf_path_for_target(pdf_path)
    out_dir = Path(out_dir)
    figures_root = Path("results") / "figures"
    try:
        rel_dir = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_dir = Path(out_dir.name) if out_dir.name else Path()
    if rel_dir.parts and rel_dir.parts[-1] == "pdf":
        rel_dir = Path(*rel_dir.parts[:-1]) if len(rel_dir.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_dir / "pgf" / f"{pdf_path.stem}.pgf"


def _pdf_path_for_target(pdf_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF figure path, got: {pdf_path}")
    if pdf_path.parent.name == "pdf":
        return pdf_path
    return pdf_path.parent / "pdf" / pdf_path.name


def _pgf_path_for_pdf(pdf_path: str | Path) -> Path:
    pdf_path = _pdf_path_for_target(pdf_path)
    parent = pdf_path.parent
    if parent.name == "pdf":
        return parent.parent / "pgf" / f"{pdf_path.stem}.pgf"
    return parent / "pgf" / f"{pdf_path.stem}.pgf"


def _render_figure_with_png(render, pdf_path: str | Path, out_dir: str | Path) -> Path:
    pdf_path = _pdf_path_for_target(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_pdf = Path(render(pdf_path))
    asset_pdf_path = _asset_pdf_path_for_pdf(rendered_pdf, out_dir)
    asset_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(rendered_pdf, asset_pdf_path)
    if _can_render_pgf():
        rendered_pgf = Path(render(_pgf_path_for_pdf(rendered_pdf)))
        _copy_pgf_asset_bundle(rendered_pgf, _asset_pgf_path_for_pdf(rendered_pdf, out_dir))
    else:
        warnings.warn(
            f"xelatex is not installed; skipped PGF export for {rendered_pdf}. "
            "PDF and PNG outputs are still generated.",
            stacklevel=1,
        )
    png_path = _asset_png_path_for_pdf(rendered_pdf, out_dir)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render(png_path)
    return rendered_pdf


def _can_render_pgf() -> bool:
    return which("xelatex") is not None


def _export_png_asset(pdf_path: str | Path, out_dir: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF figure path, got: {pdf_path}")
    png_path = _asset_png_path_for_pdf(pdf_path, out_dir)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if which("pdftoppm") is None:
        warnings.warn(
            f"pdftoppm is not installed; skipped PNG export for {pdf_path}. PDF output is still available.",
            stacklevel=1,
        )
        return png_path
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(png_path.with_suffix(""))],
        check=True,
    )
    return png_path


def _copy_pgf_asset_bundle(src_pgf_path: str | Path, dst_pgf_path: str | Path) -> Path:
    src_pgf_path = Path(src_pgf_path)
    dst_pgf_path = Path(dst_pgf_path)
    dst_pgf_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_pgf_path, dst_pgf_path)
    for companion in src_pgf_path.parent.glob(f"{src_pgf_path.stem}-img*.png"):
        copy2(companion, dst_pgf_path.parent / companion.name)
    return dst_pgf_path


def _load_referral_per_image(explicit_path: str | None, referral_wide_path: str | Path) -> pd.DataFrame:
    if explicit_path:
        return pd.read_csv(explicit_path)
    path = Path(referral_wide_path)
    inferred = path.with_name(path.name.replace("_wide.csv", "_per_image.csv"))
    if inferred.exists():
        return pd.read_csv(inferred)
    return pd.DataFrame()


def _load_missed_change_cases(
    *,
    model_name: str,
    datasets: list[str],
    missed_change_dir: str | Path,
) -> pd.DataFrame:
    model_slug = model_name.lower()
    base_dir = Path(missed_change_dir)
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        path = base_dir / f"{model_slug}_{dataset}_missed_change.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()

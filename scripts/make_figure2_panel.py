"""Generate Figure 2: qualitative failure-mode panel with actual RS imagery.

Reads figure2_qualitative_case_candidates.csv produced by make_figures.py and
renders one row per case (high-risk error, high-risk correct, large-area error)
for each of the three datasets.

Columns per row:
  Before (A) | After (B) | GT mask | Prediction | Probability + component outline

Usage:
  python scripts/make_figure2_panel.py \
      --candidates results/figures/bit/figure2_qualitative_case_candidates.csv \
      --components-dir results/components \
      --baseline-dir results/baseline \
      --uq-dir results/uq \
      --levir-root /home/mfo/Datasets/remote_sensing_cd/LEVIR-CD/test \
      --whu-root /home/mfo/Datasets/remote_sensing_cd/WHU-CD-256/test \
      --dsifn-root /home/mfo/Datasets/remote_sensing_cd/DSIFN-CD/test \
      --out results/figures/bit/figure2_rs_panel.pdf
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import subprocess
import warnings
from pathlib import Path
from shutil import copy2, which

# Allow running as `python scripts/make_figure2_panel.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
from reliable.data.bit_format import discover_split_layout, label_path_for_name
from reliable.utils.experiment import resolve_dataset_root
from reliable.visualization.panels import TEXT_WIDTH_IN, display_dataset_name, paper_style
from scipy.ndimage import label as scipy_label


# ── dataset routing ─────────────────────────────────────────────────────────

_DATASET_SLUG = {
    "levir-256": "levir-256",
    "whu-256":   "whu-256",
    "dsifn-256": "dsifn-256",
}

_DATASET_KEYS = {
    "levir-256": "levir-256",
    "whu-256": "whu-256",
    "dsifn-256": "dsifn-256",
}

_CASE_ORDER   = ["high_risk_error", "high_risk_correct", "large_area_error"]
_CASE_LABELS  = {
    "high_risk_error":   "High-risk error",
    "high_risk_correct": "High-risk correct",
    "large_area_error":  "Large-area error",
}
_DATASET_COLOR = {
    "levir-256": "#2A9D8F",
    "whu-256":   "#D9A030",
    "dsifn-256": "#E76F51",
}

PATCH_SIZE = 256


# ── image-id parsing ─────────────────────────────────────────────────────────

def parse_image_id(image_id: str) -> tuple[str, int | None, int | None]:
    """Return (filename, y_offset, x_offset).  Offsets are None for full images."""
    if "::" not in image_id:
        return image_id, None, None
    filename, coords = image_id.split("::", 1)
    m = re.fullmatch(r"y(\d+)_x(\d+)", coords)
    if not m:
        raise ValueError(f"Unrecognised patch coordinates in image_id: {image_id!r}")
    return filename, int(m.group(1)), int(m.group(2))


# ── image loading ─────────────────────────────────────────────────────────────

def _load_crop(path: Path, y: int | None, x: int | None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    if y is None:
        return arr
    return arr[y : y + PATCH_SIZE, x : x + PATCH_SIZE]


def _load_mask_crop(path: Path, y: int | None, x: int | None) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8)
    if y is None:
        return arr
    return arr[y : y + PATCH_SIZE, x : x + PATCH_SIZE]


def _resolve_panel_split_root(dataset_root: Path, split: str = "test") -> tuple[Path, str, str]:
    split_root, _, folder_a, folder_b = discover_split_layout(dataset_root, split=split)
    return split_root, folder_a, folder_b


def load_imagery(
    dataset: str,
    filename: str,
    y: int | None,
    x: int | None,
    dataset_roots: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (img_A, img_B, gt_mask) as numpy arrays, all PATCH_SIZE×PATCH_SIZE."""
    root = dataset_roots.get(dataset)
    if root is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    split_root, folder_a, folder_b = _resolve_panel_split_root(root, split="test")
    img_a = _load_crop(split_root / folder_a / filename, y, x)
    img_b = _load_crop(split_root / folder_b / filename, y, x)
    gt = _load_mask_crop(label_path_for_name(split_root, filename), y, x)
    return img_a, img_b, gt


# ── NPZ helpers ───────────────────────────────────────────────────────────────

class NpzLookup:
    """Lazy-loads an NPZ file and provides O(1) lookup by image_id."""

    def __init__(self, path: Path, map_key: str) -> None:
        self._path = path
        self._map_key = map_key
        self._data: dict[str, np.ndarray] | None = None

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        npz = np.load(self._path, allow_pickle=True)
        image_ids = npz["image_ids"]
        maps = npz[self._map_key]
        self._data = {str(iid): maps[i] for i, iid in enumerate(image_ids)}

    def get(self, image_id: str) -> np.ndarray | None:
        self._ensure_loaded()
        assert self._data is not None
        return self._data.get(image_id)


# ── component outline ─────────────────────────────────────────────────────────

def _component_outline(pred_mask: np.ndarray, component_id: int | None) -> np.ndarray:
    """Return boolean mask of the border pixels for the target component."""
    if component_id is None:
        return np.zeros_like(pred_mask, dtype=bool)
    labeled, _ = scipy_label(pred_mask)
    target = (labeled == component_id)
    # Erode: a pixel is interior iff all 4-neighbours are also target
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(target, structure=np.ones((3, 3)))
    return target & ~interior


# ── figure rendering ──────────────────────────────────────────────────────────

_COL_TITLES = ["Before (A)", "After (B)", "GT Mask", "Pred. Mask", "Prob. Map"]
N_COLS = 5
RISK_RANK_NOTE = (
    "CRS rank: Top x% indicates the component's dataset-specific CRS percentile "
    "among predicted change components; smaller percentages mean higher estimated risk."
)


def _imshow_gray(ax: plt.Axes, arr: np.ndarray, cmap: str = "gray") -> None:
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")


def load_dataset_crs_scores(
    components_dir: Path,
    model_slug: str,
    datasets: list[str],
) -> dict[str, np.ndarray]:
    """Load per-dataset CRS-4 score arrays for percentile-style risk labels."""
    score_lookup: dict[str, np.ndarray] = {}
    for dataset in datasets:
        path = components_dir / f"{model_slug}_{dataset}_components.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["score_crs4"])
        scores = pd.to_numeric(df["score_crs4"], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if scores.size:
            score_lookup[dataset] = scores
    return score_lookup


def format_risk_rank_label(score: float | None, dataset_scores: np.ndarray | None) -> str | None:
    """Return a compact CRS rank label such as 'CRS rank: Top 3%'."""
    if score is None or dataset_scores is None or dataset_scores.size == 0 or not np.isfinite(score):
        return None
    top_fraction = float(np.mean(dataset_scores >= score))
    top_percent = max(1, int(math.ceil(100.0 * top_fraction)))
    return f"CRS rank: Top {top_percent}%"


def render_panel(
    cases: list[dict],
    out_path: Path,
) -> None:
    n_rows = len(cases)
    with paper_style(out_path):
        # The panel is placed as a full-width figure* and scaled to page
        # height in paper.tex. Author it close to that final page height so
        # labels are not reduced to unreadable sizes by LaTeX.
        fig_h = min(9.1, 0.92 * n_rows + 0.65)
        fig, axes = plt.subplots(
            n_rows, N_COLS,
            figsize=(TEXT_WIDTH_IN, fig_h),
            facecolor="white",
            gridspec_kw={"hspace": 0.02, "wspace": 0.02},
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for col, title in enumerate(_COL_TITLES):
            axes[0, col].set_title(title, fontsize=8.6, fontweight="bold", pad=3)

        prev_dataset = None
        for row_idx, case in enumerate(cases):
            axs = axes[row_idx]
            dataset = case["dataset"]

            row_label = f"{_CASE_LABELS.get(case['case_type'], case['case_type'])}"
            if dataset != prev_dataset:
                row_label = f"{display_dataset_name(dataset)}\n{row_label}"
            prev_dataset = dataset

            axs[0].set_ylabel(
                row_label,
                rotation=0,
                ha="right",
                va="center",
                fontsize=9.0,
                labelpad=6,
                color=_DATASET_COLOR.get(dataset, "black"),
            )

            img_a = case["img_a"]
            img_b = case["img_b"]
            gt = case["gt"]
            pred = case["pred"]
            prob = case["prob"]
            outline = case["outline"]
            risk_rank_label = case.get("risk_rank_label")

            axs[0].imshow(img_a, interpolation="nearest")
            axs[1].imshow(img_b, interpolation="nearest")
            _imshow_gray(axs[2], gt.astype(float))
            _imshow_gray(axs[3], pred.astype(float))
            if outline.any():
                rgba = np.zeros((*pred.shape, 4), dtype=np.float32)
                rgba[outline, 0] = 1.0
                rgba[outline, 3] = 0.8
                axs[3].imshow(rgba, interpolation="nearest")

            axs[4].imshow(prob, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, interpolation="nearest")
            if outline.any():
                rgba2 = np.zeros((*prob.shape, 4), dtype=np.float32)
                rgba2[outline, 2] = 1.0
                rgba2[outline, 3] = 0.9
                axs[4].imshow(rgba2, interpolation="nearest")
            if risk_rank_label:
                axs[4].text(
                    2,
                    2,
                    risk_rank_label,
                    fontsize=6.5,
                    color="white",
                    va="top",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.18", "fc": "black", "alpha": 0.55, "lw": 0},
                )

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            color = _DATASET_COLOR.get(dataset, "#888888")
            for ax in axs:
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_color(color)
                ax.spines["left"].set_linewidth(2.7)

        fig.text(
            0.5,
            0.012,
            RISK_RANK_NOTE,
            ha="center",
            va="bottom",
            fontsize=6.6,
            color="#4B5563",
        )
        fig.subplots_adjust(left=0.08, right=1.0, top=0.975, bottom=0.045)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
    print(f"Saved: {out_path}")


def render_panel_outputs(cases: list[dict], pdf_path: Path) -> None:
    pdf_path = pdf_output_path(pdf_path)
    render_panel(cases, pdf_path)
    rendered_asset_pdf = asset_pdf_path_for_pdf(pdf_path)
    rendered_asset_pdf.parent.mkdir(parents=True, exist_ok=True)
    copy2(pdf_path, rendered_asset_pdf)
    if can_render_pgf():
        rendered_pgf = pgf_path_for_pdf(pdf_path)
        render_panel(cases, rendered_pgf)
        copy_pgf_asset_bundle(rendered_pgf, asset_pgf_path_for_pdf(pdf_path))
    else:
        warnings.warn(
            f"xelatex is not installed; skipped PGF export for {pdf_path}. "
            "PDF and PNG outputs are still generated.",
            stacklevel=1,
        )
    render_panel(cases, asset_png_path_for_pdf(pdf_path))


def can_render_pgf() -> bool:
    return which("xelatex") is not None


def asset_png_path_for_pdf(pdf_path: str | Path) -> Path:
    pdf_path = pdf_output_path(pdf_path)
    figures_root = Path("results") / "figures"
    try:
        rel_path = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_path = Path(pdf_path.parent.name) if pdf_path.parent.name else Path()
    if rel_path.parts and rel_path.parts[-1] == "pdf":
        rel_path = Path(*rel_path.parts[:-1]) if len(rel_path.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_path / "png" / f"{pdf_path.stem}.png"


def asset_pdf_path_for_pdf(pdf_path: str | Path) -> Path:
    pdf_path = pdf_output_path(pdf_path)
    figures_root = Path("results") / "figures"
    try:
        rel_path = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_path = Path(pdf_path.parent.name) if pdf_path.parent.name else Path()
    if rel_path.parts and rel_path.parts[-1] == "pdf":
        rel_path = Path(*rel_path.parts[:-1]) if len(rel_path.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_path / "pdf" / f"{pdf_path.stem}.pdf"


def asset_pgf_path_for_pdf(pdf_path: str | Path) -> Path:
    pdf_path = pdf_output_path(pdf_path)
    figures_root = Path("results") / "figures"
    try:
        rel_path = pdf_path.parent.resolve().relative_to(figures_root.resolve())
    except ValueError:
        rel_path = Path(pdf_path.parent.name) if pdf_path.parent.name else Path()
    if rel_path.parts and rel_path.parts[-1] == "pdf":
        rel_path = Path(*rel_path.parts[:-1]) if len(rel_path.parts) > 1 else Path()
    return Path("docs") / "assets" / "figures" / rel_path / "pgf" / f"{pdf_path.stem}.pgf"


def pdf_output_path(pdf_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF panel path, got: {pdf_path}")
    if pdf_path.parent.name == "pdf":
        return pdf_path
    return pdf_path.parent / "pdf" / pdf_path.name


def pgf_path_for_pdf(pdf_path: str | Path) -> Path:
    pdf_path = pdf_output_path(pdf_path)
    parent = pdf_path.parent
    if parent.name == "pdf":
        return parent.parent / "pgf" / f"{pdf_path.stem}.pgf"
    return parent / "pgf" / f"{pdf_path.stem}.pgf"


def export_png_asset(pdf_path: str | Path) -> Path:
    pdf_path = pdf_output_path(pdf_path)
    png_path = asset_png_path_for_pdf(pdf_path)
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


def copy_pgf_asset_bundle(src_pgf_path: str | Path, dst_pgf_path: str | Path) -> Path:
    src_pgf_path = Path(src_pgf_path)
    dst_pgf_path = Path(dst_pgf_path)
    dst_pgf_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(src_pgf_path, dst_pgf_path)
    for companion in src_pgf_path.parent.glob(f"{src_pgf_path.stem}-img*.png"):
        copy2(companion, dst_pgf_path.parent / companion.name)
    return dst_pgf_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2 RS imagery panel.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--candidates", default="results/figures/bit/figure2_qualitative_case_candidates.csv")
    parser.add_argument("--components-dir", default="results/components")
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--uq-dir", default="results/uq")
    parser.add_argument("--model-slug", default="bit", help="Model slug prefix for NPZ files (default: bit)")
    parser.add_argument("--out", default="results/figures/bit/figure2_rs_panel.pdf")
    args = parser.parse_args()

    cand_df = pd.read_csv(args.candidates)
    # Only component_case rows; only the three primary case types
    cand_df = cand_df[
        (cand_df["candidate_source"] == "component_case") &
        (cand_df["case_type"].isin(_CASE_ORDER))
    ].copy()

    dataset_roots = {
        dataset: resolve_dataset_root(config_key, args.config)
        for dataset, config_key in _DATASET_KEYS.items()
    }
    baseline_dir = Path(args.baseline_dir)
    components_dir = Path(args.components_dir)
    uq_dir       = Path(args.uq_dir)
    slug         = args.model_slug

    # Pre-load NPZ lookups per dataset
    npz_pred: dict[str, NpzLookup]  = {}
    npz_prob: dict[str, NpzLookup]  = {}
    npz_entropy: dict[str, NpzLookup] = {}

    datasets_present = cand_df["dataset"].unique().tolist()
    dataset_crs_scores = load_dataset_crs_scores(components_dir, slug, datasets_present)
    for ds in datasets_present:
        ds_slug = _DATASET_SLUG.get(ds, ds)
        pred_path    = baseline_dir / f"{slug}_{ds_slug}_pred_masks.npz"
        prob_path    = baseline_dir / f"{slug}_{ds_slug}_prob_maps.npz"
        entropy_path = uq_dir / f"{slug}_{ds_slug}_entropy_maps.npz"
        if pred_path.exists():
            npz_pred[ds]    = NpzLookup(pred_path, "pred_masks")
        if prob_path.exists():
            npz_prob[ds]    = NpzLookup(prob_path, "prob_maps")
        if entropy_path.exists():
            npz_entropy[ds] = NpzLookup(entropy_path, "entropy_maps")

    # Build ordered list of cases
    DATASET_ORDER_LIST = ["levir-256", "whu-256", "dsifn-256"]
    cases_ordered: list[dict] = []

    for ds in DATASET_ORDER_LIST:
        if ds not in set(cand_df["dataset"]):
            continue
        ds_rows = cand_df[cand_df["dataset"] == ds]
        for case_type in _CASE_ORDER:
            row = ds_rows[ds_rows["case_type"] == case_type]
            if row.empty:
                continue
            row = row.iloc[0]
            image_id = str(row["image_id"])
            filename, y_off, x_off = parse_image_id(image_id)
            component_id = None
            if pd.notna(row.get("component_id")):
                try:
                    component_id = int(float(row["component_id"]))
                except (ValueError, TypeError):
                    pass

            try:
                img_a, img_b, gt = load_imagery(
                    ds, filename, y_off, x_off, dataset_roots,
                )
            except FileNotFoundError as exc:
                print(f"  [WARN] image not found for {image_id}: {exc}")
                continue

            pred_arr = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            if ds in npz_pred:
                p = npz_pred[ds].get(image_id)
                if p is not None:
                    pred_arr = (p > 0).astype(np.uint8)

            prob_arr = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            if ds in npz_prob:
                p = npz_prob[ds].get(image_id)
                if p is not None:
                    prob_arr = p.astype(np.float32)

            outline = _component_outline(pred_arr, component_id)

            crs_score = None
            if "score_crs4" in row and pd.notna(row["score_crs4"]):
                crs_score = float(row["score_crs4"])
            risk_rank_label = format_risk_rank_label(crs_score, dataset_crs_scores.get(ds))

            cases_ordered.append(
                {
                    "dataset":      ds,
                    "case_type":    str(row["case_type"]),
                    "image_id":     image_id,
                    "component_id": component_id,
                    "img_a":        img_a,
                    "img_b":        img_b,
                    "gt":           gt,
                    "pred":         pred_arr,
                    "prob":         prob_arr,
                    "outline":      outline,
                    "risk_rank_label": risk_rank_label,
                }
            )

    if not cases_ordered:
        print("No cases to render — check that candidate CSV and image directories are correct.")
        return

    render_panel_outputs(cases_ordered, Path(args.out))
    print(f"\nRendered {len(cases_ordered)} cases to {args.out}")


if __name__ == "__main__":
    main()

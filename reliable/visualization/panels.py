from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Shared publication-ready style ──────────────────────────────────────────
# Single source of truth for all figure modules.

WHITE = "#FFFFFF"  # Figure, axes, and saved-output background.
BLACK = "#000000"  # Primary high-contrast text color.
INK = "#1B2430"  # Dark title/header color used in schematic figures.
BODY_TEXT = "#2B3947"  # Main body text color inside diagram boxes.
MUTED_TEXT = "#5B6B7A"  # Secondary annotation text color.
FOOTNOTE_TEXT = "#444444"  # Footnote and low-emphasis caption color.
GRID_ALPHA = 0.18  # Light grid opacity that remains visible after PDF scaling.
BAR_EDGE = "#23313F"  # Shared outline color for bar charts.
DATASET_FALLBACK_COLOR = "#6B7280"  # Neutral color for unknown dataset keys.
FRAME_EDGE = "#C7D0D9"  # Neutral panel-frame color for image grids.
SUBTLE_SPINE = "#9CA3AF"  # Low-contrast axis spine and random-baseline color.
LEGEND_FRAME = "#E5E7EB"  # Thin legend-box border color.

SAVEFIG_DPI = 300  # Raster resolution for publication exports.
FIGURE_DPI = 150  # Interactive/default figure canvas resolution.
DEFAULT_PAD_INCHES = 0.03  # Tight export padding for most figures.
FRAMEWORK_PAD_INCHES = 0.02  # Smaller padding for the full-page framework diagram.

AXIS_LABEL_SIZE = 9  # Default axis-label point size.
AXIS_TITLE_SIZE = 9.5  # Default axis-title point size.
TICK_LABEL_SIZE = 8  # Default tick-label point size.
LEGEND_FONT_SIZE = 7.5  # Default legend font size.
TITLE_SIZE = 9.2  # Compact subplot title size used in dashboards.
SMALL_TITLE_SIZE = 7.6  # Small title size used above qualitative image panels.

DEFAULT_FOCUS_BUDGETS = (0.05, 0.10)  # Main referral budgets highlighted in summary panels.
DEFAULT_BUDGET_LABELS = "1 / 3 / 5 / 10 / 20 %"  # Default budget grid shown in Figure 1.
DEFAULT_MIN_COMPONENT_AREA = 16  # Component-extraction min area mirrored in walkthrough matching.
DEFAULT_PATCH_SIZE_PX = 256.0  # Expected tile edge length used when scoring candidate walkthrough images.
ENTROPY_EPS = 1e-6  # Numeric clamp for binary entropy in the qualitative walkthrough.

PAPER_RC_PARAMS: dict[str, object] = {
    "savefig.dpi": SAVEFIG_DPI,  # Resolution used by matplotlib savefig.
    "font.family": "serif",  # Publication style uses serif fonts.
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],  # Preferred serif fallback chain.
    "mathtext.fontset": "dejavuserif",  # Match math labels to the serif text style.
    "axes.spines.top": False,  # Remove top spine for cleaner compact plots.
    "axes.spines.right": False,  # Remove right spine for cleaner compact plots.
    "axes.labelsize": AXIS_LABEL_SIZE,  # Default axis-label size.
    "axes.titlesize": AXIS_TITLE_SIZE,  # Default axes-title size.
    "xtick.labelsize": TICK_LABEL_SIZE,  # Default x tick-label size.
    "ytick.labelsize": TICK_LABEL_SIZE,  # Default y tick-label size.
    "legend.frameon": False,  # Default legends are frameless unless a plot opts in.
    "legend.fontsize": LEGEND_FONT_SIZE,  # Default legend text size.
    "figure.facecolor": WHITE,  # Figure canvas background.
    "axes.facecolor": WHITE,  # Axes background.
    "savefig.facecolor": WHITE,  # Saved-file background.
    "figure.dpi": FIGURE_DPI,  # Default figure display DPI.
    "text.color": BLACK,  # Default text color.
    "axes.labelcolor": BLACK,  # Default axis-label color.
    "xtick.color": BLACK,  # Default x tick color.
    "ytick.color": BLACK,  # Default y tick color.
}

PAPER_PGF_RC_PARAMS: dict[str, object] = {
    "text.usetex": False,  # Keep PGF export independent of a full LaTeX text renderer.
    "pgf.texsystem": "xelatex",  # Unicode-capable TeX engine for PGF output.
    "pgf.rcfonts": False,  # Respect the explicit font settings above.
    "pgf.preamble": "\n".join(  # Minimal math/font preamble used by PGF exports.
        [
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath,amssymb}",
        ]
    ),
}


def paper_style(output_path: str | Path | None = None):
    """Return matplotlib rc_context using the shared publication parameters."""
    import matplotlib as mpl

    rc_params: dict[str, Any] = dict(PAPER_RC_PARAMS)
    if output_path is not None and Path(output_path).suffix.lower() == ".pgf":
        rc_params.update(PAPER_PGF_RC_PARAMS)
    return mpl.rc_context(rc_params)


# ── Figure geometry matched to the two-column IEEE layout ───────────────────
# Generating each figure at its real on-page width is what keeps fonts
# readable. The LaTeX ``\resizebox{<width>}{!}{...}`` wrapper scales the whole
# figure -- text included -- by ``<width> / figure_width``. A figure authored
# at 14 in but placed in a 3.5 in column is scaled to ~0.25x, so a 9 pt label
# prints at ~2.3 pt. Sizing each figure to COLUMN_WIDTH_IN (single-column
# ``figure``) or TEXT_WIDTH_IN (two-column ``figure*``) makes that wrapper a
# ~1.0x no-op, so the rc-param point sizes are the true on-page sizes.
COLUMN_WIDTH_IN = 3.5  # \columnwidth: single-column ``figure`` width.
TEXT_WIDTH_IN = 7.16  # \textwidth: two-column ``figure*`` width.


def paper_figsize(width: str = "column", aspect: float = 0.62) -> tuple[float, float]:
    """Return a figure size matched to the final on-page placement width.

    Parameters
    ----------
    width: ``"column"`` for a single-column ``figure`` (\\columnwidth) or
        ``"text"`` for a two-column ``figure*`` (\\textwidth).
    aspect: height / width ratio.
    """
    base = TEXT_WIDTH_IN if width == "text" else COLUMN_WIDTH_IN
    return (base, base * aspect)


DATASET_ORDER = ["levir-256", "whu-256", "dsifn-256"]  # Canonical order used in paper figures.
DATASET_COLORS = {
    "levir-256": "#2A9D8F",  # LEVIR-CD in-domain dataset color.
    "whu-256": "#D9A030",  # WHU-CD near-domain stress dataset color.
    "dsifn-256": "#E76F51",  # DSIFN-256 far-domain stress dataset color.
    "LEVIR-256": "#2A9D8F",  # Uppercase alias for LEVIR-CD.
    "WHU-256": "#D9A030",  # Uppercase alias for WHU-CD.
    "DSIFN-256": "#E76F51",  # Uppercase alias for DSIFN-256.
}
DATASET_DISPLAY: dict[str, str] = {
    "dsifn-256": "DSIFN-CD†",  # Paper label for patched DSIFN stress condition.
    "DSIFN-256": "DSIFN-CD†",  # Uppercase alias for patched DSIFN stress condition.
    "levir-256": "LEVIR-CD",  # Paper label for LEVIR-CD.
    "LEVIR-256": "LEVIR-CD",  # Uppercase alias for LEVIR-CD.
    "whu-256": "WHU-CD",  # Paper label for WHU-CD.
    "WHU-256": "WHU-CD",  # Uppercase alias for WHU-CD.
}
DSIFN_FOOTNOTE = "† DSIFN-CD: far-domain stress condition; BIT trained on LEVIR-CD only. High error rate reflects domain gap, not method failure."  # Required DSIFN framing note.


SIGNAL_COLORS = {
    "entropy": "#457B9D",  # Predictive entropy bar/curve color.
    "tta": "#E9C46A",  # Test-time augmentation disagreement color.
    "shift": "#E76F51",  # Output-level shift sensitivity color.
    "shift_light": "#F4A261",  # Lighter shift color when paired with error-ratio red.
    "error_ratio": "#E63946",  # Error-pixel-ratio diagnostic color.
    "margin": "#E76F51",  # Margin baseline color in score-separation plots.
    "crs4": "#1D3557",  # CRS-4 main score color.
    "crs3": "#2A9D8F",  # CRS-3 ablation score color.
    "pixel_referral": "#D9453D",  # Pixel-referral reviewed-pixel overlay color.
    "component_referral": "#33548C",  # Component-referral reviewed-region overlay color.
}

SCORE_LINE_SPECS = [
    ("score_crs4", "CRS-4", SIGNAL_COLORS["crs4"], "-", 2.1),  # Full CRS score curve.
    ("score_crs3", "CRS-3", SIGNAL_COLORS["crs3"], "-", 1.8),  # Fast CRS ablation curve.
    ("score_margin", "Margin", SIGNAL_COLORS["margin"], "--", 1.5),  # Margin baseline curve.
    ("score_entropy", "Entropy", "#7B4FA3", "--", 1.5),  # Entropy baseline curve.
]

FRAMEWORK_COLORS = {
    "frozen": ("#E9EDF1", "#8A99A8"),  # Frozen detector stages in Figure 1.
    "layer_i": ("#D6E0F1", SIGNAL_COLORS["component_referral"]),  # Component-level protocol band.
    "layer_ii": ("#D9ECE4", "#2F8A73"),  # CRS scoring band.
    "layer_iii": ("#FBEBCB", "#AE7E1B"),  # Uncertainty-signal band.
    "base_zone": ("#F3F5F7", "#D3DAE1"),  # Background behind frozen-base stages.
    "layer_zone": ("#F1F4FA", "#C7D2E5"),  # Background behind ReliableCD-Lite stages.
    "scope": ("#FBF3F2", "#D9B6B3", "#B5564E"),  # Scope-warning fill, edge, and badge color.
}

FRAMEWORK_BANDS = {
    "pipeline": (0.700, 0.190),  # Pipeline band y-position and height.
    "layer_i": (0.466, 0.186),  # Primary component-level protocol band.
    "layer_ii": (0.279, 0.142),  # CRS scoring band.
    "layer_iii": (0.082, 0.146),  # Uncertainty-signal band.
    "scope": (0.012, 0.052),  # Bottom limitation band.
}

FRAMEWORK_STAGE_XS = [0.026, 0.218, 0.430, 0.622, 0.814]  # Left x-position for each pipeline stage.
FRAMEWORK_STAGE_W = 0.164  # Width of each pipeline stage box.
FRAMEWORK_STAGE_H = 0.128  # Height of each pipeline stage box.
FRAMEWORK_STAGE_Y = 0.716  # Bottom y-position for all pipeline stage boxes.
FROZEN_ZONE_LABEL = "#71808E"  # Text color for the frozen-base zone label.
FLOW_ARROW = "#5A6B7C"  # Arrow color connecting pipeline stages.
UNREVIEWED_RISK_EDGE = "#9AA7B3"  # Outline color for risk-strip components outside budget.
SIGNAL_CHIP_FILL = "#FEF8EA"  # Fill color for the uncertainty-signal chips.

RISK_STRIP_COLORS = [
    "#C8443D",  # Highest-risk component swatch.
    "#D86B3C",  # High-risk component swatch.
    "#DF8F3F",  # Medium-high-risk component swatch.
    "#DCAF49",  # Medium-risk component swatch.
    "#BCC158",  # Medium-low-risk component swatch.
    "#8AB46A",  # Low-risk component swatch.
    "#5DA081",  # Lowest-risk component swatch.
]
RISK_STRIP_REVIEWED = 3  # Number of example components shown inside the review budget.

RGB_PREDICTED_COMPONENT = (0.81, 0.88, 0.95)  # Light blue fill for predicted components.
RGB_UNREVIEWED_COMPONENT = (0.90, 0.92, 0.94)  # Grey fill for unreviewed components.
RGB_REVIEWED_COMPONENT = (0.20, 0.45, 0.72)  # Blue fill for component-referral selections.
RGB_CHANGE_CONTEXT = (0.93, 0.94, 0.95)  # Pale context fill for predicted change pixels.
RGB_REVIEWED_PIXEL = (0.85, 0.27, 0.24)  # Red fill for pixel-referral selections.
COMPONENT_CONTOUR = "#3A4654"  # Dark contour line around predicted components.
PIXEL_REFERRAL_TITLE = "#9A3B2F"  # Dark red title color for pixel-referral panel.
SIGNAL_CHIP_TEXT = "#4A3A1C"  # Brown text color inside signal chips.
SCOPE_TEXT = "#5A4140"  # Dark red-brown text color inside the scope warning.
RISK_CMAP_NAME = "RdYlGn_r"  # Green-to-red colormap for increasing CRS risk.
RISK_CMAP_LOW = 0.12  # Lower colormap crop to avoid overly pale low-risk colors.
RISK_CMAP_SPAN = 0.80  # Colormap span used for per-component risk shading.
COLORBAR_INSET = [0.15, -0.11, 0.70, 0.045]  # CRS colorbar position inside its panel.


def display_dataset_name(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def _dataset_color(dataset: object) -> str:
    """Return the shared paper color for a dataset key."""
    return DATASET_COLORS.get(str(dataset), DATASET_FALLBACK_COLOR)


def _ensure_output_path(out_path: str | Path) -> Path:
    """Create the output directory and return a normalized Path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: Any, out_path: str | Path, *, pad_inches: float = DEFAULT_PAD_INCHES) -> Path:
    """Save a figure with the repository's shared publication settings."""
    path = _ensure_output_path(out_path)
    fig.savefig(path, bbox_inches="tight", pad_inches=pad_inches, facecolor=WHITE)
    return path


def _set_panel_title(ax: Any, title: str, *, color: str = BLACK, fontsize: float = TITLE_SIZE) -> None:
    """Apply the common bold subplot title style."""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", color=color)


def _style_y_grid(ax: Any) -> None:
    """Apply the common light y-axis grid used in compact bar panels."""
    ax.grid(True, axis="y", alpha=GRID_ALPHA)


def summarize_budget_labels(
    budgets: list[float] | pd.Series | np.ndarray,
    *,
    max_items: int = 5,
) -> str:
    vals = sorted({round(float(b) * 100.0, 6) for b in budgets if pd.notna(b)})
    if not vals:
        return "n/a"
    labels = [f"{int(v) if float(v).is_integer() else v:g}%" for v in vals[:max_items]]
    if len(vals) > max_items:
        labels.append("...")
    return " / ".join(labels)


def select_focus_budgets(
    budgets: list[float] | pd.Series | np.ndarray,
    *,
    preferred: tuple[float, ...] = DEFAULT_FOCUS_BUDGETS,
    max_items: int = 2,
) -> list[float]:
    available = sorted({float(b) for b in budgets if pd.notna(b)})
    if not available:
        return []
    chosen: list[float] = [b for b in preferred if any(np.isclose(b, a) for a in available)]
    if chosen:
        return chosen[:max_items]
    if len(available) <= max_items:
        return available
    return [available[0], available[-1]]


def _ordered_datasets(values: list[str] | pd.Index | np.ndarray) -> list[str]:
    present = [str(v) for v in values]
    present_set = set(present)
    ordered = [dataset for dataset in DATASET_ORDER if dataset in present_set]
    extras = sorted(present_set - set(ordered))
    return ordered + extras


def select_failure_mode_cases(components_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in components_df.groupby("dataset", dropna=False, sort=False):
        used_keys: set[tuple[str, int]] = set()
        candidate_specs = [
            (
                "high_risk_error",
                group[group["is_error_component"].astype(bool)].sort_values(
                    ["score_crs4", "error_pixel_ratio", "area"],
                    ascending=[False, False, False],
                ),
                "High-risk error",
            ),
            (
                "high_risk_correct",
                group[~group["is_error_component"].astype(bool)].sort_values(
                    ["score_crs4", "area"],
                    ascending=[False, False],
                ),
                "High-risk correct",
            ),
            (
                "large_area_error",
                group[group["is_error_component"].astype(bool)].sort_values(
                    ["area", "score_crs4", "error_pixel_ratio"],
                    ascending=[False, False, False],
                ),
                "Large-area error",
            ),
        ]
        for case_type, candidates, display_label in candidate_specs:
            selected_row = None
            for row in candidates.itertuples(index=False):
                key = (str(row.image_id), int(row.component_id))
                if key not in used_keys:
                    selected_row = row
                    used_keys.add(key)
                    break
            if selected_row is None:
                continue
            rows.append(
                {
                    "dataset": str(dataset),
                    "case_type": case_type,
                    "case_label": display_label,
                    "image_id": str(selected_row.image_id),
                    "component_id": int(selected_row.component_id),
                    "is_error_component": bool(selected_row.is_error_component),
                    "score_crs4": float(selected_row.score_crs4),
                    "best_gt_iou": float(0.0 if pd.isna(selected_row.best_gt_iou) else selected_row.best_gt_iou),
                    "error_pixel_ratio": float(0.0 if pd.isna(selected_row.error_pixel_ratio) else selected_row.error_pixel_ratio),
                    "area": int(selected_row.area),
                }
            )
    return pd.DataFrame(rows)


def select_review_case_images(
    referral_per_image_df: pd.DataFrame,
    missed_change_df: pd.DataFrame | None = None,
    *,
    budget: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if referral_per_image_df.empty:
        return pd.DataFrame(rows)
    focus = referral_per_image_df[np.isclose(referral_per_image_df["budget"], budget)].copy()
    if focus.empty:
        return pd.DataFrame(rows)
    for dataset, group in focus.groupby("dataset", dropna=False, sort=False):
        pivot = group.pivot_table(
            index="image_id",
            columns="method",
            values=["error_pixel_recall", "f1_gain_upper_bound", "iou_gain_upper_bound"],
            aggfunc="mean",
        )
        if ("error_pixel_recall", "component") not in pivot.columns or ("error_pixel_recall", "pixel") not in pivot.columns:
            continue
        pivot = pivot.copy()
        pivot[("diff", "error_pixel_recall")] = (
            pivot[("error_pixel_recall", "component")] - pivot[("error_pixel_recall", "pixel")]
        )
        pivot = pivot.dropna().reset_index()
        pivot.columns = [
            col if isinstance(col, str) else "_".join(str(part) for part in col if part)
            for col in pivot.columns.to_flat_index()
        ]
        if pivot.empty:
            continue
        component_better = pivot.sort_values("diff_error_pixel_recall", ascending=False).iloc[0]
        pixel_better = pivot.sort_values("diff_error_pixel_recall", ascending=True).iloc[0]
        rows.extend(
            [
                {
                    "dataset": str(dataset),
                    "case_type": "component_better",
                    "case_label": "Component better",
                    "image_id": str(component_better["image_id"]),
                    "budget": budget,
                    "component_error_pixel_recall": float(component_better["error_pixel_recall_component"]),
                    "pixel_error_pixel_recall": float(component_better["error_pixel_recall_pixel"]),
                    "component_f1_gain_upper_bound": float(component_better["f1_gain_upper_bound_component"]),
                    "pixel_f1_gain_upper_bound": float(component_better["f1_gain_upper_bound_pixel"]),
                },
                {
                    "dataset": str(dataset),
                    "case_type": "pixel_better",
                    "case_label": "Pixel better",
                    "image_id": str(pixel_better["image_id"]),
                    "budget": budget,
                    "component_error_pixel_recall": float(pixel_better["error_pixel_recall_component"]),
                    "pixel_error_pixel_recall": float(pixel_better["error_pixel_recall_pixel"]),
                    "component_f1_gain_upper_bound": float(pixel_better["f1_gain_upper_bound_component"]),
                    "pixel_f1_gain_upper_bound": float(pixel_better["f1_gain_upper_bound_pixel"]),
                },
            ]
        )

        if missed_change_df is None or missed_change_df.empty:
            continue
        missed_group = missed_change_df[
            (missed_change_df["dataset"] == dataset) & missed_change_df["is_missed"].astype(bool)
        ].copy()
        if missed_group.empty:
            continue
        missed_case = missed_group.sort_values(["gt_area", "pred_overlap_ratio"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "dataset": str(dataset),
                "case_type": "complete_miss",
                "case_label": "Complete miss",
                "image_id": str(missed_case["image_id"]),
                "budget": budget,
                "gt_area": float(missed_case["gt_area"]),
                "pred_overlap_ratio": float(missed_case["pred_overlap_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def plot_framework_diagram(
    out_path: str | Path,
    model_name: str = "BIT",
    *,
    dataset_labels: list[str] | None = None,
    budget_labels: str | None = None,
    journal: bool = False,
) -> Path:
    """Render the framework overview figure (Figure 1).

    The figure tells the paper's story in four stacked sections, drawn top to
    bottom in axes-fraction coordinates (0..1):

        1. title      -- name, one-line claim, "training-free" badge
        2. pipeline   -- the five-stage data flow, split into a frozen base
                         (stages 1-2) and the ReliableCD-Lite layer (3-5)
        3. hierarchy  -- the same layer shown as three bands:
                         (i) component-level protocol  [primary contribution],
                         (ii) CRS scoring, (iii) uncertainty signals
        4. scope      -- the observable-set limitation

    Layout is intentionally hand-placed (not data-driven) so the diagram stays
    stable; every magic number lives in the ``Layout`` block or in the small
    section helpers below.

    Parameters
    ----------
    out_path:       file to write (``.pdf`` / ``.png`` / ``.pgf``).
    model_name:     label used for the frozen detector (stage 2).
    dataset_labels: accepted for API compatibility; not drawn in this layout.
    budget_labels:  review-budget grid shown in layer (i); falls back to the
                    paper's default grid when omitted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    _ = dataset_labels  # kept for API compatibility; not drawn in this layout
    budget_text = budget_labels or DEFAULT_BUDGET_LABELS
    frozen = FRAMEWORK_COLORS["frozen"]
    layer_i = FRAMEWORK_COLORS["layer_i"]
    layer_ii = FRAMEWORK_COLORS["layer_ii"]
    layer_iii = FRAMEWORK_COLORS["layer_iii"]
    band_i = FRAMEWORK_BANDS["layer_i"]
    band_ii = FRAMEWORK_BANDS["layer_ii"]
    band_iii = FRAMEWORK_BANDS["layer_iii"]

    with paper_style(out_path):
        fig = plt.figure(figsize=paper_figsize("text", aspect=0.56), facecolor=WHITE)
        ax = fig.add_axes([0.008, 0.008, 0.984, 0.984])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # ===================================================================
        # Drawing primitives (axes-fraction coordinates)
        # ===================================================================
        def box(x, y, w, h, fill, edge, *, lw=1.5, rounding=0.020, z=2):
            """A rounded rectangle."""
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.004,rounding_size={rounding}",
                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z))

        def arrow(p0, p1, *, color, lw=1.7, scale=11):
            """A filled-head arrow from p0 to p1."""
            ax.add_patch(FancyArrowPatch(
                p0, p1, arrowstyle="-|>",
                mutation_scale=scale, linewidth=lw, color=color, zorder=5))

        def badge(x, y, text, facecolor, *, ha="left"):
            """A small white-on-colour pill, vertically centred on ``y``."""
            ax.text(x, y, f" {text} ", fontsize=7.0, fontweight="bold",
                    color=WHITE, ha=ha, va="center",
                    bbox=dict(boxstyle="round,pad=0.32",
                              facecolor=facecolor, edgecolor="none"))

        # ===================================================================
        # Section 1 -- title
        # ===================================================================
        def draw_title():
            fig.text(0.014, 0.962, "ReliableCD-Lite",
                     fontsize=14.5, fontweight="bold", color=INK)
            fig.text(0.986, 0.964,
                     "  training-free    ·    post-hoc    ·    no model retraining  ",
                     fontsize=7.3, fontweight="bold", color=WHITE,
                     ha="right", va="center",
                     bbox=dict(boxstyle="round,pad=0.20",
                               facecolor=layer_ii[1], edgecolor="none"))
            fig.text(0.014, 0.924,
                     "A post-hoc layer that ranks predicted change components by "
                     "error risk and spends a fixed analyst review budget on the "
                     "riskiest regions.",
                     fontsize=8.6, color=MUTED_TEXT)

        # ===================================================================
        # Section 2 -- five-stage pipeline
        # ===================================================================
        def draw_pipeline():
            # Two background zones: the untouched base vs. the post-hoc layer.
            zy, zh = FRAMEWORK_BANDS["pipeline"]
            box(0.010, zy, 0.394, zh, *FRAMEWORK_COLORS["base_zone"], lw=1.0, rounding=0.013, z=1)
            box(0.420, zy, 0.570, zh, *FRAMEWORK_COLORS["layer_zone"], lw=1.0, rounding=0.013, z=1)
            ax.text(0.207, zy + zh - 0.026, "FROZEN BASE   ·   never modified",
                    fontsize=7.5, fontweight="bold", color=FROZEN_ZONE_LABEL, ha="center")
            ax.text(0.705, zy + zh - 0.026,
                    "RELIABLECD-LITE   ·   post-hoc referral layer   (this paper)",
                    fontsize=7.5, fontweight="bold", color=layer_i[1], ha="center")

            # Stage boxes. The colour ties each stage to its zone / layer.
            stages = [
                ("1.  Image pair",
                 "bitemporal (A, B)\n256$\\times$256 patches", frozen),
                ("2.  Frozen CD model",
                 f"{model_name}, weights fixed\nP(change) map, $\\tau$=0.5", frozen),
                ("3.  Uncertainty signals",
                 "entropy, TTA disagree.,\nshift sens., boundary", layer_iii),
                ("4.  Component extraction",
                 "8-connected, $\\geq$16 px\npixel $\\rightarrow$ component", layer_i),
                ("5.  CRS-ranked referral",
                 "rank by risk, fill budget $\\beta$\nreview riskiest first", layer_i),
            ]
            for (title, body, (fill, edge)), x in zip(stages, FRAMEWORK_STAGE_XS):
                box(x, FRAMEWORK_STAGE_Y, FRAMEWORK_STAGE_W, FRAMEWORK_STAGE_H, fill, edge, lw=1.6)
                ax.text(x + FRAMEWORK_STAGE_W / 2, FRAMEWORK_STAGE_Y + FRAMEWORK_STAGE_H - 0.032, title,
                        fontsize=7.9, fontweight="bold", color=edge, ha="center")
                ax.text(x + FRAMEWORK_STAGE_W / 2, FRAMEWORK_STAGE_Y + 0.026, body,
                        fontsize=6.9, color=BODY_TEXT, ha="center", va="bottom",
                        linespacing=1.32)
            # Flow arrows between consecutive stages.
            y_mid = FRAMEWORK_STAGE_Y + FRAMEWORK_STAGE_H / 2
            for x_left, x_right in zip(FRAMEWORK_STAGE_XS[:-1], FRAMEWORK_STAGE_XS[1:]):
                arrow((x_left + FRAMEWORK_STAGE_W + 0.004, y_mid), (x_right - 0.004, y_mid),
                      color=FLOW_ARROW)

        # ===================================================================
        # Section 3 -- three-layer hierarchy
        # ===================================================================
        def draw_layer(band, tag, title, title_x, colour, bullets, *, primary=False):
            """Draw one hierarchy band: panel + tag + title + bullet lines.

            ``primary`` thickens the border and adds the contribution badge;
            it is used only for layer (i). ``title_x`` is explicit because the
            tags ``(i)`` / ``(ii)`` / ``(iii)`` have visibly different widths.
            """
            y, h = band
            fill, edge = colour
            box(0.010, y, 0.980, h, fill, edge, lw=2.3 if primary else 1.6)

            header_y = y + h - 0.034
            ax.text(0.030, header_y, tag, fontweight="bold", color=edge,
                    fontsize=13.0 if primary else 12.0)
            ax.text(title_x, header_y, title, fontweight="bold", color=INK,
                    fontsize=10.0 if primary else 9.4)
            if primary:
                badge(0.972, y + h - 0.021, "PRIMARY CONTRIBUTION", edge, ha="right")

            first_bullet_dy = 0.080 if primary else 0.068
            bullet_x = 0.036
            text_x = 0.048
            for i, line in enumerate(bullets):
                line_y = y + h - first_bullet_dy - i * 0.031
                ax.scatter([bullet_x], [line_y], s=9.0, marker="o",
                           color=BODY_TEXT, linewidths=0, zorder=4)
                ax.text(text_x, line_y, line, fontsize=7.3,
                        color=BODY_TEXT, va="center")

        def draw_risk_strip():
            """Layer (i) visual: components ranked by risk, the top-k that fit
            the area budget highlighted with a dark border + bracket."""
            sq_w, sq_h, sq_gap, x0 = 0.0300, 0.050, 0.0150, 0.662
            y = band_i[0] + 0.050

            for i, colour in enumerate(RISK_STRIP_COLORS):
                x = x0 + i * (sq_w + sq_gap)
                in_budget = i < RISK_STRIP_REVIEWED
                ax.add_patch(FancyBboxPatch(
                    (x, y), sq_w, sq_h,
                    boxstyle="round,pad=0.002,rounding_size=0.010",
                    linewidth=1.8 if in_budget else 0.8,
                    edgecolor=INK if in_budget else UNREVIEWED_RISK_EDGE,
                    facecolor=colour, zorder=3))

            # Bracket spanning the reviewed (budgeted) components.
            bx0 = x0 - 0.004
            bx1 = x0 + RISK_STRIP_REVIEWED * (sq_w + sq_gap) - sq_gap + 0.004
            by = y + sq_h + 0.012
            ax.plot([bx0, bx0, bx1, bx1], [by - 0.013, by, by, by - 0.013],
                    color=layer_i[1], linewidth=1.3, zorder=4)
            ax.text((bx0 + bx1) / 2, by + 0.009, "reviewed within budget $\\beta$",
                    fontsize=6.8, fontweight="bold", color=layer_i[1], ha="center")
            ax.text(x0, y - 0.024,
                    "high risk  $\\longrightarrow$  low risk   "
                    "(components ranked by CRS)",
                    fontsize=6.6, color=MUTED_TEXT)

        def draw_signal_chips():
            """Layer (iii) visual: the four uncertainty signals as chips."""
            chips = [
                ("$U_{\\mathrm{ent}}$", "predictive entropy"),
                ("$U_{\\mathrm{tta}}$", "TTA disagreement (6 views)"),
                ("$U_{\\mathrm{shift}}$", "shift sensitivity (4 shifts)"),
                ("$U_{\\mathrm{bnd}}$", "boundary-ring uncertainty"),
            ]
            chip_w, chip_gap, x0 = 0.2235, 0.0140, 0.032
            chip_h, y = 0.070, band_iii[0] + 0.022
            for i, (symbol, desc) in enumerate(chips):
                x = x0 + i * (chip_w + chip_gap)
                box(x, y, chip_w, chip_h, SIGNAL_CHIP_FILL, layer_iii[1],
                    lw=1.1, rounding=0.012, z=3)
                ax.text(x + chip_w / 2, y + chip_h - 0.026, symbol,
                        fontsize=9.6, fontweight="bold", color=layer_iii[1],
                        ha="center")
                ax.text(x + chip_w / 2, y + 0.014, desc,
                        fontsize=6.8, color=SIGNAL_CHIP_TEXT, ha="center")

        def connect_layers(lower, upper, colour, label):
            """Upward dependency arrow + caption between two stacked bands."""
            y_from = lower[0] + lower[1]   # top edge of the lower band
            y_to = upper[0]                # bottom edge of the upper band
            arrow((0.5, y_from + 0.002), (0.5, y_to - 0.002),
                  color=colour, lw=1.6, scale=10)
            ax.text(0.514, (y_from + y_to) / 2, label,
                    fontsize=6.7, style="italic", color=colour, va="center")

        def draw_hierarchy():
            ax.text(0.014, 0.668,
                    "The same post-hoc layer, organized as a three-layer "
                    "hierarchy — the protocol is the spine:",
                    fontsize=8.9, fontweight="bold", color=INK)

            draw_layer(
                band_i, "(i)", "Component-level protocol", 0.074, layer_i,
                bullets=[
                    "Review unit = one predicted change component.",
                    "Rank components by score; greedily fill area budget "
                    f"$\\beta$ ({budget_text}).",
                    "Analyst inspects the highest-risk regions first.",
                    "Component- vs. pixel-level referral at the same reviewed area.",
                ],
                primary=True,
            )
            draw_risk_strip()

            draw_layer(
                band_ii, "(ii)", "CRS scoring — the ranking instrument", 0.082,
                layer_ii,
                bullets=[
                    "CRS-3  (fast, one forward pass):   "
                    "$z(U_{\\mathrm{ent}}) + z(U_{\\mathrm{bnd}}) - z(m)$",
                    "CRS-4  (full):   "
                    "CRS-3 $+\\ z(U_{\\mathrm{tta}}) + z(U_{\\mathrm{shift}})$",
                    "Equal weights; z-scores fit once on source-domain "
                    "validation; no target-domain labels.",
                ],
            )

            draw_layer(
                band_iii, "(iii)", "Uncertainty signals — post-hoc diagnostics",
                0.090, layer_iii, bullets=[],
            )
            draw_signal_chips()

            # Build-up: signals feed the score, the score orders the components.
            connect_layers(band_iii, band_ii, layer_ii[1], "signals feed the score")
            connect_layers(band_ii, band_i, layer_i[1], "score orders the components")

        # ===================================================================
        # Section 4 -- observable-set limitation
        # ===================================================================
        def draw_scope():
            y, h = FRAMEWORK_BANDS["scope"]
            scope_fill, scope_edge, scope_badge = FRAMEWORK_COLORS["scope"]
            box(0.010, y, 0.980, h, scope_fill, scope_edge, lw=1.1, rounding=0.014)
            badge(0.028, y + h / 2, "SCOPE", scope_badge)
            ax.text(0.096, y + h / 2,
                    "Only predicted components are scored — completely missed "
                    "changes (false negatives) lie outside the ranked, "
                    "observable set.",
                    fontsize=7.6, color=SCOPE_TEXT, va="center")

        # ===================================================================
        # Compose the figure
        # ===================================================================
        # The ``journal`` style omits the poster-like banner (method name,
        # tagline, badge); the LaTeX caption carries that description and the
        # tight bounding box trims the vacated top margin for a clean schematic.
        if not journal:
            draw_title()
        draw_pipeline()
        draw_hierarchy()
        draw_scope()

        out_path = _save_figure(fig, out_path, pad_inches=FRAMEWORK_PAD_INCHES)
        plt.close(fig)
        return out_path


def plot_failure_mode_summary(
    components_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Quantitative failure-mode dashboard.

    Renders a single, self-evident row of three bar panels: components per
    dataset, error rate, and signal intensity. The per-dataset qualitative
    case examples are presented separately in the qualitative panel
    (Fig. 2b) and are no longer duplicated here, which keeps this figure to
    one readable row instead of a mixed 2x3 layout.
    """
    import matplotlib.pyplot as plt

    df = components_df.copy()
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(df["dataset"])]
    if not datasets:
        datasets = sorted(df["dataset"].unique().tolist())

    # Authored at the true two-column width so the LaTeX \resizebox wrapper
    # scales by ~1.0 and the point sizes below are the real on-page sizes.
    fig = plt.figure(figsize=paper_figsize("text", aspect=0.42), facecolor=WHITE)
    gs = fig.add_gridspec(1, 3, wspace=0.34, top=0.73, bottom=0.24, left=0.07, right=0.97)

    fig.text(0.07, 0.95, "Failure-Mode Dashboard", fontsize=11, fontweight="bold", color=BLACK)
    fig.text(
        0.07,
        0.87,
        "Observable component volume and risk-signal strength across operating regimes.",
        fontsize=7.8,
        color=FOOTNOTE_TEXT,
    )

    summary = []
    for dataset, group in df.groupby("dataset", dropna=False):
        error_mask = group["is_error_component"].astype(bool)
        summary.append(
            {
                "dataset": dataset,
                "components": len(group),
                "error_rate": float(error_mask.mean()) if len(group) else 0.0,
                "mean_entropy": float(group["mean_entropy"].mean()),
                "mean_shift": float(group["mean_shift"].mean()),
                "median_area": float(group["area"].median()),
                "mean_best_iou": float(group["best_gt_iou"].fillna(0.0).mean()),
                "mean_error_pixel_ratio": float(group["error_pixel_ratio"].fillna(0.0).mean()),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df["dataset"] = pd.Categorical(summary_df["dataset"], categories=datasets, ordered=True)
    summary_df = summary_df.sort_values("dataset")

    display_labels = [display_dataset_name(str(d)) for d in summary_df["dataset"]]
    axis_labels = [
        label.replace("-CD", "").replace("DSIFN†", "DSIFN†")
        for label in display_labels
    ]
    ax0 = fig.add_subplot(gs[0, 0])
    colors = [_dataset_color(dataset) for dataset in summary_df["dataset"]]
    ax0.bar(axis_labels, summary_df["components"], color=colors, edgecolor=BAR_EDGE, linewidth=1.1)
    _set_panel_title(ax0, "Components per Dataset", fontsize=9)
    ax0.set_ylabel("Count")
    _style_y_grid(ax0)
    ax0.tick_params(axis="x", labelsize=7.4)
    for x, y in zip(axis_labels, summary_df["components"]):
        ax0.text(x, y + max(summary_df["components"]) * 0.02, f"{int(y)}", ha="center", fontsize=8, color=BLACK)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(axis_labels, summary_df["error_rate"] * 100.0, color=colors, edgecolor=BAR_EDGE, linewidth=1.1)
    _set_panel_title(ax1, "Error Rate", fontsize=9)
    ax1.set_ylabel("Rate (%)")
    ax1.set_ylim(0, max(15.0, float(summary_df["error_rate"].max() * 120.0)))
    _style_y_grid(ax1)
    ax1.tick_params(axis="x", labelsize=7.4)
    for x, y in zip(axis_labels, summary_df["error_rate"] * 100.0):
        ax1.text(x, y + 1.5, f"{y:.1f}%", ha="center", fontsize=8, color=BLACK)

    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(summary_df))
    width = 0.24
    entropy_vals = summary_df["mean_entropy"].to_numpy()
    shift_vals = summary_df["mean_shift"].to_numpy()
    epr_vals = summary_df["mean_error_pixel_ratio"].to_numpy()
    ax2.bar(x - width, entropy_vals, width=width, color=SIGNAL_COLORS["entropy"], label="Entropy")
    ax2.bar(x, shift_vals, width=width, color=SIGNAL_COLORS["shift_light"], label="Shift")
    ax2.bar(x + width, epr_vals, width=width, color=SIGNAL_COLORS["error_ratio"], label="Error-pixel ratio")
    ax2.set_xticks(x)
    ax2.set_xticklabels(axis_labels)
    _set_panel_title(ax2, "Signal Intensity", fontsize=9)
    ax2.set_ylabel("Mean value")
    _style_y_grid(ax2)
    ax2.tick_params(axis="x", labelsize=7.4)
    ax2.legend(frameon=False, fontsize=7.2, loc="upper left")

    if "dsifn-256" in set(df["dataset"].astype(str)):
        fig.text(
            0.5,
            0.035,
            DSIFN_FOOTNOTE,
            ha="center",
            fontsize=8,
            color=FOOTNOTE_TEXT,
        )

    out_path = _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_dataset_difficulty_dashboard(
    components_df: pd.DataFrame,
    referral_df: pd.DataFrame,
    out_path: str | Path,
    model_name: str = "BIT",
) -> Path:
    import matplotlib.pyplot as plt

    datasets = _ordered_datasets(components_df["dataset"].dropna().unique())
    component_summary = (
        components_df.groupby("dataset", dropna=False)
        .agg(
            components=("component_id", "count"),
            error_rate=("is_error_component", lambda s: float(s.astype(bool).mean()) if len(s) else 0.0),
            mean_entropy=("mean_entropy", "mean"),
            mean_tta=("mean_tta", "mean"),
            mean_shift=("mean_shift", "mean"),
        )
        .reset_index()
    )
    component_summary["dataset"] = pd.Categorical(component_summary["dataset"], categories=datasets, ordered=True)
    component_summary = component_summary.sort_values("dataset")

    focus_budgets = select_focus_budgets(referral_df["budget"], preferred=DEFAULT_FOCUS_BUDGETS, max_items=2)
    referral_focus = referral_df[referral_df["budget"].isin(focus_budgets)].copy()
    referral_focus["dataset"] = pd.Categorical(referral_focus["dataset"], categories=datasets, ordered=True)
    referral_focus = referral_focus.sort_values(["dataset", "budget"])

    fig, axes = plt.subplots(3, 1, figsize=paper_figsize("column", aspect=1.36), facecolor=WHITE)
    colors = [_dataset_color(dataset) for dataset in component_summary["dataset"]]

    difficulty_display_labels = [display_dataset_name(str(d)) for d in component_summary["dataset"]]
    axes[0].bar(difficulty_display_labels, component_summary["components"], color=colors, edgecolor=BAR_EDGE)
    _set_panel_title(axes[0], "Component Volume")
    axes[0].set_ylabel("Components")
    _style_y_grid(axes[0])

    width = 0.23
    x = np.arange(len(component_summary))
    axes[1].bar(x - width, component_summary["mean_entropy"], width=width, color=SIGNAL_COLORS["entropy"], label="Entropy")
    axes[1].bar(x, component_summary["mean_tta"], width=width, color=SIGNAL_COLORS["tta"], label="TTA")
    axes[1].bar(x + width, component_summary["mean_shift"], width=width, color=SIGNAL_COLORS["shift"], label="Shift")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(difficulty_display_labels)
    _set_panel_title(axes[1], "Signal Strength")
    axes[1].set_ylabel("Mean value")
    _style_y_grid(axes[1])
    axes[1].legend(frameon=False, fontsize=7.4, loc="upper right")

    for dataset, group in referral_focus.groupby("dataset", dropna=False, observed=False):
        color = _dataset_color(dataset)
        axes[2].plot(
            group["budget"] * 100.0,
            group["component_f1_gain_upper_bound"],
            marker="o",
            linewidth=2.4,
            color=color,
            label=f"{display_dataset_name(str(dataset))} component gain",
        )
    budget_suffix = f" ({summarize_budget_labels(focus_budgets, max_items=2)})" if focus_budgets else ""
    _set_panel_title(axes[2], f"F1 Gain Upper Bound{budget_suffix}")
    axes[2].set_xlabel("Budget (%)")
    axes[2].set_ylabel("F1 gain")
    axes[2].grid(True, alpha=GRID_ALPHA)
    axes[2].legend(frameon=False, fontsize=7.4, loc="lower right")

    fig.suptitle(f"Dataset Difficulty ({model_name})", fontsize=10.2, fontweight="bold", color=BLACK)
    if "dsifn-256" in set(components_df["dataset"].astype(str)):
        fig.text(
            0.5,
            0.012,
            textwrap.fill(DSIFN_FOOTNOTE, 58),
            ha="center",
            fontsize=6.8,
            color=FOOTNOTE_TEXT,
        )
    fig.tight_layout(rect=(0.05, 0.04, 1.0, 0.965))
    out_path = _save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def plot_score_separation_panel(components_df: pd.DataFrame, out_path: str | Path, model_name: str = "BIT") -> Path:
    """Component error-detection ROC curves, one panel per dataset.

    Each panel overlays the ROC curve of every main component score
    (CRS-4, CRS-3, Margin, Entropy) against the binary error label, with
    AUROC in the legend and the random baseline as a dotted diagonal.
    Plotting every score on the common, unitless TPR/FPR axes makes the
    ranking-quality comparison direct: a curve nearer the top-left corner
    separates error components from correct ones more effectively.
    """
    import matplotlib.pyplot as plt

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        _has_sklearn = True
    except ImportError:
        _has_sklearn = False

    df = components_df.copy()
    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(df["dataset"])]
    if not datasets:
        datasets = sorted(df["dataset"].unique().tolist())

    # (column, legend label, colour, linestyle, linewidth)
    score_specs = SCORE_LINE_SPECS
    score_specs = [spec for spec in score_specs if spec[0] in df.columns]

    with paper_style(out_path):
        fig, axes = plt.subplots(
            1, len(datasets),
            figsize=paper_figsize("text", aspect=0.42),
            facecolor=WHITE,
            squeeze=False,
        )

        for col_idx, dataset in enumerate(datasets):
            ax = axes[0, col_idx]
            group = df[df["dataset"] == dataset]
            error_mask = group["is_error_component"].astype(bool)
            n_correct = int((~error_mask).sum())
            n_error   = int(error_mask.sum())
            n_total   = len(group)
            y_true = error_mask.astype(int).to_numpy()
            single_class = len(np.unique(y_true)) < 2

            for score_col, label, color, ls, lw in score_specs:
                if single_class or not _has_sklearn:
                    break
                scores = group[score_col].fillna(0.0).to_numpy(dtype=float)
                fpr, tpr, _ = roc_curve(y_true, scores)
                auroc = roc_auc_score(y_true, scores)
                ax.plot(
                    fpr, tpr,
                    color=color, linestyle=ls, linewidth=lw,
                    zorder=3, label=f"{label}  ({auroc:.3f})",
                )

            ax.plot(
                [0, 1], [0, 1],
                color=SUBTLE_SPINE, linewidth=1.0, linestyle=(0, (1, 1.8)),
                zorder=1, label="Random  (0.500)",
            )

            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_box_aspect(1.0)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.grid(True, alpha=GRID_ALPHA, linestyle=":")
            ax.set_axisbelow(True)
            ax.set_facecolor(WHITE)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SUBTLE_SPINE)
            ax.spines["bottom"].set_color(SUBTLE_SPINE)

            ax.set_title(
                f"{display_dataset_name(dataset)}\n"
                f"N={n_total}  ({n_error} error / {n_correct} correct)",
                fontsize=8.4, fontweight="bold", color=BLACK,
            )
            ax.set_xlabel("False positive rate", fontsize=8.0)
            if col_idx == 0:
                ax.set_ylabel("True positive rate", fontsize=8.0)

            if single_class:
                ax.text(
                    0.5, 0.5, "single-class\n(AUROC undefined)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, color=DATASET_FALLBACK_COLOR,
                )
            else:
                leg = ax.legend(
                    loc="lower right", fontsize=6.6, frameon=True,
                    title="Score (AUROC)", title_fontsize=6.8,
                    handlelength=1.9, borderpad=0.5, labelspacing=0.34,
                )
                leg.get_frame().set_edgecolor(LEGEND_FRAME)
                leg.get_frame().set_facecolor(WHITE)
                leg.get_frame().set_linewidth(0.6)

        fig.suptitle(
            f"Component Error-Detection ROC ({model_name})",
            fontsize=9.4, fontweight="bold", color=BLACK,
        )
        fig.tight_layout()
        out_path = _save_figure(fig, out_path)
        plt.close(fig)
    return out_path


def plot_protocol_walkthrough(
    components_df: pd.DataFrame,
    pred_masks_path: str | Path,
    prob_maps_path: str | Path,
    out_path: str | Path,
    *,
    dataset: str = "whu-256",
    image_id: str | None = None,
    budget: float = 0.05,
    model_name: str = "BIT",
) -> Path:
    """Worked example of the referral protocol on one real image.

    Four panels make the abstract protocol concrete and contrast component-
    vs. pixel-level referral at the same area budget:

        1. predicted change components (the review units)
        2. components shaded by CRS error risk
        3. component referral -- whole components picked under budget beta
        4. pixel referral -- top-entropy pixels picked under the same budget

    Panels 3 vs. 4 show the paper's core qualitative claim: component
    referral selects spatially coherent regions, pixel referral selects
    scattered boundary pixels.
    """
    import matplotlib.pyplot as plt
    from scipy import ndimage

    pred = np.load(pred_masks_path, allow_pickle=True)
    prob = np.load(prob_maps_path, allow_pickle=True)
    image_ids = [str(x) for x in pred["image_ids"]]
    pred_masks = pred["pred_masks"]
    prob_key = "prob_maps" if "prob_maps" in prob.files else prob.files[0]
    prob_maps = prob[prob_key]

    comp = components_df[components_df["dataset"].astype(str) == dataset].copy()
    crs_col = "score_crs4" if "score_crs4" in comp.columns else "crs4"

    # -- pick an instructive image -------------------------------------------
    # Want: several components, both error and correct, no single blob that
    # dominates, a total area large enough that the budget selects a clear
    # subset, and -- for the CRS panel -- a readable colour gradient: visible
    # mid-tones plus a high-risk (red) component large enough to be seen.
    if image_id is None:
        budget_px_sel = budget * DEFAULT_PATCH_SIZE_PX * DEFAULT_PATCH_SIZE_PX
        best, best_score = None, -1e9
        for img, grp in comp.groupby("image_id"):
            if str(img) not in image_ids:
                continue
            n = len(grp)
            n_err = int(grp["is_error_component"].astype(bool).sum())
            if not (5 <= n <= 18 and n_err >= 2 and n - n_err >= 2):
                continue
            areas = grp["area"].to_numpy(dtype=float)
            total = float(areas.sum())
            crs_vals = grp[crs_col].to_numpy(dtype=float)
            lo, hi = float(crs_vals.min()), float(crs_vals.max())
            rng = (hi - lo) or 1.0
            norms = (crs_vals - lo) / rng
            # green (norm=0) and red (norm=1) always exist by construction;
            # additionally require visible mid-tone colours and reward a
            # high-risk component that is large enough to read in the panel.
            n_mid = int(((norms > 0.34) & (norms < 0.66)).sum())
            red_vis = float(areas[norms >= 0.66].max()) / float(areas.max())
            score = 6.0 * min(red_vis, 0.5)        # visible red component
            score += 0.5 * min(hi - lo, 4.0)       # useful CRS range, no extreme-chasing
            if n_mid < 1:
                score -= 50.0           # want mid-tone (yellow/orange) colours present
            if total < 2.5 * budget_px_sel or total > 18.0 * budget_px_sel:
                score -= 100.0          # budget would select all / too little
            if areas.max() > 0.5 * total:
                score -= 100.0          # one component dominates the image
            if score > best_score:
                best, best_score = str(img), score
        image_id = best
    if image_id is None or image_id not in image_ids:
        # fallback: any image that has components
        for img in comp["image_id"].astype(str).unique():
            if img in image_ids:
                image_id = img
                break

    idx = image_ids.index(str(image_id))
    mask = np.asarray(pred_masks[idx]).astype(bool)
    prob_map = np.asarray(prob_maps[idx]).astype(float)
    if prob_map.ndim == 3:
        prob_map = prob_map.squeeze()
    h, w = mask.shape

    # -- label predicted components (8-connectivity, min area 16 px) ----------
    labeled, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    rows = comp[comp["image_id"].astype(str) == str(image_id)]

    # match each labelled component to a parquet row by centroid proximity
    centroids = ndimage.center_of_mass(mask, labeled, range(1, n_labels + 1))
    parts = []   # (label, crs, is_error, area)
    used = set()
    for lab, (cy, cx) in zip(range(1, n_labels + 1), centroids):
        area = int((labeled == lab).sum())
        if area < DEFAULT_MIN_COMPONENT_AREA:
            continue
        cand = rows[~rows.index.isin(used)]
        if cand.empty:
            continue
        pcx = cand["bbox_x"] + cand["bbox_w"] / 2.0
        pcy = cand["bbox_y"] + cand["bbox_h"] / 2.0
        j = ((pcx - cx) ** 2 + (pcy - cy) ** 2).idxmin()
        used.add(j)
        parts.append({
            "label": lab,
            "crs": float(rows.loc[j, crs_col]),
            "is_error": bool(rows.loc[j, "is_error_component"]),
            "area": area,
        })

    pdf = pd.DataFrame(parts)
    if pdf.empty:
        raise ValueError(f"no matched components for image {image_id}")

    # -- component referral: rank by CRS, greedily fill the area budget ------
    pdf = pdf.sort_values("crs", ascending=False).reset_index(drop=True)
    budget_px = budget * h * w
    pdf["cum_area"] = pdf["area"].cumsum()
    pdf["selected"] = pdf["cum_area"] - pdf["area"] < budget_px
    sel_labels = set(pdf.loc[pdf["selected"], "label"])
    n_sel = int(pdf["selected"].sum())
    n_err_sel = int((pdf["selected"] & pdf["is_error"]).sum())
    n_err_total = int(pdf["is_error"].sum())

    # -- pixel referral: top-budget fraction of pixels by binary entropy -----
    p = np.clip(prob_map, ENTROPY_EPS, 1.0 - ENTROPY_EPS)
    entropy = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    k = int(budget * h * w)
    thr = np.partition(entropy.ravel(), -k)[-k] if 0 < k < entropy.size else entropy.max()
    pixel_sel = entropy >= thr

    crs_lookup = dict(zip(pdf["label"], pdf["crs"]))
    crs_lo, crs_hi = pdf["crs"].min(), pdf["crs"].max()
    crs_rng = (crs_hi - crs_lo) or 1.0

    with paper_style(out_path):
        fig, axes = plt.subplots(1, 4, figsize=paper_figsize("text", aspect=0.40),
                                 facecolor=WHITE)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(FRAME_EDGE)

        risk_cmap = plt.get_cmap(RISK_CMAP_NAME)

        def _color_key(ax, items):
            """Draw a small colour key (legend) under a panel."""
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c,
                                     edgecolor=MUTED_TEXT, linewidth=0.5)
                       for c, _ in items]
            ax.legend(handles, [lab for _, lab in items],
                      loc="upper center", bbox_to_anchor=(0.5, -0.015),
                      ncol=len(items), fontsize=5.8, frameon=False,
                      handlelength=1.0, handleheight=1.0,
                      handletextpad=0.4, columnspacing=0.9)

        def _risk_colorbar(ax, vmin, vmax):
            """Continuous colour bar for the per-component CRS risk shading."""
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import LinearSegmentedColormap, Normalize
            if vmax <= vmin:
                vmax = vmin + 1.0
            bar_cmap = LinearSegmentedColormap.from_list(
                "crs_risk", risk_cmap(np.linspace(RISK_CMAP_LOW, RISK_CMAP_LOW + RISK_CMAP_SPAN, 256)))
            cax = ax.inset_axes(COLORBAR_INSET)
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=bar_cmap)
            cb = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
            cb.outline.set_edgecolor(MUTED_TEXT)
            cb.outline.set_linewidth(0.5)
            cb.set_ticks([vmin, vmax])
            cb.set_ticklabels(["low risk", "high risk"])
            cax.tick_params(length=0, labelsize=5.8, colors=BLACK, pad=1.5)

        # panel 1 -- predicted components
        rgb1 = np.ones((h, w, 3))
        rgb1[mask] = RGB_PREDICTED_COMPONENT
        axes[0].imshow(rgb1)
        axes[0].contour(labeled > 0, levels=[0.5], colors=SIGNAL_COLORS["component_referral"], linewidths=0.8)
        axes[0].set_title("1. Predicted components\n(the review units)",
                          fontsize=SMALL_TITLE_SIZE, fontweight="bold", color=BLACK)
        _color_key(axes[0], [(RGB_PREDICTED_COMPONENT, "predicted component")])

        # panel 2 -- components shaded by CRS risk
        rgb2 = np.ones((h, w, 3))
        for lab in pdf["label"]:
            norm = (crs_lookup[lab] - crs_lo) / crs_rng
            rgb2[labeled == lab] = risk_cmap(RISK_CMAP_LOW + RISK_CMAP_SPAN * norm)[:3]
        axes[1].imshow(rgb2)
        axes[1].contour(labeled > 0, levels=[0.5], colors=COMPONENT_CONTOUR, linewidths=0.6)
        axes[1].set_title("2. CRS risk score\n(shaded per component)",
                          fontsize=SMALL_TITLE_SIZE, fontweight="bold", color=BLACK)
        _risk_colorbar(axes[1], crs_lo, crs_hi)

        # panel 3 -- component referral under the budget
        rgb3 = np.ones((h, w, 3))
        for lab in pdf["label"]:
            rgb3[labeled == lab] = RGB_UNREVIEWED_COMPONENT
        for lab in sel_labels:
            rgb3[labeled == lab] = RGB_REVIEWED_COMPONENT
        axes[2].imshow(rgb3)
        axes[2].contour(labeled > 0, levels=[0.5], colors=COMPONENT_CONTOUR, linewidths=0.6)
        axes[2].set_title(
            f"3. Component referral @ {int(budget * 100)}%\n"
            f"{n_sel} whole regions reviewed",
            fontsize=SMALL_TITLE_SIZE, fontweight="bold", color=SIGNAL_COLORS["component_referral"])
        _color_key(axes[2], [(RGB_REVIEWED_COMPONENT, "reviewed"),
                             (RGB_UNREVIEWED_COMPONENT, "not reviewed")])

        # panel 4 -- pixel referral under the same budget
        rgb4 = np.ones((h, w, 3))
        rgb4[mask] = RGB_CHANGE_CONTEXT
        rgb4[pixel_sel] = RGB_REVIEWED_PIXEL
        axes[3].imshow(rgb4)
        axes[3].set_title(
            f"4. Pixel referral @ {int(budget * 100)}%\n"
            "scattered top-entropy pixels",
            fontsize=SMALL_TITLE_SIZE, fontweight="bold", color=PIXEL_REFERRAL_TITLE)
        _color_key(axes[3], [(RGB_REVIEWED_PIXEL, "reviewed pixels"),
                             (RGB_CHANGE_CONTEXT, "change pixels")])

        fig.suptitle(
            f"Referral protocol on one {display_dataset_name(dataset)} image "
            f"({model_name})   ·   component triage caught "
            f"{n_err_sel}/{n_err_total} error components",
            fontsize=8.6, fontweight="bold", color=BLACK, y=1.02)

        fig.tight_layout()
        out_path = _save_figure(fig, out_path)
        plt.close(fig)
    return out_path


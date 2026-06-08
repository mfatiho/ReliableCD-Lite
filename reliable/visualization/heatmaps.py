from __future__ import annotations

from pathlib import Path

import pandas as pd

from reliable.visualization.panels import paper_figsize, paper_style

# Style imported from panels.PAPER_RC_PARAMS via paper_style()

_FEATURE_LABELS: dict[str, str] = {
    "mean_entropy":          "Entropy",
    "mean_tta":              "TTA disagreement",
    "mean_shift":            "Shift Sensitivity",
    "boundary_uncertainty":  "Boundary Unc.",
    "probability_margin":    "Prob. Margin",
    "area":                  "Area (px)",
    "compactness":           "Compactness",
    "eccentricity":          "Eccentricity",
}

def _prepare_error_corr_df(corr_df: pd.DataFrame) -> pd.DataFrame:
    df = corr_df.copy()
    df["abs_rho"] = df["spearman_rho_with_error"].abs()
    df["label"] = df["feature"].map(lambda f: _FEATURE_LABELS.get(f, f))
    return df.sort_values(
        ["abs_rho", "spearman_rho_with_error"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_spearman_heatmap(matrix_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Apply readable labels where possible
    col_labels = [_FEATURE_LABELS.get(c, c) for c in matrix_df.columns]
    row_labels  = [_FEATURE_LABELS.get(r, r) for r in matrix_df.index]

    matrix = matrix_df.to_numpy(dtype=float)
    n = matrix.shape[0]
    with paper_style(out_path):
        fig, ax = plt.subplots(figsize=paper_figsize("column", aspect=1.05), facecolor="#FFFFFF")

        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7.2)
        ax.set_yticklabels(row_labels, fontsize=7.2)

        ax.set_title(
            "Spearman Correlation - Component Features",
            fontsize=8.9, fontweight="bold", color="#000000", pad=8,
        )

        # Cell annotations - dark text on light cells, light text on dark cells
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                text_color = "#FFFFFF" if abs(val) > 0.55 else "#000000"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6.7, color=text_color, fontweight="bold" if abs(val) > 0.6 else "normal",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cbar.ax.set_ylabel(r"Spearman $\rho$", rotation=90, labelpad=7, fontsize=7.5)
        cbar.ax.tick_params(labelsize=7.0)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def plot_error_correlation_forest(
    corr_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Feature-Error Spearman Correlation",
) -> Path:
    """Cleveland dot plot of each feature's Spearman rho with the error label.

    ``corr_df`` must have columns ``feature`` and ``spearman_rho_with_error``.
    Optional CI columns are ignored in this simplified main-text rendering.
    """
    import matplotlib.pyplot as plt

    df = _prepare_error_corr_df(corr_df)
    y_pos = list(range(len(df)))
    x_vals = df["spearman_rho_with_error"].astype(float)
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    x_span = max(x_max - x_min, 0.4)
    x_pad = max(0.22, 0.10 * x_span)

    with paper_style(out_path):
        fig, ax = plt.subplots(figsize=paper_figsize("column", aspect=0.96), facecolor="#FFFFFF")

        point_colors = ["#2A9D8F" if v >= 0 else "#E76F51" for v in df["spearman_rho_with_error"]]
        stem_color = "#D1D5DB"

        for y, val, color in zip(y_pos, x_vals, point_colors, strict=False):
            ax.hlines(y, xmin=min(0.0, val), xmax=max(0.0, val), color=stem_color, linewidth=1.8, zorder=1)

        ax.scatter(
            x_vals,
            y_pos,
            s=98,
            c=point_colors,
            edgecolors="#23313F",
            linewidths=0.9,
            zorder=2,
        )
        ax.axvline(0, color="#000000", linewidth=0.8, linestyle="-", zorder=0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["label"], fontsize=7.8)

        # Value labels sit centred just above each dot, clear of the marker,
        # so the numbers never overlap the points or each other.
        for idx, row in enumerate(df.itertuples(index=False)):
            val = float(row.spearman_rho_with_error)
            ax.text(
                val,
                y_pos[idx] - 0.36,
                f"{val:+.3f}",
                va="bottom",
                ha="center",
                fontsize=7.5,
                color="#000000",
                zorder=3,
            )

        ax.set_xlabel(r"Spearman $\rho$  (feature vs. error label)", fontsize=8.0)
        ax.set_title(title, fontsize=9.2, fontweight="bold", color="#000000", pad=8)
        ax.grid(True, axis="x", alpha=0.18, linestyle=":")
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9CA3AF")
        ax.spines["bottom"].set_color("#9CA3AF")
        if y_pos:
            ax.set_ylim(y_pos[-1] + 0.55, -0.78)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0.34, right=0.97, top=0.90, bottom=0.16)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def plot_error_correlation_bars(
    corr_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Feature-Error Spearman Correlation",
) -> Path:
    """Backward-compatible wrapper; renders the Cleveland dot plot."""
    return plot_error_correlation_forest(corr_df, out_path, title=title)

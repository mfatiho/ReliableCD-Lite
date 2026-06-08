from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from reliable.visualization.panels import (
    DATASET_COLORS,
    DATASET_ORDER,
    DSIFN_FOOTNOTE,
    PAPER_RC_PARAMS,
    display_dataset_name,
    paper_figsize,
    paper_style,
    select_focus_budgets,
)

_METHOD_STYLES: dict[str, dict] = {
    "CRS-4":   {"color": "#2A9D8F", "linestyle": "-",  "marker": "o", "lw": 2.4, "zorder": 5},
    "CRS-3":   {"color": "#457B9D", "linestyle": "-",  "marker": "s", "lw": 2.0, "zorder": 4},
    "CRS-2":   {"color": "#70B8C8", "linestyle": "-",  "marker": "^", "lw": 1.8, "zorder": 3},
    "CRS-1":   {"color": "#A8DADC", "linestyle": "-",  "marker": "D", "lw": 1.8, "zorder": 3},
    "Margin":  {"color": "#E76F51", "linestyle": "--", "marker": "^", "lw": 2.0, "zorder": 4},
    "Entropy": {"color": "#F4A261", "linestyle": "--", "marker": "s", "lw": 1.6, "zorder": 3},
    "MI":      {"color": "#8338EC", "linestyle": ":",  "marker": "D", "lw": 1.6, "zorder": 2},
    "Random":  {"color": "#ADB5BD", "linestyle": "--", "marker": "",  "lw": 1.4, "zorder": 1},
}

# Style imported from panels.PAPER_RC_PARAMS via paper_style()

REVIEW_BUDGET_COMPONENT_COMPARE_COL = "component_error_pixel_recall"
REVIEW_BUDGET_PIXEL_COMPARE_COL = "pixel_error_pixel_recall"
REVIEW_BUDGET_COMPONENT_PRIMARY_COL = "component_error_recall"


def _ordered_datasets(values: list[str] | pd.Index | np.ndarray) -> list[str]:
    present = [str(v) for v in values]
    present_set = set(present)
    ordered = [dataset for dataset in DATASET_ORDER if dataset in present_set]
    extras = sorted(dataset for dataset in present if dataset not in ordered)
    return ordered + extras


def plot_review_budget_curves(referral_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with paper_style(out_path):
        fig, axes = plt.subplots(2, 1, figsize=paper_figsize("column", aspect=1.36), facecolor="#FFFFFF")
        ordered = referral_df.copy()
        datasets = _ordered_datasets(ordered["dataset"].dropna().unique())
        key_budgets_pct = [b * 100.0 for b in select_focus_budgets(ordered["budget"], preferred=(0.05, 0.10), max_items=2)]
        ordered["dataset"] = pd.Categorical(ordered["dataset"], categories=datasets, ordered=True)
        ordered = ordered.sort_values(["dataset", "budget"]).copy()

        has_dsifn = any("dsifn" in str(d).lower() for d in datasets)
        for dataset in datasets:
            group = ordered[ordered["dataset"] == dataset]
            if group.empty:
                continue
            color = DATASET_COLORS.get(dataset, "#5A6C7D")
            budget_pct = group["budget"].to_numpy() * 100.0
            label = display_dataset_name(dataset)

            # Apples-to-apples comparison: both methods use error-pixel recall.
            axes[0].plot(
                budget_pct,
                group[REVIEW_BUDGET_COMPONENT_COMPARE_COL].to_numpy(),
                marker="o",
                linewidth=2.5,
                color=color,
                label=label,
                zorder=3,
            )
            axes[0].plot(
                budget_pct,
                group[REVIEW_BUDGET_PIXEL_COMPARE_COL].to_numpy(),
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color=color,
                alpha=0.60,
                zorder=2,
            )

            # F1 gain upper-bound panel
            axes[1].plot(
                budget_pct,
                group["component_f1_gain_upper_bound"].to_numpy(),
                marker="o",
                linewidth=2.5,
                color=color,
                label=label,
                zorder=3,
            )
            axes[1].plot(
                budget_pct,
                group["pixel_f1_gain_upper_bound"].to_numpy(),
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color=color,
                alpha=0.60,
                zorder=2,
            )

        # Vertical reference lines at key budget points
        for ax in axes:
            for b in key_budgets_pct:
                ax.axvline(b, color="#E5E7EB", linewidth=1.0, linestyle=":", zorder=0)

        axes[0].set_ylim(0.0, 1.10)
        axes[0].set_xlim(0.3, 20.7)
        axes[0].set_xlabel("Review budget (% area)")
        axes[0].set_ylabel("Error-pixel recall", labelpad=3)
        axes[0].set_title(
            "(a) Error-Pixel Recall vs. Review Budget",
            fontsize=8.6, fontweight="bold", color="#000000",
        )
        axes[0].grid(True, axis="y", alpha=0.25, linestyle=":", color="#E5E7EB", linewidth=0.5)
        # Custom legend: dataset color + solid/dashed style explanation
        handles, labels = axes[0].get_legend_handles_labels()
        import matplotlib.lines as mlines
        solid_patch = mlines.Line2D([], [], color="#444444", linestyle="-",  marker="o", label="component referral")
        dash_patch  = mlines.Line2D([], [], color="#444444", linestyle="--", marker="s", label="pixel referral", alpha=0.6)
        legend_handles = handles + [solid_patch, dash_patch]
        legend_labels = labels + ["component referral", "pixel referral"]
        axes[0].legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left", ncol=1, fontsize=6.9, handlelength=1.25,
        )

        axes[1].set_xlim(0.3, 20.7)
        axes[1].set_xlabel("Review budget (% area)")
        axes[1].set_ylabel("Oracle F1 gain", labelpad=3)
        axes[1].set_title(
            "(b) F1 Gain Upper Bound vs. Review Budget",
            fontsize=8.6, fontweight="bold", color="#000000",
        )
        axes[1].grid(True, axis="y", alpha=0.25, linestyle=":", color="#E5E7EB", linewidth=0.5)
        axes[1].legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper left", ncol=1, fontsize=6.9, handlelength=1.25,
        )

        model_names = sorted({str(name) for name in referral_df.get("model_name", pd.Series(dtype=object)).dropna().unique()})
        model_label = model_names[0] if len(model_names) == 1 else "CD"
        fig.suptitle(
            f"Budgeted Referral ({model_label})",
            fontsize=9.0, fontweight="bold", color="#000000", y=0.99,
        )
        if has_dsifn:
            fig.text(
                0.5, 0.012, textwrap.fill(DSIFN_FOOTNOTE, 58),
                ha="center", fontsize=6.1, color="#444444", style="italic",
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0.02, 0.035, 1.0, 0.965))
        fig.subplots_adjust(left=0.05, right=0.995)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def plot_risk_coverage_curves(
    curves_input: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame],
    out_path: str | Path,
    model_name: str = "BIT",
) -> Path:
    """Risk-coverage curves.

    Accepts:
    - ``dict[dataset, dict[method, DataFrame]]`` — per-dataset, multi-method (preferred)
    - ``dict[method, DataFrame]`` — legacy single-dataset fallback
    """
    import matplotlib.pyplot as plt

    first_val = next(iter(curves_input.values()))
    per_dataset = isinstance(first_val, dict)

    with paper_style(out_path):
        if per_dataset:
            datasets = _ordered_datasets(list(curves_input))
            n = len(datasets)
            fig, axes = plt.subplots(1, n, figsize=paper_figsize("text", aspect=0.45), facecolor="#FFFFFF", squeeze=False)
            axes_flat = axes[0]
            focus_methods = [m for m in ["Random", "Margin", "Entropy", "CRS-3", "CRS-4"] if any(m in curves_input[d] for d in datasets)]

            for col_idx, dataset in enumerate(datasets):
                ax = axes_flat[col_idx]
                method_curves = curves_input[dataset]
                best_crs: tuple[str, float] | None = None
                for method in focus_methods:
                    if method not in method_curves:
                        continue
                    df_s = method_curves[method].sort_values("coverage")
                    style = _METHOD_STYLES.get(method, {"color": "#6B7280", "linestyle": "-", "marker": "", "lw": 1.8, "zorder": 2})
                    ax.plot(
                        df_s["coverage"],
                        df_s["residual_risk"],
                        label=method,
                        linewidth=style["lw"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        zorder=style["zorder"],
                    )
                    if method in {"CRS-3", "CRS-4"}:
                        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
                        aurc = abs(float(_trapz(df_s["residual_risk"].to_numpy(), df_s["coverage"].to_numpy())))
                        if best_crs is None or aurc < best_crs[1]:
                            best_crs = (method, aurc)

                if best_crs is not None:
                    ax.text(
                        0.04, 0.96,
                        f"Best CRS AURC = {best_crs[1]:.4f} ({best_crs[0]})",
                        transform=ax.transAxes,
                        fontsize=7.0,
                        color="#2A9D8F",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#E5E7EB", alpha=0.85),
                    )

                ax.set_title(display_dataset_name(dataset), fontsize=8.8, fontweight="bold", color="#000000")
                ax.set_xlabel("Coverage (fraction un-reviewed)")
                if col_idx == 0:
                    ax.set_ylabel("Residual Risk")
                ax.grid(True, alpha=0.25, linestyle=":", color="#E5E7EB", linewidth=0.5)
                ax.set_facecolor("#FFFFFF")
                ax.legend(loc="upper right", fontsize=6.8)

            has_dsifn_rc = any("dsifn" in str(d).lower() for d in datasets)
            fig.suptitle(
                f"Risk-Coverage ({model_name})",
                fontsize=9.4, fontweight="bold", color="#000000",
            )
            if has_dsifn_rc:
                fig.text(
                    0.5, -0.02, DSIFN_FOOTNOTE,
                    ha="center", fontsize=6.8, color="#444444", style="italic",
                )

        else:
            # Legacy single-chart mode
            palette = {"CRS-4": "#2A9D8F", "Margin": "#E76F51"}
            fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="#FFFFFF")
            for label, df in curves_input.items():
                df_s = df.sort_values("coverage")
                ax.plot(
                    df_s["coverage"], df_s["residual_risk"],
                    label=label, linewidth=2.6, color=palette.get(label, "#5A6C7D"),
                )
            ax.set_xlabel("Coverage")
            ax.set_ylabel("Residual Risk")
            ax.set_title("Risk-Coverage Comparison", fontsize=14, fontweight="bold", color="#000000")
            ax.grid(True, alpha=0.22)
            ax.legend()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def plot_risk_coverage_summary(
    curves_input: dict[str, dict[str, pd.DataFrame]],
    out_path: str | Path,
    model_name: str = "BIT",
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    datasets = _ordered_datasets(list(curves_input))
    focus_methods = [m for m in ["Random", "Margin", "Entropy", "CRS-3", "CRS-4"] if any(m in curves_input[d] for d in datasets)]
    aurc_rows: list[dict[str, object]] = []
    for dataset in datasets:
        method_curves = curves_input[dataset]
        for method in focus_methods:
            if method not in method_curves:
                continue
            df_s = method_curves[method].sort_values("coverage")
            _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
            aurc = abs(float(_trapz(df_s["residual_risk"].to_numpy(), df_s["coverage"].to_numpy())))
            aurc_rows.append({"dataset": dataset, "method": method, "aurc": aurc})

    aurc_df = pd.DataFrame(aurc_rows)
    method_to_y = {method: idx for idx, method in enumerate(reversed(focus_methods))}
    dataset_offsets = np.linspace(-0.22, 0.22, num=max(1, len(datasets)))
    dataset_offset_map = {dataset: dataset_offsets[idx] for idx, dataset in enumerate(datasets)}

    with paper_style(out_path):
        fig, ax = plt.subplots(figsize=paper_figsize("column", aspect=0.92), facecolor="#FFFFFF")

        for method in focus_methods:
            y = method_to_y[method]
            ax.axhline(y, color="#E2E8F0", linewidth=0.8, zorder=0)
            method_slice = aurc_df[aurc_df["method"] == method]
            if len(method_slice) >= 2:
                x_min = float(method_slice["aurc"].min())
                x_max = float(method_slice["aurc"].max())
                ax.hlines(y, x_min, x_max, color="#E5E7EB", linewidth=2.0, zorder=1)

        for dataset in datasets:
            ds_color = DATASET_COLORS.get(dataset, "#6B7280")
            ds_slice = aurc_df[aurc_df["dataset"] == dataset]
            for row in ds_slice.itertuples(index=False):
                y = method_to_y[str(row.method)] + dataset_offset_map[dataset]
                ax.scatter(
                    float(row.aurc),
                    y,
                    s=48,
                    color=ds_color,
                    edgecolor="#000000",
                    linewidth=0.7,
                    zorder=3,
                )

        handles = [
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=6.0,
                markerfacecolor=DATASET_COLORS.get(dataset, "#6B7280"),
                markeredgecolor="#000000",
                label=display_dataset_name(dataset),
            )
            for dataset in datasets
        ]
        ax.set_yticks([method_to_y[m] for m in reversed(focus_methods)])
        ax.set_yticklabels(list(reversed(focus_methods)))
        ax.set_xlabel("AURC (lower is better)")
        ax.set_ylabel("Method")
        ax.set_title(f"AURC Summary ({model_name})", fontsize=9.3, fontweight="bold", color="#000000")
        ax.grid(True, axis="x", alpha=0.18, linestyle=":")
        ax.set_facecolor("#FFFFFF")
        # Place the legend in the dot-free mid-AURC band with an opaque box
        # so it does not visually merge with the data points / connectors.
        ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.40, 0.5),
                  fontsize=7.0, title="Dataset", title_fontsize=7.0,
                  frameon=True, facecolor="#FFFFFF", framealpha=0.96,
                  edgecolor="#D8DEE5")

        if any("dsifn" in str(d).lower() for d in datasets):
            fig.text(
                0.5, 0.015, textwrap.fill(DSIFN_FOOTNOTE, 58),
                ha="center", fontsize=6.5, color="#444444", style="italic",
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def plot_crs_ablation_bars(
    ablation_df: pd.DataFrame,
    out_path: str | Path,
    budget_label: str = "5%",
    model_name: str = "BIT",
) -> Path:
    """Grouped bar chart: Error Recall@budget for CRS variants and reference baselines.

    ``ablation_df`` columns: [dataset, method, error_recall]
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    METHOD_ORDER   = ["Random", "Entropy", "Margin", "CRS-1", "CRS-2", "CRS-3", "CRS-4"]

    if ablation_df.empty:
        raise ValueError("CRS ablation input is empty; no dataset/method rows were available for plotting.")

    datasets = _ordered_datasets(ablation_df["dataset"].dropna().unique())
    present  = set(ablation_df["method"].unique())
    methods  = [m for m in METHOD_ORDER if m in present]

    n_methods  = len(methods)
    n_datasets = len(datasets)
    if n_datasets == 0 or n_methods == 0:
        raise ValueError("CRS ablation plot requires at least one dataset and one method.")

    def _muted(color: str, blend: float = 0.40) -> tuple[float, float, float]:
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1.0 - blend) * rgb + blend * np.ones(3))

    with paper_style(out_path):
        fig, ax = plt.subplots(figsize=paper_figsize("text", aspect=0.43), facecolor="#FFFFFF")

        group_width = 0.8
        bar_w = group_width / n_datasets
        x = np.arange(n_methods)

        for ds_idx, dataset in enumerate(datasets):
            ds_data = ablation_df[ablation_df["dataset"] == dataset].set_index("method")
            ds_color = DATASET_COLORS.get(dataset, "#6B7280")
            is_stress_dataset = "dsifn" in str(dataset).lower()
            offsets = (ds_idx - (n_datasets - 1) / 2.0) * bar_w
            vals = [float(ds_data.loc[m, "error_recall"]) if m in ds_data.index else 0.0 for m in methods]
            bars = ax.bar(
                x + offsets, vals,
                width=bar_w * 0.92,
                color=_muted(ds_color) if is_stress_dataset else ds_color,
                edgecolor="#23313F",
                linewidth=0.9,
                label=display_dataset_name(dataset),
                alpha=0.95 if is_stress_dataset else 0.88,
            )

            if is_stress_dataset:
                for bar_patch in bars:
                    bar_patch.set_hatch("///")

            for bar_patch, val in zip(bars, vals):
                ax.text(
                    bar_patch.get_x() + bar_patch.get_width() / 2.0,
                    val + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.0, color="#000000",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right", rotation_mode="anchor", fontsize=8.2)
        ax.set_ylim(0, 1.14)
        ax.set_ylabel(f"Error Recall @ {budget_label} Budget", fontsize=9.0)
        ax.grid(True, axis="y", alpha=0.18, linestyle=":")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(axis="y", labelsize=8.4)

        # Highlight the CRS-family region rather than a single variant.
        if "CRS-1" in methods and "CRS-4" in methods:
            crs1_x = methods.index("CRS-1")
            crs4_x = methods.index("CRS-4")
            ax.axvspan(crs1_x - 0.45, crs4_x + 0.45, color="#2A9D8F", alpha=0.035, zorder=0)

        # Separator between baselines and CRS variants
        if "CRS-1" in methods and "Margin" in methods:
            sep = (methods.index("CRS-1") + methods.index("Margin")) / 2.0
            ax.axvline(sep + 0.5, color="#E5E7EB", linewidth=1.5, linestyle=":")
            ax.text(sep + 0.55, 1.06, "CRS family ->", fontsize=8.0, color="#444444")

        # Dataset legend patches (fill color = dataset)
        has_dsifn_abl = any("dsifn" in str(d).lower() for d in datasets)
        ds_patches = [
            mpatches.Patch(
                facecolor=_muted(DATASET_COLORS.get(d, "#6B7280")) if "dsifn" in str(d).lower() else DATASET_COLORS.get(d, "#6B7280"),
                edgecolor="#23313F",
                linewidth=0.9,
                hatch="///" if "dsifn" in str(d).lower() else None,
                label=display_dataset_name(d),
            )
            for d in datasets
        ]
        fig.suptitle(
            f"CRS Ablation - Error Recall @ {budget_label} ({model_name})",
            fontsize=10.2, fontweight="bold", color="#000000", y=0.98,
        )
        fig.legend(
            handles=ds_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=max(1, len(ds_patches)),
            fontsize=8.0,
            frameon=False,
        )
        if has_dsifn_abl:
            fig.text(
                0.5, -0.04, DSIFN_FOOTNOTE,
                ha="center", fontsize=7.0, color="#444444", style="italic",
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.84))
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path


def _cost_accuracy_methods(runtime_df: pd.DataFrame, dataset: str) -> list[tuple[str, str, float, bool]]:
    rt = runtime_df[runtime_df["dataset"].astype(str) == dataset]
    cost = rt.set_index("mode")["cost_multiplier"].astype(float).to_dict()
    cost_one = cost.get("deterministic", 1.0)
    cost_full = cost.get("full", 14.0)

    # Margin, entropy, and CRS-3 all use the same deterministic probability map.
    # Keep them at one-forward-pass cost even if fast-mode timing is lower due to
    # measurement variance.
    return [
        ("Entropy", "score_entropy", cost_one, False),
        ("Margin", "score_margin", cost_one, False),
        ("CRS-3", "score_crs3", cost_one, True),
        ("CRS-4", "score_crs4", cost_full, False),
    ]


def plot_cost_accuracy(
    components_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    out_path: str | Path,
    *,
    model_name: str = "BIT",
    dataset: str = "levir-256",
) -> Path:
    """Inference cost vs. error-detection accuracy.

    Makes the deployment argument visual: CRS-3 reaches the best Error AUROC
    at one-forward-pass cost, whereas CRS-4 pays a large cost multiplier for
    no AUROC gain. Single-map scores are plotted by required forward-pass
    class, so measurement variance in the fast runtime row cannot make CRS-3
    appear cheaper than entropy or margin.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    df = components_df[components_df["dataset"].astype(str) == dataset].copy()
    if df.empty:
        df = components_df.copy()
    labels_bin = df["is_error_component"].astype(int).to_numpy()

    # (label, score column, cost multiplier, recommended default?)
    methods = _cost_accuracy_methods(runtime_df, dataset)

    with paper_style(out_path):
        fig, ax = plt.subplots(figsize=paper_figsize("column", aspect=0.84),
                               facecolor="#FFFFFF")

        # Shade the "single forward pass" (near-baseline cost) region.
        ax.axvspan(0.55, 1.25, color="#2A9D8F", alpha=0.09, zorder=0)

        for label, col, cm, is_default in methods:
            if col not in df.columns:
                continue
            auroc = float(roc_auc_score(labels_bin, df[col].to_numpy(dtype=float)))
            if is_default:
                ax.scatter(cm, auroc, s=210, marker="*", color="#2A9D8F",
                           edgecolor="#15403A", linewidth=1.1, zorder=5)
                ax.annotate(f"{label}  (default)", (cm, auroc),
                            textcoords="offset points", xytext=(12, -5),
                            fontsize=8.2, fontweight="bold", color="#15403A")
            else:
                ax.scatter(cm, auroc, s=66, color="#5B6B7A",
                           edgecolor="#23313F", linewidth=0.9, zorder=4)
                ax.annotate(label, (cm, auroc),
                            textcoords="offset points", xytext=(8, 5),
                            fontsize=7.7, color="#23313F")

        ax.set_xscale("log")
        ax.set_xlabel("Inference cost  ($\\times$ deterministic forward pass)")
        ax.set_ylabel("Error detection AUROC")
        ax.grid(True, which="both", axis="both", alpha=0.16, linestyle=":")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xticklabels(["1$\\times$", "2$\\times$", "5$\\times$", "10$\\times$"])
        # Head/foot room so the star annotation clears the title.
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(lo - 0.07 * span, hi + 0.22 * span)
        ax.text(0.83, lo - 0.03 * span, "near-baseline cost", fontsize=6.0,
                style="italic", color="#2F8A73", ha="center", va="bottom",
                linespacing=1.18)
        ax.set_title(
            f"Cost vs. Accuracy ({model_name}, {display_dataset_name(dataset)})",
            fontsize=9.3, fontweight="bold", color="#000000", pad=8)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04, facecolor="#FFFFFF")
        plt.close(fig)
    return out_path

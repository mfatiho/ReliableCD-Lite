from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_calibration_before_after(metrics_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    stages = metrics_df["stage"].tolist()

    axes[0].bar(stages, metrics_df["ece"], color=["#9fb3c8", "#356a9a"])
    axes[0].set_title("ECE Before/After")
    axes[0].set_ylabel("ECE")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(stages, metrics_df["brier"], color=["#d8c3a5", "#9c6644"])
    axes[1].set_title("Brier Before/After")
    axes[1].set_ylabel("Brier Score")
    axes[1].grid(True, axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Temperature Scaling Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def summarize_components_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "model_name": "",
                    "dataset": "",
                    "num_images": 0,
                    "num_components": 0,
                    "num_error_components": 0,
                    "error_component_rate": 0.0,
                    "mean_area": 0.0,
                    "median_area": 0.0,
                    "mean_entropy": 0.0,
                    "mean_tta": 0.0,
                    "mean_shift": 0.0,
                }
            ]
        )
    return (
        df.groupby(["model_name", "dataset"], dropna=False)
        .agg(
            num_images=("image_id", "nunique"),
            num_components=("component_id", "count"),
            num_error_components=("is_error_component", lambda s: int(s.astype(bool).sum())),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            mean_entropy=("mean_entropy", "mean"),
            mean_tta=("mean_tta", "mean"),
            mean_shift=("mean_shift", "mean"),
        )
        .reset_index()
        .assign(
            error_component_rate=lambda x: np.where(
                x["num_components"] > 0,
                x["num_error_components"] / x["num_components"],
                0.0,
            )
        )
    )


def plot_component_summary(summary_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"{row.model_name}\n{row.dataset}" for row in summary_df.itertuples()]

    axes[0].bar(labels, summary_df["num_components"], color="#3c6e71")
    axes[0].set_title("Component Count")
    axes[0].set_ylabel("Number of Components")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, summary_df["error_component_rate"], color="#d1495b")
    axes[1].set_title("Error Component Rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, max(0.05, float(summary_df["error_component_rate"].max()) * 1.15))
    axes[1].grid(True, axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Component Extraction Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_baseline_metrics(main_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = main_df["score"].tolist()

    axes[0].bar(labels, main_df["error_auroc"], color="#457b9d")
    axes[0].set_title("Error AUROC by Baseline")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, main_df["error_auprc"], color="#f4a261")
    axes[1].set_title("Error AUPRC by Baseline")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Baseline Error-Detection Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def summarize_missed_change(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "model_name": "",
                    "dataset": "",
                    "num_images": 0,
                    "num_gt_components": 0,
                    "num_missed_components": 0,
                    "miss_rate": 0.0,
                    "mean_gt_area": 0.0,
                    "mean_overlap_ratio": 0.0,
                }
            ]
        )
    return (
        df.groupby(["model_name", "dataset"], dropna=False)
        .agg(
            num_images=("image_id", "nunique"),
            num_gt_components=("gt_component_id", "count"),
            num_missed_components=("is_missed", lambda s: int(s.astype(bool).sum())),
            mean_gt_area=("gt_area", "mean"),
            mean_overlap_ratio=("pred_overlap_ratio", "mean"),
        )
        .reset_index()
        .assign(
            miss_rate=lambda x: np.where(
                x["num_gt_components"] > 0,
                x["num_missed_components"] / x["num_gt_components"],
                0.0,
            )
        )
    )


def plot_missed_change_summary(summary_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"{row.model_name}\n{row.dataset}" for row in summary_df.itertuples()]

    axes[0].bar(labels, summary_df["miss_rate"], color="#bc4749")
    axes[0].set_title("Missed-Change Rate")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0.0, max(0.05, float(summary_df["miss_rate"].max()) * 1.15))
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, summary_df["mean_overlap_ratio"], color="#588157")
    axes[1].set_title("Mean GT Overlap Ratio")
    axes[1].set_ylabel("Overlap Ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Missed-Change Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_runtime_modes(runtime_df: pd.DataFrame, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = runtime_df["mode"].tolist()

    axes[0].bar(labels, runtime_df["runtime_ms_mean"], yerr=runtime_df["runtime_ms_std"], color="#6d597a", capsize=4)
    axes[0].set_title("Runtime by Mode")
    axes[0].set_ylabel("Mean Runtime (ms)")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(labels, runtime_df["cost_multiplier"], color="#b56576")
    axes[1].set_title("Relative Cost Multiplier")
    axes[1].set_ylabel("Multiplier")
    axes[1].grid(True, axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Runtime Mode Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

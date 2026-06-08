"""Boundary ring-width ablation: Error AUROC vs. ring_width ∈ {1, 2, 3, 5}.

Re-uses saved prob_maps and pred_masks NPZ files — no inference needed.
Outputs a CSV and prints an inline summary table.

Usage:
    conda run -n bitcd python scripts/run_boundary_ring_ablation.py \
        --datasets levir-256 whu-256 \
        --out-dir results/ablation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import roc_auc_score

RING_WIDTHS = [1, 2, 3, 5]
MIN_AREA = 16
THRESHOLD = 0.5

DATASET_NAMES = {
    "levir-256": "LEVIR-CD",
    "whu-256":   "WHU-CD",
    "dsifn-256": "DSIFN-CD†",
}


def _binary_entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _boundary_uncertainty(component_mask: np.ndarray, uncertainty_map: np.ndarray, ring_width: int) -> float:
    dilated = ndimage.binary_dilation(component_mask, iterations=ring_width)
    ring = dilated & ~component_mask
    if ring.sum() == 0:
        return float(uncertainty_map[component_mask].mean())
    return float(uncertainty_map[ring].mean())


def _extract_components(pred_mask: np.ndarray):
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n = ndimage.label(pred_mask, structure=structure)
    sizes = ndimage.sum(pred_mask, labeled, range(1, n + 1))
    return labeled, [i + 1 for i, s in enumerate(sizes) if s >= MIN_AREA]


def process_dataset(dataset_key: str, baseline_dir: Path, parquet_path: Path) -> pd.DataFrame:
    prob_npz  = np.load(baseline_dir / f"bit_{dataset_key}_prob_maps.npz",  allow_pickle=True)
    mask_npz  = np.load(baseline_dir / f"bit_{dataset_key}_pred_masks.npz", allow_pickle=True)

    prob_maps  = prob_npz["prob_maps"]   # (N, H, W) float32
    pred_masks = mask_npz["pred_masks"]  # (N, H, W)
    image_ids  = prob_npz["image_ids"]   # (N,) str

    parquet = pd.read_parquet(parquet_path)
    error_map = (
        parquet.set_index(["image_id", "component_id"])["is_error_component"]
        .astype(bool)
        .to_dict()
    )

    rows = []
    for idx, image_id in enumerate(image_ids):
        prob  = prob_maps[idx].astype(float)
        pmask = (pred_masks[idx] > THRESHOLD).astype(bool) if pred_masks[idx].dtype != bool else pred_masks[idx].astype(bool)
        entropy = _binary_entropy(prob)

        labeled, valid_ids = _extract_components(pmask)

        for cid in valid_ids:
            comp_mask = labeled == cid
            key = (image_id, cid)
            if key not in error_map:
                continue
            is_error = error_map[key]

            row = {"image_id": image_id, "component_id": cid, "is_error": int(is_error)}
            for rw in RING_WIDTHS:
                row[f"bu_rw{rw}"] = _boundary_uncertainty(comp_mask, entropy, rw)
            rows.append(row)

    return pd.DataFrame(rows)


def compute_auroc_table(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    y = df["is_error"].values
    if y.sum() == 0 or (1 - y).sum() == 0:
        return pd.DataFrame()
    results = []
    for rw in RING_WIDTHS:
        col = f"bu_rw{rw}"
        auroc = roc_auc_score(y, df[col].values)
        results.append({"dataset": dataset_label, "ring_width": rw, "error_auroc": round(auroc, 4)})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",     nargs="+", default=["levir-256", "whu-256"])
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--components-dir", default="results/components")
    parser.add_argument("--out-dir",      default="results/ablation")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []

    for dataset_key in args.datasets:
        parquet_path = Path(args.components_dir) / f"bit_{dataset_key}_components.parquet"
        if not parquet_path.exists():
            print(f"[skip] {parquet_path} not found")
            continue

        print(f"Processing {dataset_key}...")
        df = process_dataset(dataset_key, baseline_dir, parquet_path)
        label = DATASET_NAMES.get(dataset_key, dataset_key)
        auroc_df = compute_auroc_table(df, label)
        if not auroc_df.empty:
            all_rows.append(auroc_df)

    if not all_rows:
        print("No results produced.")
        return

    results = pd.concat(all_rows, ignore_index=True)
    out_path = out_dir / "boundary_ring_ablation.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}\n")

    # Pivot for inline display
    pivot = results.pivot(index="ring_width", columns="dataset", values="error_auroc")
    pivot.index.name = "ring_width (px)"
    print(pivot.to_string())
    print()

    # Highlight best ring_width per dataset
    for col in pivot.columns:
        best = pivot[col].idxmax()
        print(f"  {col}: best ring_width = {best} px (AUROC={pivot.loc[best, col]:.4f})")


if __name__ == "__main__":
    main()

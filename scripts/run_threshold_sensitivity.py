from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Allow running as `python scripts/run_threshold_sensitivity.py` without installation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from reliable.components.extraction import extract_components
from reliable.data import canonical_dataset_name, dataset_instance_slug
from reliable.inference.save_maps import load_prediction_bundle
from reliable.metrics.cd_metrics import binary_cd_metrics, binary_dataset_metrics
from reliable.utils.manifest import write_json_manifest
from reliable.utils.progress import make_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6])
    parser.add_argument("--baseline-dir", default="results/baseline")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    model_slug = args.model.lower()
    rows: list[dict[str, object]] = []
    work_items = [(dataset, threshold) for dataset in args.datasets for threshold in args.thresholds]
    progress = make_progress(work_items, total=len(work_items), desc=f"{args.model} threshold sensitivity", unit="setting")
    for dataset_name, threshold in progress:
        slug = dataset_instance_slug(dataset_name)
        bundle = load_prediction_bundle(baseline_dir / f"{model_slug}_{slug}_predictions.npz")
        prob_maps = bundle["prob_maps"].astype(np.float32)
        gt_masks = bundle["gt_masks"].astype(np.uint8)
        pred_masks = (prob_maps >= threshold).astype(np.uint8)
        rows.append(
            _summarize_threshold_setting(
                model_name=args.model,
                dataset=dataset_name,
                dataset_canonical=canonical_dataset_name(dataset_name),
                threshold=threshold,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
            )
        )
    progress.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["dataset", "threshold"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    write_json_manifest(
        out_path.with_suffix(".json"),
        {
            "model_name": args.model,
            "datasets": args.datasets,
            "thresholds": args.thresholds,
            "outputs": {"csv": str(out_path)},
        },
    )


def _summarize_threshold_setting(
    *,
    model_name: str,
    dataset: str,
    dataset_canonical: str,
    threshold: float,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> dict[str, object]:
    per_image_f1 = []
    per_image_iou = []
    component_counts = []
    nonempty_pred_images = 0
    for pred_mask, gt_mask in zip(pred_masks, gt_masks, strict=False):
        metrics = binary_cd_metrics(pred_mask, gt_mask, empty_policy="nan")
        per_image_f1.append(float(metrics["f1"]))
        per_image_iou.append(float(metrics["iou"]))
        labeled, component_count = extract_components(pred_mask.astype(np.uint8), min_area=16)
        _ = labeled
        component_counts.append(int(component_count))
        nonempty_pred_images += int(pred_mask.sum() > 0)
    dataset_metrics = binary_dataset_metrics(pred_masks, gt_masks)
    return {
        "model_name": model_name,
        "dataset": dataset,
        "dataset_canonical": dataset_canonical,
        "threshold": threshold,
        "global_precision": float(dataset_metrics["precision"]),
        "global_recall": float(dataset_metrics["recall"]),
        "global_f1": float(dataset_metrics["f1"]),
        "global_iou": float(dataset_metrics["iou"]),
        "f1_mean": _nanmean(per_image_f1),
        "iou_mean": _nanmean(per_image_iou),
        "num_images": int(len(pred_masks)),
        "num_components": int(sum(component_counts)),
        "mean_components_per_image": float(np.mean(component_counts)) if component_counts else 0.0,
        "nonempty_pred_image_frac": float(nonempty_pred_images / len(pred_masks)) if len(pred_masks) else 0.0,
    }


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


if __name__ == "__main__":
    main()

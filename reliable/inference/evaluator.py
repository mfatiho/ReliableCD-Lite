from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from reliable.data import dataset_instance_slug, make_inference_loader
from reliable.inference.save_maps import save_npz_maps, save_prediction_bundle
from reliable.metrics.cd_metrics import binary_cd_metrics, binary_dataset_metrics
from reliable.utils.experiment import resolve_dataset_loader_kwargs
from reliable.utils.manifest import write_json_manifest
from reliable.utils.progress import make_progress
from reliable.utils.tensors import tensor_to_numpy


@dataclass(slots=True)
class EvalResultPaths:
    metrics_csv: Path
    metrics_summary_csv: Path
    predictions_npz: Path
    prob_maps_npz: Path
    pred_masks_npz: Path
    manifest_json: Path


@torch.no_grad()
def evaluate_adapter_on_dataset(
    adapter,
    dataset_name: str,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    out_dir: str | Path,
    split: str = "test",
    img_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    empty_policy: str = "zero",
    config_path: str | Path = "configs/default.yaml",
    progress_desc: str | None = None,
) -> EvalResultPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader_kwargs = resolve_dataset_loader_kwargs(dataset_name, config_path)
    loader = make_inference_loader(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        with_labels=True,
        **loader_kwargs,
    )

    image_ids: list[str] = []
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    gt_batches: list[np.ndarray] = []
    metric_rows: list[dict[str, object]] = []

    batch_iter = loader
    progress = None
    if progress_desc:
        progress = make_progress(loader, total=len(loader), desc=progress_desc, unit="batch")
        batch_iter = progress

    for batch in batch_iter:
        img_A = batch["A"]
        img_B = batch["B"]
        gt = batch["L"]
        names = list(batch["name"])

        start = perf_counter()
        logits = adapter.predict_logits(img_A, img_B)
        prob = adapter.postprocess_prob(torch.softmax(logits, dim=1)[:, 1:2])
        pred = (prob >= adapter.threshold).to(dtype=torch.uint8)
        runtime_ms = (perf_counter() - start) * 1000.0

        gt_np = tensor_to_numpy(gt > 0, np.uint8)
        prob_np = tensor_to_numpy(prob, np.float32)
        pred_np = tensor_to_numpy(pred, np.uint8)
        logits_np = tensor_to_numpy(logits, np.float32)

        image_ids.extend(names)
        logits_batches.append(logits_np)
        prob_batches.append(prob_np)
        pred_batches.append(pred_np)
        gt_batches.append(gt_np)

        per_image_runtime = runtime_ms / max(len(names), 1)
        for idx, image_id in enumerate(names):
            gt_positive_pixels = int(gt_np[idx, 0].sum())
            pred_positive_pixels = int(pred_np[idx, 0].sum())
            is_empty_empty = gt_positive_pixels == 0 and pred_positive_pixels == 0
            metrics = binary_cd_metrics(pred_np[idx, 0], gt_np[idx, 0], empty_policy=empty_policy)
            metric_rows.append(
                {
                    "model_name": adapter.model_name,
                    "dataset": dataset_name,
                    "image_id": image_id,
                    "f1": metrics["f1"],
                    "iou": metrics["iou"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "oa": metrics["oa"],
                    "threshold": adapter.threshold,
                    "input_size": img_size,
                    "checkpoint_id": str(checkpoint_path),
                    "runtime_ms": per_image_runtime,
                    "gt_positive_pixels": gt_positive_pixels,
                    "pred_positive_pixels": pred_positive_pixels,
                    "empty_empty": is_empty_empty,
                }
            )

    if progress is not None:
        progress.close()

    logits_all = np.concatenate(logits_batches, axis=0)
    prob_all = np.concatenate(prob_batches, axis=0)
    pred_all = np.concatenate(pred_batches, axis=0)
    gt_all = np.concatenate(gt_batches, axis=0)
    slug = dataset_instance_slug(dataset_name)
    model_slug = adapter.model_name.lower()

    metrics_csv = out_dir / f"{model_slug}_{slug}_metrics.csv"
    metrics_summary_csv = out_dir / f"{model_slug}_{slug}_metrics_summary.csv"
    predictions_npz = out_dir / f"{model_slug}_{slug}_predictions.npz"
    prob_maps_npz = out_dir / f"{model_slug}_{slug}_prob_maps.npz"
    pred_masks_npz = out_dir / f"{model_slug}_{slug}_pred_masks.npz"
    manifest_json = out_dir / f"{model_slug}_{slug}_manifest.json"

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_csv, index=False)
    _summarize_metrics(
        metrics_df,
        dataset_name,
        adapter.model_name,
        adapter.threshold,
        empty_policy,
        pred_all[:, 0],
        gt_all[:, 0],
    ).to_csv(metrics_summary_csv, index=False)
    save_prediction_bundle(predictions_npz, image_ids, logits_all, prob_all[:, 0], pred_all[:, 0], gt_all[:, 0])
    save_npz_maps(prob_maps_npz, {"image_ids": np.asarray(image_ids), "prob_maps": prob_all[:, 0]})
    save_npz_maps(pred_masks_npz, {"image_ids": np.asarray(image_ids), "pred_masks": pred_all[:, 0], "gt_masks": gt_all[:, 0]})
    write_json_manifest(
        manifest_json,
        {
            "model_name": adapter.model_name,
            "dataset": dataset_name,
            "dataset_root": str(dataset_root),
            "split": split,
            "checkpoint_path": str(checkpoint_path),
            "device": adapter.device,
            "batch_size": batch_size,
            "img_size": img_size,
            "num_images": len(image_ids),
            "empty_policy": empty_policy,
            "loader_kwargs": loader_kwargs,
        },
    )
    return EvalResultPaths(
        metrics_csv=metrics_csv,
        metrics_summary_csv=metrics_summary_csv,
        predictions_npz=predictions_npz,
        prob_maps_npz=prob_maps_npz,
        pred_masks_npz=pred_masks_npz,
        manifest_json=manifest_json,
    )


def _summarize_metrics(
    metrics_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    threshold: float,
    empty_policy: str,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
) -> pd.DataFrame:
    policy_label = {"zero": "zero", "one": "one", "nan": "excluded"}[empty_policy]
    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "threshold": threshold,
        "empty_policy": policy_label,
        "num_images": int(len(metrics_df)),
        "empty_empty_count": int(metrics_df["empty_empty"].sum()) if len(metrics_df) else 0,
        "empty_empty_frac": float(metrics_df["empty_empty"].mean()) if len(metrics_df) else 0.0,
    }
    for metric_name in ("f1", "iou", "precision", "recall", "oa", "runtime_ms"):
        values = metrics_df[metric_name].to_numpy(dtype=float)
        summary[f"{metric_name}_mean"] = _nan_reduction(values, np.nanmean)
        summary[f"{metric_name}_median"] = _nan_reduction(values, np.nanmedian)
        summary[f"{metric_name}_std"] = _nan_reduction(values, np.nanstd)
    dataset_metrics = binary_dataset_metrics(pred_masks, gt_masks)
    summary.update(
        {
            "global_precision": float(dataset_metrics["precision"]),
            "global_recall": float(dataset_metrics["recall"]),
            "global_f1": float(dataset_metrics["f1"]),
            "global_iou": float(dataset_metrics["iou"]),
            "global_oa": float(dataset_metrics["oa"]),
        }
    )
    return pd.DataFrame([summary])


def _nan_reduction(values: np.ndarray, reducer) -> float:
    if len(values) == 0 or np.isnan(values).all():
        return float("nan")
    return float(reducer(values))

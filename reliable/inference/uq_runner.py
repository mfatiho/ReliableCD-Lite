from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reliable.data import dataset_instance_slug, make_inference_loader
from reliable.inference.save_maps import save_npz_maps
from reliable.uq.entropy import binary_entropy
from reliable.uq.mc_dropout import MCDropoutEstimator
from reliable.uq.shift_sensitivity import ShiftSensitivity
from reliable.uq.tta import TTAEstimator
from reliable.utils.experiment import resolve_dataset_loader_kwargs
from reliable.utils.manifest import write_json_manifest
from reliable.utils.progress import make_progress
from reliable.utils.tensors import tensor_to_numpy

SUPPORTED_UQ_MODES = {"entropy", "mi", "tta", "shift"}


@dataclass(slots=True)
class UQResultPaths:
    manifest_json: Path
    output_paths: dict[str, Path]


def _append_map(buffers: dict[str, list[np.ndarray]], key: str, tensor: torch.Tensor) -> None:
    buffers[key].append(tensor_to_numpy(tensor, np.float32)[:, 0])


def run_uq_on_dataset(
    adapter,
    dataset_name: str,
    dataset_root: str | Path,
    modes: list[str],
    out_dir: str | Path,
    split: str = "test",
    img_size: int = 256,
    batch_size: int = 1,
    num_workers: int = 0,
    mc_passes: int = 20,
    config_path: str | Path = "configs/default.yaml",
) -> UQResultPaths:
    return _run_uq_on_dataset(
        adapter=adapter,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        modes=modes,
        out_dir=out_dir,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mc_passes=mc_passes,
        config_path=config_path,
    )


@torch.inference_mode()
def _run_uq_on_dataset(
    adapter,
    dataset_name: str,
    dataset_root: str | Path,
    modes: list[str],
    out_dir: str | Path,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    mc_passes: int,
    config_path: str | Path,
) -> UQResultPaths:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader_kwargs = resolve_dataset_loader_kwargs(dataset_name, config_path)
    loader = make_inference_loader(
        dataset_root=dataset_root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        with_labels=False,
        **loader_kwargs,
    )

    unknown_modes = [mode for mode in modes if mode not in SUPPORTED_UQ_MODES]
    if unknown_modes:
        raise ValueError(
            f"Unsupported UQ mode(s): {unknown_modes}. "
            f"Supported modes are: {sorted(SUPPORTED_UQ_MODES)}."
        )

    allowed_modes = []
    skipped_modes = []
    for mode in modes:
        if mode == "mi" and not adapter.supports_mc_dropout:
            skipped_modes.append(mode)
            continue
        allowed_modes.append(mode)

    tta_estimator = TTAEstimator() if "tta" in allowed_modes else None
    shift_estimator = ShiftSensitivity() if "shift" in allowed_modes else None
    mc_estimator = MCDropoutEstimator(adapter, n_passes=mc_passes) if "mi" in allowed_modes else None

    image_ids: list[str] = []
    buffers: dict[str, list[np.ndarray]] = {
        "entropy_maps": [],
        "mi_maps": [],
        "mi_mean_prob": [],
        "tta_maps": [],
        "tta_mean_prob": [],
        "shift_maps": [],
        "shift_base_prob": [],
    }
    progress_bars = _create_mode_progress_bars(adapter.model_name, dataset_name, allowed_modes, total=len(loader))
    try:
        for batch in loader:
            img_A = batch["A"].to(adapter.device, non_blocking=True)
            img_B = batch["B"].to(adapter.device, non_blocking=True)
            names = list(batch["name"])
            image_ids.extend(names)

            base_prob: torch.Tensor | None = None
            if "entropy" in allowed_modes or "shift" in allowed_modes:
                base_prob = adapter.predict_prob(img_A, img_B)
            if "entropy" in allowed_modes:
                _append_map(buffers, "entropy_maps", binary_entropy(base_prob))
                progress_bars["entropy"].update(1)
            if "mi" in allowed_modes and mc_estimator is not None:
                mi_out = mc_estimator(img_A, img_B)
                _append_map(buffers, "mi_maps", mi_out["mutual_information"])
                _append_map(buffers, "mi_mean_prob", mi_out["mean_prob"])
                progress_bars["mi"].update(1)
            if "tta" in allowed_modes and tta_estimator is not None:
                tta_out = tta_estimator(adapter, img_A, img_B)
                _append_map(buffers, "tta_maps", tta_out["disagreement"])
                _append_map(buffers, "tta_mean_prob", tta_out["mean_prob"])
                progress_bars["tta"].update(1)
            if "shift" in allowed_modes and shift_estimator is not None:
                shift_out = shift_estimator(adapter, img_A, img_B, base_prob=base_prob)
                _append_map(buffers, "shift_maps", shift_out["shift_sensitivity"])
                _append_map(buffers, "shift_base_prob", shift_out["base_prob"])
                progress_bars["shift"].update(1)
    finally:
        _close_progress_bars(progress_bars)

    slug = dataset_instance_slug(dataset_name)
    model_slug = adapter.model_name.lower()
    output_paths: dict[str, Path] = {}
    for key, chunks in buffers.items():
        if not chunks:
            continue
        bundle = np.concatenate(chunks, axis=0) if chunks else np.empty((0, img_size, img_size), dtype=np.float32)
        path = out_dir / f"{model_slug}_{slug}_{key}.npz"
        save_npz_maps(path, {"image_ids": np.asarray(image_ids), key: bundle})
        output_paths[key] = path

    summary_rows = []
    for key, path in output_paths.items():
        data = np.load(path)
        values = data[key]
        summary_rows.append(
            {
                "model_name": adapter.model_name,
                "dataset": dataset_name,
                "map_name": key,
                "num_images": len(image_ids),
                "mean": float(np.nanmean(values)) if values.size else 0.0,
                "std": float(np.nanstd(values)) if values.size else 0.0,
                "nan_count": int(np.isnan(values).sum()),
                "path": str(path),
            }
        )
    if summary_rows:
        summary_path = out_dir / f"{model_slug}_{slug}_uq_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        output_paths["summary_csv"] = summary_path

    manifest_path = out_dir / f"{model_slug}_{slug}_uq_manifest.json"
    write_json_manifest(
        manifest_path,
        {
            "model_name": adapter.model_name,
            "dataset": dataset_name,
            "dataset_root": str(dataset_root),
            "device": adapter.device,
            "split": split,
            "requested_modes": modes,
            "effective_modes": allowed_modes,
            "skipped_modes": skipped_modes,
            "mc_passes": mc_passes if "mi" in allowed_modes else 0,
            "loader_kwargs": loader_kwargs,
        },
    )
    return UQResultPaths(manifest_json=manifest_path, output_paths=output_paths)


def _create_mode_progress_bars(model_name: str, dataset_name: str, modes: list[str], total: int) -> dict[str, object]:
    bars: dict[str, object] = {}
    for position, mode in enumerate(modes):
        bars[mode] = make_progress(
            total=total,
            desc=f"{model_name}/{dataset_name} {mode}",
            unit="batch",
            position=position,
        )
    return bars


def _close_progress_bars(progress_bars: dict[str, object]) -> None:
    for bar in progress_bars.values():
        bar.close()

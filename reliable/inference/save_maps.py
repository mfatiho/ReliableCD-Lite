from __future__ import annotations

from pathlib import Path

import numpy as np


def save_npz_maps(path: str | Path, maps: dict[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **maps)
    return target


def save_prediction_bundle(
    path: str | Path,
    image_ids: list[str],
    logits: np.ndarray,
    prob_maps: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray | None = None,
) -> Path:
    payload: dict[str, np.ndarray] = {
        "image_ids": np.asarray(image_ids),
        "logits": logits,
        "prob_maps": prob_maps,
        "pred_masks": pred_masks,
    }
    if gt_masks is not None:
        payload["gt_masks"] = gt_masks
    return save_npz_maps(path, payload)


def load_prediction_bundle(path: str | Path) -> dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=False)
    return {key: bundle[key] for key in bundle.files}

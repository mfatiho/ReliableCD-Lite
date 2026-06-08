from __future__ import annotations

from pathlib import Path
import re

import torch

from reliable.adapters import BITAdapter, ChangeFormerAdapter
from reliable.data import canonical_dataset_name
from reliable.utils.config import load_yaml


def load_default_paths(config_path: str | Path = "configs/default.yaml") -> dict:
    config = load_yaml(config_path)
    return {
        "datasets": config.get("datasets", {}),
        "paths": config.get("paths", {}),
        "threshold": float(config.get("threshold", 0.5)),
        "device": str(config.get("device", "auto")),
        "preprocessing": config.get("preprocessing", {}),
    }


def _resolve_dataset_config_entry(dataset_name: str, config_path: str | Path = "configs/default.yaml") -> tuple[str, object]:
    config = load_default_paths(config_path)
    datasets_cfg = config["datasets"]
    normalized_name = dataset_name.strip().lower()
    if normalized_name in datasets_cfg:
        return normalized_name, datasets_cfg[normalized_name]

    canonical = canonical_dataset_name(dataset_name)
    key_map = {
        "LEVIR-CD": "levir",
        "WHU-CD": "whu",
        "DSIFN-CD": "dsifn",
    }
    fallback_key = key_map[canonical]
    fallback_candidates = (
        fallback_key,
        f"{fallback_key}-256",
    )
    for candidate in fallback_candidates:
        if candidate in datasets_cfg:
            return candidate, datasets_cfg[candidate]
    raise KeyError(f"Dataset root for {dataset_name} not found in {config_path}.")


def _resolve_dataset_path_from_entry(dataset_name: str, entry: object) -> Path:
    if isinstance(entry, str):
        _reject_windows_drive_relative_path(dataset_name, entry)
        return Path(entry)
    if isinstance(entry, dict) and "path" in entry:
        raw_path = str(entry["path"])
        _reject_windows_drive_relative_path(dataset_name, raw_path)
        return Path(raw_path)
    raise KeyError(
        f"Dataset entry for {dataset_name} must be either a string path or a mapping with a 'path' field."
    )


def _reject_windows_drive_relative_path(dataset_name: str, raw_path: str) -> None:
    if re.match(r"^[A-Za-z]:[^\\/]", raw_path):
        raise ValueError(
            f"Dataset path for {dataset_name} is drive-relative: {raw_path!r}. "
            "Use an absolute Windows path such as 'D:\\datasets\\LEVIR-CD'."
        )


def resolve_device(device: str | None) -> str:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return requested
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:<index>")


def resolve_dataset_root(dataset_name: str, config_path: str | Path = "configs/default.yaml") -> Path:
    _, entry = _resolve_dataset_config_entry(dataset_name, config_path)
    return _resolve_dataset_path_from_entry(dataset_name, entry)


def resolve_dataset_loader_kwargs(dataset_name: str, config_path: str | Path = "configs/default.yaml") -> dict[str, int]:
    _, entry = _resolve_dataset_config_entry(dataset_name, config_path)
    if not isinstance(entry, dict):
        return {}
    preprocessing_cfg = entry.get("preprocessing", {})
    patching_cfg = preprocessing_cfg.get("patching", {})
    if not bool(patching_cfg.get("enabled", False)):
        return {}
    patch_size = int(patching_cfg.get("patch_size", 256))
    patch_stride = int(patching_cfg.get("stride", patch_size))
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("Dataset patching requires positive patch_size and stride values.")
    return {"patch_size": patch_size, "patch_stride": patch_stride}


def instantiate_adapter(
    model_name: str,
    checkpoint_path: str | Path,
    dataset_name: str,
    device: str | None = None,
    threshold: float | None = None,
    bit_repo: str | Path | None = None,
    changeformer_repo: str | Path | None = None,
    config_path: str | Path = "configs/default.yaml",
) -> BITAdapter | ChangeFormerAdapter:
    config = load_default_paths(config_path)
    paths = config["paths"]
    device = resolve_device(device or config["device"])
    threshold = config["threshold"] if threshold is None else threshold
    canonical = canonical_dataset_name(dataset_name)
    model_key = model_name.strip().lower()
    if model_key == "bit":
        return BITAdapter(
            checkpoint_path=str(checkpoint_path),
            device=device,
            threshold=threshold,
            repo_root=str(bit_repo or paths.get("bit_repo", "third_party/BIT_CD")),
        )
    if model_key == "changeformer":
        return ChangeFormerAdapter(
            checkpoint_path=str(checkpoint_path),
            device=device,
            threshold=threshold,
            repo_root=str(changeformer_repo or paths.get("changeformer_repo", "third_party/ChangeFormer")),
            dataset_name=canonical,
        )
    raise KeyError(f"Unsupported model name: {model_name}")

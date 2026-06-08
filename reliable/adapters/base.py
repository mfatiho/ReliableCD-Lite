from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn


class ChangeDetectionAdapter(Protocol):
    """Common interface for binary remote sensing change detection models."""

    model_name: str
    device: str
    threshold: float

    def predict_logits(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (B, 2, H, W) or compatible binary logits."""

    def predict_prob(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """Return change probability map with shape (B, 1, H, W), values in [0, 1]."""

    def predict_mask(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """Return binary change mask with shape (B, 1, H, W)."""

    @property
    def supports_mc_dropout(self) -> bool:
        """Whether dropout-induced predictive variability is supported."""

    def enable_dropout_only(self) -> None:
        """Set only dropout modules to train mode while keeping normalizers frozen."""

    def disable_dropout(self) -> None:
        """Restore deterministic evaluation mode."""


def ensure_repo_path(repo_root: str | Path) -> Path:
    path = Path(repo_root)
    if not path.exists():
        raise FileNotFoundError(f"Expected third-party repository at {path}.")
    return path.resolve()


def validate_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize model output to two-channel binary logits."""
    if logits.ndim != 4:
        raise ValueError(f"Expected 4D logits, got shape {tuple(logits.shape)}.")
    if logits.shape[1] == 2:
        return logits.float()
    if logits.shape[1] == 1:
        one_channel = logits.float()
        return torch.cat([-one_channel, one_channel], dim=1)
    raise ValueError(f"Expected 1 or 2 logit channels, got {logits.shape[1]}.")


def extract_logits_tensor(output: Any) -> torch.Tensor:
    """Extract a logits tensor from model output conventions used by third-party repos."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and output:
        tensor = output[-1]
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise TypeError(f"Unsupported model output type for logits extraction: {type(output)!r}")


def enable_dropout_only(model: nn.Module) -> None:
    """Activate dropout layers while leaving normalizers frozen."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


class AdapterModule(nn.Module):
    """Small adapter base with shared tensor validation helpers."""

    model_name = "unknown"

    def __init__(self, device: str = "cpu", threshold: float = 0.5) -> None:
        super().__init__()
        self.device = device
        self.threshold = threshold

    def preprocess(self, img_A: torch.Tensor, img_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_A = img_A.to(self.device, dtype=torch.float32)
        img_B = img_B.to(self.device, dtype=torch.float32)
        if img_A.shape != img_B.shape:
            raise ValueError(f"Input pair shapes must match, got {tuple(img_A.shape)} and {tuple(img_B.shape)}.")
        if img_A.ndim != 4:
            raise ValueError(f"Expected BCHW tensors, got {tuple(img_A.shape)}.")
        return img_A, img_B

    def postprocess_prob(self, prob: torch.Tensor) -> torch.Tensor:
        if prob.ndim != 4 or prob.shape[1] != 1:
            raise ValueError(f"Probability map must have shape (B, 1, H, W), got {tuple(prob.shape)}.")
        return prob.clamp(0.0, 1.0).to(dtype=torch.float32)

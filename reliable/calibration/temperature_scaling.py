from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reliable.adapters.base import validate_binary_logits


class TemperatureScaler(torch.nn.Module):
    """Single-scalar temperature scaling for binary CD logits."""

    def __init__(self, init_T: float = 1.5) -> None:
        super().__init__()
        self.log_T = torch.nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32)))

    @property
    def T(self) -> torch.Tensor:
        return torch.exp(self.log_T)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.to(logits.device)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        logits = validate_binary_logits(logits).detach()
        self.to(logits.device)
        labels = _flatten_binary_labels(labels).to(logits.device)
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, 2)

        optimizer = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(flat_logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.T.detach().cpu().item())

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"temperature": float(self.T.detach().cpu().item())}, indent=2), encoding="utf-8")
        return target


def _flatten_binary_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError(f"Labels must have shape (B, H, W) or (B, 1, H, W), got {tuple(labels.shape)}.")
    return (labels > 0).to(dtype=torch.long).reshape(-1)

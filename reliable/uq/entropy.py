from __future__ import annotations

import torch


def binary_entropy(prob: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Binary predictive entropy for P(change)."""
    prob = prob.clamp(eps, 1 - eps)
    return -prob * torch.log(prob) - (1 - prob) * torch.log(1 - prob)

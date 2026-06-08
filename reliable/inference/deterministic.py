from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch


@dataclass(slots=True)
class DeterministicPrediction:
    logits: torch.Tensor
    prob: torch.Tensor
    mask: torch.Tensor
    runtime_ms: float


@torch.no_grad()
def deterministic_predict(adapter, img_A: torch.Tensor, img_B: torch.Tensor) -> DeterministicPrediction:
    start = perf_counter()
    logits = adapter.predict_logits(img_A, img_B)
    prob = adapter.predict_prob(img_A, img_B)
    mask = (prob >= adapter.threshold).to(dtype=torch.float32)
    runtime_ms = (perf_counter() - start) * 1000.0
    return DeterministicPrediction(logits=logits, prob=prob, mask=mask, runtime_ms=runtime_ms)

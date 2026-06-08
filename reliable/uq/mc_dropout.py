from __future__ import annotations

import torch

from reliable.uq.entropy import binary_entropy


class MCDropoutEstimator:
    """Dropout-induced predictive variability estimator."""

    def __init__(self, adapter, n_passes: int = 20, inject_p: float = 0.1) -> None:
        self.adapter = adapter
        self.n_passes = n_passes
        self.inject_p = inject_p

    @torch.no_grad()
    def __call__(self, img_A: torch.Tensor, img_B: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.adapter.supports_mc_dropout:
            raise RuntimeError(f"{self.adapter.model_name} does not support required MC Dropout mode.")

        self.adapter.enable_dropout_only()
        probs = []
        for _ in range(self.n_passes):
            probs.append(self.adapter.predict_prob(img_A, img_B))
        self.adapter.disable_dropout()

        probs_tensor = torch.stack(probs, dim=0)
        mean_prob = probs_tensor.mean(dim=0)
        pred_entropy = binary_entropy(mean_prob)
        expected_entropy = binary_entropy(probs_tensor).mean(dim=0)
        mi = torch.clamp(pred_entropy - expected_entropy, min=0.0)
        return {
            "all_probs": probs_tensor,
            "mean_prob": mean_prob,
            "predictive_entropy": pred_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mi,
        }

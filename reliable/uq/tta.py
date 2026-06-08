from __future__ import annotations

import torch


def apply_aug(x: torch.Tensor, aug: str) -> torch.Tensor:
    if aug == "identity":
        return x
    if aug == "hflip":
        return torch.flip(x, dims=(-1,))
    if aug == "vflip":
        return torch.flip(x, dims=(-2,))
    if aug == "rot90":
        return torch.rot90(x, k=1, dims=(-2, -1))
    if aug == "rot180":
        return torch.rot90(x, k=2, dims=(-2, -1))
    if aug == "rot270":
        return torch.rot90(x, k=3, dims=(-2, -1))
    raise ValueError(f"Unsupported augmentation: {aug}")


def invert_aug(x: torch.Tensor, aug: str) -> torch.Tensor:
    inverse_map = {
        "identity": "identity",
        "hflip": "hflip",
        "vflip": "vflip",
        "rot90": "rot270",
        "rot180": "rot180",
        "rot270": "rot90",
    }
    return apply_aug(x, inverse_map[aug])


class TTAEstimator:
    """Test-time augmentation disagreement estimator."""

    def __init__(self, augmentations: list[str] | None = None) -> None:
        self.augmentations = augmentations or ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]

    @torch.no_grad()
    def __call__(self, adapter, img_A: torch.Tensor, img_B: torch.Tensor) -> dict[str, torch.Tensor]:
        aligned_probs = []
        for aug in self.augmentations:
            A_aug = apply_aug(img_A, aug)
            B_aug = apply_aug(img_B, aug)
            p_aug = adapter.predict_prob(A_aug, B_aug)
            aligned_probs.append(invert_aug(p_aug, aug))

        stack = torch.stack(aligned_probs, dim=0)
        return {
            "all_probs": stack,
            "mean_prob": stack.mean(dim=0),
            "disagreement": stack.var(dim=0, unbiased=False),
        }

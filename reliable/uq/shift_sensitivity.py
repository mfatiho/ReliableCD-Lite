from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_tensor(x: torch.Tensor, dy: int, dx: int, mode: str = "replicate") -> torch.Tensor:
    """Shift tensor spatially while preserving shape."""
    if x.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}.")
    _, _, H, W = x.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    y0 = pad_bottom
    x0 = pad_right
    return padded[:, :, y0 : y0 + H, x0 : x0 + W]


class ShiftSensitivity:
    """Output-level co-registration sensitivity."""

    def __init__(
        self,
        shift_pixels: int = 1,
        directions: list[tuple[int, int]] | None = None,
        shift_image: str = "A",
    ) -> None:
        self.shift_pixels = shift_pixels
        self.directions = directions or [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.shift_image = shift_image

    @torch.no_grad()
    def __call__(
        self,
        adapter,
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        base_prob: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        p0 = base_prob if base_prob is not None else adapter.predict_prob(img_A, img_B)
        diffs: list[torch.Tensor] = []
        for dy, dx in self.directions:
            dy *= self.shift_pixels
            dx *= self.shift_pixels
            if self.shift_image == "A":
                shifted_A = shift_tensor(img_A, dy=dy, dx=dx)
                shifted_B = img_B
            elif self.shift_image == "B":
                shifted_A = img_A
                shifted_B = shift_tensor(img_B, dy=dy, dx=dx)
            else:
                raise ValueError("shift_image must be 'A' or 'B'.")
            shifted_prob = adapter.predict_prob(shifted_A, shifted_B)
            aligned_prob = shift_tensor(shifted_prob, dy=-dy, dx=-dx)
            diffs.append(torch.abs(p0 - aligned_prob))

        score = torch.stack(diffs, dim=0).mean(dim=0)
        return {
            "base_prob": p0,
            "shift_sensitivity": score,
            "directional_diffs": diffs,
        }

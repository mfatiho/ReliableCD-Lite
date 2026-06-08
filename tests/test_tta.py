from __future__ import annotations

import torch

from reliable.adapters import BITAdapter
from reliable.uq.tta import TTAEstimator, apply_aug, invert_aug
from tests.test_adapters import DummyCDModel, make_pair


def test_inverse_transform_restores_original_shape() -> None:
    x = torch.arange(1 * 1 * 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    for aug in ["identity", "hflip", "vflip", "rot90", "rot180", "rot270"]:
        restored = invert_aug(apply_aug(x, aug), aug)
        assert restored.shape == x.shape
        assert torch.equal(restored, x)


def test_tta_variance_shape_valid() -> None:
    adapter = BITAdapter(checkpoint_path="unused.pt", model=DummyCDModel())
    img_A, img_B = make_pair()
    out = TTAEstimator()(adapter, img_A, img_B)
    assert out["disagreement"].shape == (2, 1, 16, 16)

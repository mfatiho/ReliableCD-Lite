from __future__ import annotations

import torch

from reliable.adapters import BITAdapter
from reliable.uq.shift_sensitivity import ShiftSensitivity
from tests.test_adapters import DummyCDModel, make_pair


def test_shift_sensitivity_shape() -> None:
    adapter = BITAdapter(checkpoint_path="unused.pt", model=DummyCDModel())
    est = ShiftSensitivity()
    img_A, img_B = make_pair()
    out = est(adapter, img_A, img_B)
    assert out["shift_sensitivity"].shape == adapter.predict_prob(img_A, img_B).shape
    assert torch.all(out["shift_sensitivity"] >= 0)


def test_shift_sensitivity_has_no_backward_dependency() -> None:
    adapter = BITAdapter(checkpoint_path="unused.pt", model=DummyCDModel())
    img_A, img_B = make_pair()
    out = ShiftSensitivity()(adapter, img_A, img_B)
    assert "shift_sensitivity" in out


def test_shift_sensitivity_accepts_precomputed_base_prob() -> None:
    adapter = BITAdapter(checkpoint_path="unused.pt", model=DummyCDModel())
    img_A, img_B = make_pair()
    base_prob = adapter.predict_prob(img_A, img_B)
    out = ShiftSensitivity()(adapter, img_A, img_B, base_prob=base_prob)
    assert out["base_prob"].shape == base_prob.shape

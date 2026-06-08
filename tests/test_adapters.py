from __future__ import annotations

import torch
from torch import nn

from reliable.adapters import BITAdapter, ChangeFormerAdapter
from reliable.adapters.bit_adapter import _infer_bit_net_name


class DummyCDModel(nn.Module):
    def __init__(self, one_channel: bool = False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.one_channel = one_channel

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        delta = self.dropout(img_B - img_A).mean(dim=1, keepdim=True)
        if self.one_channel:
            return delta
        return torch.cat([-delta, delta], dim=1)


def make_pair() -> tuple[torch.Tensor, torch.Tensor]:
    img_A = torch.zeros(2, 3, 16, 16)
    img_B = torch.ones(2, 3, 16, 16)
    return img_A, img_B


def test_bit_predict_prob_shape_range() -> None:
    adapter = BITAdapter(checkpoint_path="unused.pt", model=DummyCDModel())
    img_A, img_B = make_pair()
    prob = adapter.predict_prob(img_A, img_B)
    assert prob.shape == (2, 1, 16, 16)
    assert 0.0 <= float(prob.min()) <= 1.0
    assert 0.0 <= float(prob.max()) <= 1.0


def test_changeformer_predict_prob_shape_range() -> None:
    adapter = ChangeFormerAdapter(checkpoint_path="unused.pth", model=DummyCDModel(one_channel=True))
    img_A, img_B = make_pair()
    prob = adapter.predict_prob(img_A, img_B)
    assert prob.shape == (2, 1, 16, 16)
    assert 0.0 <= float(prob.min()) <= 1.0
    assert 0.0 <= float(prob.max()) <= 1.0


def test_changeformer_only_levir_guard() -> None:
    try:
        ChangeFormerAdapter(checkpoint_path="unused.pth", dataset_name="WHU-CD", model=DummyCDModel())
    except ValueError as exc:
        assert "LEVIR-CD" in str(exc)
    else:
        raise AssertionError("Expected LEVIR-only scope guard to trigger.")


def test_bit_checkpoint_variant_inference_detects_dedim8() -> None:
    checkpoint = {
        "model_G_state_dict": {
            "transformer_decoder.layers.0.0.fn.fn.to_q.weight": torch.zeros(64, 32),
        }
    }
    assert _infer_bit_net_name(checkpoint) == "base_transformer_pos_s4_dd8_dedim8"


def test_bit_checkpoint_variant_inference_detects_default_decoder_dim() -> None:
    checkpoint = {
        "model_G_state_dict": {
            "transformer_decoder.layers.0.0.fn.fn.to_q.weight": torch.zeros(512, 32),
        }
    }
    assert _infer_bit_net_name(checkpoint) == "base_transformer_pos_s4_dd8"

from __future__ import annotations

import pandas as pd
import torch

from reliable.metrics.runtime import _extract_inputs, measure_runtime


class _DummyAdapter:
    model_name = "BIT"
    supports_mc_dropout = False

    def predict_prob(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((img_A[:, :1] + img_B[:, :1]) / 2.0)


def test_extract_inputs_supports_dict_batch() -> None:
    batch = {
        "A": torch.zeros((2, 3, 16, 16), dtype=torch.float32),
        "B": torch.ones((2, 3, 16, 16), dtype=torch.float32),
        "name": ["a", "b"],
    }
    img_A, img_B = _extract_inputs(batch, "cpu")
    assert tuple(img_A.shape) == (2, 3, 16, 16)
    assert tuple(img_B.shape) == (2, 3, 16, 16)


def test_measure_runtime_accepts_dict_batches() -> None:
    loader = [
        {
            "A": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
            "B": torch.ones((2, 3, 8, 8), dtype=torch.float32),
            "name": ["img1", "img2"],
        }
        for _ in range(3)
    ]
    runtime_df = measure_runtime(
        adapter=_DummyAdapter(),
        dataloader=loader,
        mode="deterministic",
        warmup_batches=1,
        measure_batches=2,
        device="cpu",
    )
    assert isinstance(runtime_df, pd.DataFrame)
    assert runtime_df.loc[0, "batch_size"] == 2
    assert runtime_df.loc[0, "input_size"] == "8x8"
    assert runtime_df.loc[0, "num_images"] == 4


def test_measure_runtime_accepts_progress_options() -> None:
    loader = [
        {
            "A": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "B": torch.ones((1, 3, 8, 8), dtype=torch.float32),
            "name": ["img1"],
        }
        for _ in range(2)
    ]
    runtime_df = measure_runtime(
        adapter=_DummyAdapter(),
        dataloader=loader,
        mode="deterministic",
        warmup_batches=1,
        measure_batches=1,
        device="cpu",
        progress_desc="BIT/levir-256/deterministic",
        progress_position=1,
    )
    assert isinstance(runtime_df, pd.DataFrame)
    assert runtime_df.loc[0, "num_images"] == 1

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

from reliable.inference.evaluator import evaluate_adapter_on_dataset


class _DummyAdapter:
    model_name = "Dummy"

    def __init__(self) -> None:
        self.device = "cpu"
        self.threshold = 0.5

    def predict_logits(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = img_A.shape
        logits = torch.zeros((batch, 2, height, width), dtype=torch.float32)
        logits[:, 0] = 1.0
        logits[:, 1] = -1.0
        return logits

    def postprocess_prob(self, prob: torch.Tensor) -> torch.Tensor:
        return prob


def test_evaluator_writes_summary_with_empty_policy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "dataset"
        for folder in ["A", "B", "label", "list"]:
            (root / folder).mkdir(parents=True, exist_ok=True)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        label = np.zeros((8, 8), dtype=np.uint8)
        Image.fromarray(image).save(root / "A" / "sample.png")
        Image.fromarray(image).save(root / "B" / "sample.png")
        Image.fromarray(label).save(root / "label" / "sample.png")
        (root / "list" / "test.txt").write_text("sample.png\n", encoding="utf-8")

        out = evaluate_adapter_on_dataset(
            adapter=_DummyAdapter(),
            dataset_name="LEVIR",
            dataset_root=root,
            checkpoint_path="dummy.ckpt",
            out_dir=Path(tmpdir) / "out",
            img_size=8,
            batch_size=1,
            empty_policy="nan",
            progress_desc="Dummy/LEVIR eval",
        )

        metrics_df = pd.read_csv(out.metrics_csv)
        summary_df = pd.read_csv(out.metrics_summary_csv)

        assert bool(metrics_df.loc[0, "empty_empty"])
        assert int(summary_df.loc[0, "empty_empty_count"]) == 1
        assert summary_df.loc[0, "empty_policy"] == "excluded"
        assert np.isnan(summary_df.loc[0, "f1_mean"])
        assert float(summary_df.loc[0, "global_oa"]) == 1.0
        assert "global_precision" in summary_df.columns
        assert "bit_cd_mf1" not in summary_df.columns

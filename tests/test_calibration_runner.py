from __future__ import annotations

import json

import pandas as pd
import torch

from reliable.calibration.calibration_runner import run_temperature_scaling


def test_run_temperature_scaling_supports_device_auto(tmp_path) -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0], [1.0, -2.0]],
                [[-2.0, 1.0], [-1.0, 2.0]],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)

    outputs = run_temperature_scaling(logits, labels, tmp_path, device="auto", max_iter=5)

    metrics = pd.read_csv(outputs["metrics_csv"])
    scaler_payload = json.loads(outputs["scaler_json"].read_text(encoding="utf-8"))

    assert list(metrics["stage"]) == ["before", "after"]
    assert outputs["calibration_plot_pdf"].exists()
    assert "temperature" in scaler_payload
    assert scaler_payload["temperature"] > 0.0

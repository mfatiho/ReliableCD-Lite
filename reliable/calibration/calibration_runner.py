from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from reliable.calibration.temperature_scaling import TemperatureScaler
from reliable.metrics.calibration_metrics import brier_score, expected_calibration_error
from reliable.utils.experiment import resolve_device
from reliable.utils.tensors import tensor_to_numpy
from reliable.visualization.reports import plot_calibration_before_after


def run_temperature_scaling(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str | Path,
    device: str | None = "auto",
    max_iter: int = 50,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    logits = logits.to(resolved_device)
    labels = labels.to(resolved_device)

    scaler = TemperatureScaler().to(resolved_device)
    scaler.fit(logits, labels, max_iter=max_iter)
    scaled_logits = scaler(logits)
    probs_before = tensor_to_numpy(torch.softmax(logits, dim=1)[:, 1]).reshape(-1)
    probs_after = tensor_to_numpy(torch.softmax(scaled_logits, dim=1)[:, 1]).reshape(-1)
    flat_labels = tensor_to_numpy(labels > 0).reshape(-1)
    table = pd.DataFrame(
        [
            {
                "stage": "before",
                "ece": expected_calibration_error(probs_before, flat_labels),
                "brier": brier_score(probs_before, flat_labels),
            },
            {
                "stage": "after",
                "ece": expected_calibration_error(probs_after, flat_labels),
                "brier": brier_score(probs_after, flat_labels),
            },
        ]
    )
    metrics_path = output_dir / "ece_brier_before_after.csv"
    table.to_csv(metrics_path, index=False)
    plot_path = plot_calibration_before_after(table, output_dir / "calibration_before_after.pdf")
    scaler_path = scaler.save(output_dir / "temperature_scaling.json")
    return {"metrics_csv": metrics_path, "calibration_plot_pdf": plot_path, "scaler_json": scaler_path}

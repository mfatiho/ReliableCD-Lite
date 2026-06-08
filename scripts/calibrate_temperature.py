from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from reliable.calibration.calibration_runner import run_temperature_scaling
from reliable.utils.experiment import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", required=True, help="Path to .npz file with a 'logits' array (N, 2, H, W)")
    parser.add_argument("--labels", help="Path to .npz file with a 'labels' array (N, H, W)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--max-iter", default=50, type=int, help="LBFGS iterations for temperature fitting")
    args = parser.parse_args()

    logits_data = np.load(args.logits)
    logits = torch.from_numpy(logits_data["logits"]).float()
    if args.labels:
        labels_data = np.load(args.labels)
        label_key = "labels" if "labels" in labels_data.files else "gt_masks"
        labels = torch.from_numpy(labels_data[label_key]).long()
    else:
        if "gt_masks" not in logits_data.files:
            raise KeyError("When --labels is omitted, the --logits bundle must include 'gt_masks'.")
        labels = torch.from_numpy(logits_data["gt_masks"]).long()

    resolved_device = resolve_device(args.device)
    print(f"Using device: {resolved_device}")
    paths = run_temperature_scaling(
        logits,
        labels,
        Path(args.out_dir),
        device=resolved_device,
        max_iter=args.max_iter,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

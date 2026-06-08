from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reliable.components.missed_change import analyze_missed_gt_components
from reliable.utils.progress import make_progress
from reliable.visualization.reports import plot_missed_change_summary, summarize_missed_change


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--image-id", default="")
    args = parser.parse_args()
    gt_loaded = np.load(args.gt, allow_pickle=False)
    pred_loaded = np.load(args.pred, allow_pickle=False)
    if isinstance(gt_loaded, np.ndarray) and isinstance(pred_loaded, np.ndarray):
        df = analyze_missed_gt_components(gt_loaded, pred_loaded)
        df.insert(0, "image_id", args.image_id)
        df.insert(0, "dataset", args.dataset)
        df.insert(0, "model_name", args.model_name)
    else:
        gt_masks = gt_loaded["gt_masks"] if "gt_masks" in gt_loaded.files else gt_loaded["gt_mask"]
        pred_masks = pred_loaded["pred_masks"] if "pred_masks" in pred_loaded.files else pred_loaded["pred_mask"]
        image_ids = [str(x) for x in (gt_loaded["image_ids"] if "image_ids" in gt_loaded.files else pred_loaded["image_ids"])]
        frames = []
        progress = make_progress(
            enumerate(image_ids),
            total=len(image_ids),
            desc=f"{args.model_name}/{args.dataset} missed-change",
            unit="image",
        )
        for idx, image_id in progress:
            frame = analyze_missed_gt_components(gt_masks[idx], pred_masks[idx])
            frame.insert(0, "image_id", image_id)
            frame.insert(0, "dataset", args.dataset)
            frame.insert(0, "model_name", args.model_name)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary_df = summarize_missed_change(df)
    out_path = Path(args.out)
    summary_df.to_csv(out_path.with_name(f"{out_path.stem}_summary.csv"), index=False)
    plot_missed_change_summary(summary_df, out_path.with_name(f"{out_path.stem}_summary.pdf"))


if __name__ == "__main__":
    main()

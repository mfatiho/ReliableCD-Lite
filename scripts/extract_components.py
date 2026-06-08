from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reliable.components.extraction import extract_components
from reliable.components.features import compute_component_features
from reliable.inference.save_maps import load_prediction_bundle
from reliable.utils.progress import make_progress
from reliable.visualization.reports import plot_component_summary, summarize_components_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-mask", help="Path to .npz with a 'pred_mask' array")
    parser.add_argument("--prob-map", help="Path to .npz with a 'prob_map' array")
    parser.add_argument("--entropy-map", help="Path to .npz with an 'entropy_map' array")
    parser.add_argument("--mi-map", help="Path to .npz with an 'mi_map' array")
    parser.add_argument("--tta-map", help="Path to .npz with a 'tta_map' array")
    parser.add_argument("--shift-map", help="Path to .npz with a 'shift_map' array")
    parser.add_argument("--gt-mask", help="Path to .npz with a 'gt_mask' array (enables oracle labeling)")
    parser.add_argument("--out", required=True, help="Output .parquet path")
    parser.add_argument("--model-name", default="unknown")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--image-id", default="unknown")
    parser.add_argument("--prediction-bundle", help="Dataset-level predictions bundle with image_ids/prob_maps/pred_masks/gt_masks")
    parser.add_argument("--entropy-bundle", help="Dataset-level entropy bundle with image_ids and entropy_maps")
    parser.add_argument("--mi-bundle", help="Dataset-level MI bundle with image_ids and mi_maps")
    parser.add_argument("--tta-bundle", help="Dataset-level TTA bundle with image_ids and tta_maps")
    parser.add_argument("--shift-bundle", help="Dataset-level shift bundle with image_ids and shift_maps")
    args = parser.parse_args()

    def load_map(path: str | None, key: str) -> np.ndarray | None:
        if path is None:
            return None
        return np.load(path)[key]

    if args.prediction_bundle:
        bundle = load_prediction_bundle(args.prediction_bundle)
        image_ids = [str(x) for x in bundle["image_ids"]]
        entropy_maps = _load_bundle_array(args.entropy_bundle, "entropy_maps")
        mi_maps = _load_bundle_array(args.mi_bundle, "mi_maps")
        tta_maps = _load_bundle_array(args.tta_bundle, "tta_maps")
        shift_maps = _load_bundle_array(args.shift_bundle, "shift_maps")
        frames: list[pd.DataFrame] = []
        progress = make_progress(
            enumerate(image_ids),
            total=len(image_ids),
            desc=f"{args.model_name}/{args.dataset} components",
            unit="image",
        )
        for idx, image_id in progress:
            labeled, _ = extract_components(bundle["pred_masks"][idx])
            frames.append(
                compute_component_features(
                    labeled,
                    bundle["prob_maps"][idx],
                    entropy_map=None if entropy_maps is None else entropy_maps[idx],
                    mi_map=None if mi_maps is None else mi_maps[idx],
                    tta_map=None if tta_maps is None else tta_maps[idx],
                    shift_map=None if shift_maps is None else shift_maps[idx],
                    gt_mask=bundle["gt_masks"][idx] if "gt_masks" in bundle else None,
                    image_id=image_id,
                    dataset=args.dataset,
                    model_name=args.model_name,
                )
            )
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        summary_df = summarize_components_df(df)
        summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        plot_component_summary(summary_df, out_path.with_name(f"{out_path.stem}_summary.pdf"))
        print(f"Saved {len(df)} components to {out_path}")
        return

    if not args.pred_mask or not args.prob_map:
        parser.error("--pred-mask and --prob-map are required unless --prediction-bundle is provided.")

    pred_mask = np.load(args.pred_mask)["pred_mask"]
    prob_map = np.load(args.prob_map)["prob_map"]
    entropy_map = load_map(args.entropy_map, "entropy_map")
    mi_map = load_map(args.mi_map, "mi_map")
    tta_map = load_map(args.tta_map, "tta_map")
    shift_map = load_map(args.shift_map, "shift_map")
    gt_mask = load_map(args.gt_mask, "gt_mask")

    labeled, _ = extract_components(pred_mask)
    df = compute_component_features(
        labeled,
        prob_map,
        entropy_map=entropy_map,
        mi_map=mi_map,
        tta_map=tta_map,
        shift_map=shift_map,
        gt_mask=gt_mask,
        image_id=args.image_id,
        dataset=args.dataset,
        model_name=args.model_name,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    summary_df = summarize_components_df(df)
    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    plot_component_summary(summary_df, out_path.with_name(f"{out_path.stem}_summary.pdf"))
    print(f"Saved {len(df)} components to {out_path}")


def _load_bundle_array(path: str | None, key: str) -> np.ndarray | None:
    if path is None:
        return None
    return np.load(path, allow_pickle=False)[key]


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from reliable.data import canonical_dataset_name, make_inference_loader
from reliable.metrics.runtime import measure_runtime, recompute_cost_multipliers
from reliable.utils.experiment import instantiate_adapter, resolve_dataset_loader_kwargs, resolve_dataset_root
from reliable.utils.progress import make_progress
from reliable.visualization.reports import plot_runtime_modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--bit-repo", default=None)
    parser.add_argument("--changeformer-repo", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    adapter = instantiate_adapter(
        args.model,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        threshold=args.threshold,
        bit_repo=args.bit_repo,
        changeformer_repo=args.changeformer_repo,
        config_path=args.config,
    )
    loader_kwargs = resolve_dataset_loader_kwargs(args.dataset, args.config)
    loader = make_inference_loader(
        resolve_dataset_root(args.dataset, args.config),
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        with_labels=False,
        **loader_kwargs,
    )
    rows = []
    mode_progress = make_progress(args.modes, total=len(args.modes), desc="Runtime modes", unit="mode")
    for mode in mode_progress:
        runtime_df = measure_runtime(
            adapter,
            loader,
            mode=mode,
            warmup_batches=args.warmup_batches,
            measure_batches=args.measure_batches,
            device=adapter.device,
            progress_desc=f"{args.model}/{args.dataset}/{mode}",
            progress_position=1,
        )
        runtime_df["dataset"] = args.dataset
        runtime_df["dataset_canonical"] = canonical_dataset_name(args.dataset)
        rows.append(runtime_df)
    mode_progress.close()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_df = pd.concat(rows, ignore_index=True)
    runtime_df = recompute_cost_multipliers(runtime_df)
    runtime_df.to_csv(out_path, index=False)
    try:
        plot_runtime_modes(runtime_df, out_path.with_name(f"{out_path.stem}.pdf"))
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        warnings.warn("matplotlib is not installed; skipped runtime PDF generation.", stacklevel=1)


if __name__ == "__main__":
    main()

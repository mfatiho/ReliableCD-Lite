from __future__ import annotations

import argparse

from reliable.inference.evaluator import evaluate_adapter_on_dataset
from reliable.utils.experiment import instantiate_adapter, resolve_dataset_root
from reliable.utils.progress import make_progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-policy", choices=["zero", "one", "nan"], default="zero")
    parser.add_argument("--bit-repo", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    dataset_progress = make_progress(args.datasets, total=len(args.datasets), desc="BIT cross-domain datasets", unit="dataset")
    for dataset_name in dataset_progress:
        adapter = instantiate_adapter(
            "BIT",
            checkpoint_path=args.checkpoint,
            dataset_name=dataset_name,
            device=args.device,
            threshold=args.threshold,
            bit_repo=args.bit_repo,
            config_path=args.config,
        )
        evaluate_adapter_on_dataset(
            adapter=adapter,
            dataset_name=dataset_name,
            dataset_root=resolve_dataset_root(dataset_name, args.config),
            checkpoint_path=args.checkpoint,
            out_dir=args.out_dir,
            split=args.split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            empty_policy=args.empty_policy,
            config_path=args.config,
            progress_desc=f"BIT/{dataset_name} eval",
        )
    dataset_progress.close()


if __name__ == "__main__":
    main()

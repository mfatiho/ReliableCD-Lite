from __future__ import annotations

import argparse

from reliable.inference.evaluator import evaluate_adapter_on_dataset
from reliable.utils.experiment import instantiate_adapter, resolve_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-policy", choices=["zero", "one", "nan"], default="zero")
    parser.add_argument("--changeformer-repo", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    adapter = instantiate_adapter(
        "ChangeFormer",
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        device=args.device,
        threshold=args.threshold,
        changeformer_repo=args.changeformer_repo,
        config_path=args.config,
    )
    evaluate_adapter_on_dataset(
        adapter=adapter,
        dataset_name=args.dataset,
        dataset_root=resolve_dataset_root(args.dataset, args.config),
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        empty_policy=args.empty_policy,
        config_path=args.config,
        progress_desc=f"ChangeFormer/{args.dataset} eval",
    )


if __name__ == "__main__":
    main()

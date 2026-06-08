from __future__ import annotations

import argparse

from reliable.inference.uq_runner import SUPPORTED_UQ_MODES, run_uq_on_dataset
from reliable.utils.experiment import instantiate_adapter, resolve_dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--out-dir", default="results/uq")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bit-repo", default=None)
    parser.add_argument("--changeformer-repo", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mc-passes", type=int, default=20)
    args = parser.parse_args()
    modes = _normalize_modes(args.modes)
    for dataset_name in args.datasets:
        adapter = instantiate_adapter(
            args.model,
            checkpoint_path=args.checkpoint,
            dataset_name=dataset_name,
            device=args.device,
            threshold=args.threshold,
            bit_repo=args.bit_repo,
            changeformer_repo=args.changeformer_repo,
            config_path=args.config,
        )
        run_uq_on_dataset(
            adapter=adapter,
            dataset_name=dataset_name,
            dataset_root=resolve_dataset_root(dataset_name, args.config),
            modes=modes,
            out_dir=args.out_dir,
            split=args.split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mc_passes=args.mc_passes,
            config_path=args.config,
        )

def _normalize_modes(raw_modes: list[str]) -> list[str]:
    modes: list[str] = []
    for raw_mode in raw_modes:
        for part in raw_mode.split(","):
            mode = part.strip().lower()
            if not mode:
                continue
            modes.append(mode)
    deduped: list[str] = []
    seen = set()
    for mode in modes:
        if mode not in seen:
            deduped.append(mode)
            seen.add(mode)
    unknown = [mode for mode in deduped if mode not in SUPPORTED_UQ_MODES]
    if unknown:
        raise ValueError(
            f"Unsupported UQ mode(s): {unknown}. "
            f"Supported modes are: {sorted(SUPPORTED_UQ_MODES)}."
        )
    return deduped


if __name__ == "__main__":
    main()

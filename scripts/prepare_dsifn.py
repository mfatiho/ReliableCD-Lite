from __future__ import annotations

import argparse

from reliable.data.dsifn_prepare import prepare_dsifn_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    prepare_dsifn_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()

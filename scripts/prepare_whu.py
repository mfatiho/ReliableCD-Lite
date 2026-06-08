from __future__ import annotations

import argparse

from reliable.data.whu_prepare import prepare_whu_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    prepare_whu_dataset(args.src, args.dst)


if __name__ == "__main__":
    main()

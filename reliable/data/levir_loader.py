from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetSplitInfo:
    dataset: str
    split: str
    root: Path
    list_file: Path


def levir_split(root: str | Path, split: str = "test") -> DatasetSplitInfo:
    root = Path(root)
    return DatasetSplitInfo(
        dataset="LEVIR-CD",
        split=split,
        root=root,
        list_file=root / "list" / f"{split}.txt",
    )

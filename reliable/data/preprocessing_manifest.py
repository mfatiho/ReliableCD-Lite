from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json


@dataclass(slots=True)
class PreprocessingManifest:
    dataset: str
    source_root: str
    output_root: str
    image_size_original: tuple[int, int] | list[int]
    patch_size: tuple[int, int] | list[int]
    overlap: int
    label_conversion: str
    normalization: str
    num_test_patches: int
    script: str
    created_at: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["created_at"] = self.created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

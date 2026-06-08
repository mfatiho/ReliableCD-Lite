from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json


def write_json_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    target.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return target

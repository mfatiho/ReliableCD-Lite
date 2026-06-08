from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path = ".") -> Path:
    return Path(start).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

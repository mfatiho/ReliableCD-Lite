from __future__ import annotations

from collections.abc import Iterable

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - dependency fallback
    class _tqdm:  # type: ignore[override]
        def __init__(
            self,
            iterable=None,
            *,
            total: int | None = None,
            desc: str = "",
            unit: str = "item",
            dynamic_ncols: bool = True,
            leave: bool = True,
            position: int = 0,
        ) -> None:
            self.iterable = iterable
            self.total = 0 if total is None else total
            self.desc = desc
            self.unit = unit
            self.n = 0
            print(f"{desc}: 0/{self.total} {unit}")

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, delta: int) -> None:
            self.n += delta
            print(f"{self.desc}: {self.n}/{self.total} {self.unit}")

        def close(self) -> None:
            return None


def make_progress(
    iterable: Iterable | None = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str = "item",
    position: int = 0,
    leave: bool = True,
):
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        position=position,
    )

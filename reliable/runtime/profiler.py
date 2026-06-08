from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class RuntimeSample:
    elapsed_ms: float
    peak_memory_mb: float | None = None


class Timer:
    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        self.end = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self.end - self.start) * 1000.0

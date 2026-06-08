from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tests_dir = root / "tests"
    sys.path.insert(0, str(root))
    failures: list[str] = []
    for module_info in pkgutil.iter_modules([str(tests_dir)]):
        if not module_info.name.startswith("test_"):
            continue
        module = importlib.import_module(f"tests.{module_info.name}")
        for name, obj in inspect.getmembers(module):
            if name.startswith("test_") and callable(obj):
                try:
                    obj()
                except Exception as exc:  # pragma: no cover - explicit smoke runner
                    failures.append(f"{module_info.name}.{name}: {exc}")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

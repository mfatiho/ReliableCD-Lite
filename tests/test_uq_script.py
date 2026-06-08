from __future__ import annotations

from scripts.run_uq import _normalize_modes


def test_normalize_modes_supports_comma_separated_input() -> None:
    assert _normalize_modes(["entropy,tta,shift,mi"]) == ["entropy", "tta", "shift", "mi"]


def test_normalize_modes_supports_space_separated_input() -> None:
    assert _normalize_modes(["entropy", "tta", "shift", "mi"]) == ["entropy", "tta", "shift", "mi"]


def test_normalize_modes_rejects_unknown_mode() -> None:
    try:
        _normalize_modes(["entropy,foo"])
    except ValueError as exc:
        assert "Unsupported UQ mode" in str(exc)
    else:
        raise AssertionError("Expected unknown mode validation to fail.")

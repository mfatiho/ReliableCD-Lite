"""Visualization entry points."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from reliable.visualization.panels import (  # noqa: E402
    DATASET_DISPLAY,
    PAPER_RC_PARAMS,
    display_dataset_name,
    paper_style,
)

__all__ = [
    "DATASET_DISPLAY",
    "PAPER_RC_PARAMS",
    "display_dataset_name",
    "paper_style",
]

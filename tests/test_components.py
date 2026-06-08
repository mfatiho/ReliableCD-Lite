from __future__ import annotations

import numpy as np

from reliable.components.boundary import boundary_uncertainty
from reliable.components.extraction import extract_components


def test_connected_component_relabeling_and_min_area_filtering() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[6, 6] = 1
    labeled, count = extract_components(mask, min_area=4)
    assert count == 1
    assert labeled.max() == 1


def test_boundary_uncertainty_fallback() -> None:
    component = np.zeros((3, 3), dtype=bool)
    component[1, 1] = True
    uncertainty = np.ones((3, 3), dtype=float)
    value = boundary_uncertainty(component, uncertainty, ring_width=0)
    assert value == 1.0

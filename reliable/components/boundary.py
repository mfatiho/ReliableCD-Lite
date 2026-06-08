from __future__ import annotations

import numpy as np
from scipy import ndimage


def boundary_uncertainty(component_mask: np.ndarray, uncertainty_map: np.ndarray, ring_width: int = 3) -> float:
    """Mean uncertainty in a boundary ring around a component."""
    component_mask = np.asarray(component_mask).astype(bool)
    uncertainty_map = np.asarray(uncertainty_map, dtype=float)
    if component_mask.sum() == 0:
        return 0.0
    dilated = ndimage.binary_dilation(component_mask, iterations=ring_width)
    ring = np.logical_and(dilated, ~component_mask)
    if ring.sum() == 0:
        return float(uncertainty_map[component_mask].mean())
    return float(uncertainty_map[ring].mean())

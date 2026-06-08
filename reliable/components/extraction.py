from __future__ import annotations

import numpy as np
from scipy import ndimage


def extract_components(pred_mask: np.ndarray, connectivity: int = 8, min_area: int = 16) -> tuple[np.ndarray, int]:
    """Extract connected components from predicted change mask."""
    binary = np.asarray(pred_mask) > 0
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    else:
        raise ValueError("connectivity must be 4 or 8.")

    labeled, count = ndimage.label(binary, structure=structure)
    if count == 0:
        return labeled.astype(np.int32), 0

    out = np.zeros_like(labeled, dtype=np.int32)
    next_id = 1
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if int(component.sum()) < min_area:
            continue
        out[component] = next_id
        next_id += 1
    return out, next_id - 1

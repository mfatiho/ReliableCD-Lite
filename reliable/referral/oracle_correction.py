from __future__ import annotations

import numpy as np


def oracle_correct_pixels(pred_mask: np.ndarray, gt_mask: np.ndarray, review_mask: np.ndarray) -> np.ndarray:
    corrected = np.asarray(pred_mask).copy()
    gt_binary = (np.asarray(gt_mask) > 0).astype(corrected.dtype)
    review_mask = np.asarray(review_mask).astype(bool)
    corrected[review_mask] = gt_binary[review_mask]
    return corrected

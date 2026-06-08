from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert tensors even when PyTorch's direct NumPy bridge is unavailable."""
    cpu_tensor = tensor.detach().cpu()
    try:
        array = cpu_tensor.numpy()
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise
        array = np.asarray(cpu_tensor.tolist())
    if dtype is not None:
        array = array.astype(dtype)
    return array

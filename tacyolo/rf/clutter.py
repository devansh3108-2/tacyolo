from __future__ import annotations

import numpy as np


def subspace_residual(rd_map: np.ndarray, rank: int = 3) -> np.ndarray:
    """Subtract the top-k SVD clutter subspace; return the residual magnitude map."""
    mag = np.abs(rd_map).astype(np.float32)
    # Randomized / economy SVD on the range-Doppler matrix.
    k = max(1, min(int(rank), min(mag.shape) - 1))
    u, s, vt = np.linalg.svd(mag, full_matrices=False)
    clutter = (u[:, :k] * s[:k]) @ vt[:k, :]
    residual = np.clip(mag - clutter, 0.0, None)
    return residual.astype(np.float32)

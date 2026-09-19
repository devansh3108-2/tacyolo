from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_pyramid(rd_map: np.ndarray, sigmas: list[float] | tuple[float, ...] = (1.0, 2.0, 4.0)) -> list[np.ndarray]:
    base = np.abs(rd_map).astype(np.float32)
    return [gaussian_filter(base, sigma=float(s)) for s in sigmas]


def stable_peaks(
    rd_map: np.ndarray,
    sigmas: list[float] | tuple[float, ...] = (1.0, 2.0, 4.0),
    min_rel: float = 0.35,
    max_peaks: int = 8,
) -> list[tuple[int, int, float]]:
    """Peaks that remain local maxima across Gaussian scales."""
    layers = gaussian_pyramid(rd_map, sigmas)
    stacked = np.stack(layers, axis=0)
    # A location is stable if it is a local max on the finest scale and strong on the coarsest.
    fine = layers[0]
    coarse = layers[-1]
    h, w = fine.shape
    thresh = min_rel * float(fine.max() + 1e-6)
    candidates: list[tuple[int, int, float]] = []
    for r in range(1, h - 1):
        for d in range(1, w - 1):
            val = fine[r, d]
            if val < thresh:
                continue
            patch = fine[r - 1 : r + 2, d - 1 : d + 2]
            if val < patch.max():
                continue
            if coarse[r, d] < 0.2 * coarse.max():
                continue
            score = float(stacked[:, r, d].mean())
            candidates.append((r, d, score))
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:max_peaks]

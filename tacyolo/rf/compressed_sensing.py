from __future__ import annotations

import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def fista(
    A: np.ndarray,
    y: np.ndarray,
    lam: float = 0.02,
    n_iter: int = 40,
) -> np.ndarray:
    """FISTA LASSO: min 0.5||Ax-y||^2 + lam||x||_1."""
    y = y.astype(np.float32)
    A = A.astype(np.float32)
    gram = A.T @ A
    # Lipschitz constant: largest eigenvalue of A.T A
    try:
        L = float(np.linalg.eigvalsh(gram)[-1])
    except np.linalg.LinAlgError:
        L = float(np.linalg.norm(A, ord=2) ** 2)
    L = max(L, 1e-6)
    t = 1.0
    x = np.zeros(A.shape[1], dtype=np.float32)
    z = x.copy()
    step = 1.0 / L
    for _ in range(int(n_iter)):
        x_prev = x
        grad = A.T @ (A @ z - y)
        x = soft_threshold(z - step * grad, lam * step).astype(np.float32)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x + ((t - 1.0) / t_next) * (x - x_prev)
        t = float(t_next)
    return x


def recover_range_profile(
    rd_map: np.ndarray,
    sample_frac: float = 0.35,
    lam: float = 0.02,
    n_iter: int = 40,
    seed: int = 0,
) -> np.ndarray:
    """Compressed-sensing recovery of a 1D range profile from random time samples."""
    profile = rd_map.mean(axis=1).astype(np.float32)
    n = profile.shape[0]
    m = max(8, int(n * sample_frac))
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)).astype(np.float32) / np.sqrt(m)
    y = A @ profile
    recovered = fista(A, y, lam=lam, n_iter=n_iter)
    return recovered

from __future__ import annotations

import numpy as np

from tacyolo.rf.clutter import subspace_residual
from tacyolo.rf.compressed_sensing import fista, recover_range_profile
from tacyolo.rf.tda import connected_components, persistence_1d


def test_fista_recovers_sparse_support() -> None:
    rng = np.random.default_rng(0)
    n, m = 64, 28
    x = np.zeros(n, dtype=np.float32)
    x[7] = 3.0
    x[40] = 2.2
    A = rng.standard_normal((m, n)).astype(np.float32) / np.sqrt(m)
    y = A @ x
    x_hat = fista(A, y, lam=0.05, n_iter=80)
    top = set(np.argsort(np.abs(x_hat))[-2:].tolist())
    assert 7 in top
    assert 40 in top


def test_range_profile_recovers_planted_tone() -> None:
    rd = np.zeros((64, 32), dtype=np.float32)
    rd[12, :] = 4.0
    rd += 0.05
    recovered = recover_range_profile(rd, sample_frac=0.5, lam=0.01, n_iter=60, seed=1)
    assert int(np.argmax(np.abs(recovered))) == 12


def test_subspace_clutter_reduces_low_rank_ridge() -> None:
    rd = np.zeros((48, 24), dtype=np.float32)
    rd[:, 12] = 8.0
    rd[20, 4] = 6.0
    residual = subspace_residual(rd, rank=1)
    assert residual[:, 12].mean() < rd[:, 12].mean() * 0.5
    assert residual[20, 4] > residual.mean()


def test_tda_counts_two_clusters() -> None:
    pts = np.array([[0, 0], [0.5, 0.2], [20, 20], [20.4, 19.7]], dtype=np.float32)
    comps = connected_components(pts, eps=2.0)
    assert len(comps) == 2
    pairs = persistence_1d(np.array([0.1, 0.2, 2.0, 2.1], dtype=np.float32))
    assert pairs

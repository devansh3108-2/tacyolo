from __future__ import annotations

import numpy as np


def bipolar(x: np.ndarray) -> np.ndarray:
    signed = np.sign(x)
    signed[signed == 0] = 1
    return signed.astype(np.int8)


class HDEncoder:
    def __init__(self, dim: int = 10000, feat_dim: int = 64, seed: int = 7) -> None:
        self.dim = int(dim)
        self.feat_dim = int(feat_dim)
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((self.dim, self.feat_dim)).astype(np.float32)
        norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        self.proj = proj / norms
        self.radar_id = bipolar(rng.standard_normal(self.dim))

    def encode(self, features: np.ndarray) -> np.ndarray:
        feat = np.asarray(features, dtype=np.float32).ravel()
        if feat.size < self.feat_dim:
            feat = np.pad(feat, (0, self.feat_dim - feat.size))
        else:
            feat = feat[: self.feat_dim]
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        return bipolar(self.proj @ feat)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.int8) * b.astype(np.int8)).astype(np.int8)

    def bundle(self, hvs: list[np.ndarray] | np.ndarray) -> np.ndarray:
        stacked = np.stack([np.asarray(h, dtype=np.float32) for h in hvs], axis=0)
        return bipolar(stacked.sum(axis=0))

    def bind_radar(self, visual: np.ndarray, radar_features: np.ndarray | None) -> np.ndarray:
        if radar_features is None or np.asarray(radar_features).size == 0:
            return visual
        radar_hv = self.encode(radar_features)
        mixed = self.bind(radar_hv, self.radar_id)
        return self.bind(visual, mixed)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

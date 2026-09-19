from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tacyolo.hdc.encode import HDEncoder, cosine


def crop_features(crop: np.ndarray, radar: np.ndarray | None = None) -> np.ndarray:
    """Fixed-length visual descriptor: color histograms + gradient hist + optional RF stats."""
    if crop is None or crop.size == 0:
        visual = np.zeros(56, dtype=np.float32)
    else:
        img = crop
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        feats = []
        for c in range(3):
            hist = cv2.calcHist([small], [c], None, [16], [0, 256]).ravel()
            hist = hist / (hist.sum() + 1e-6)
            feats.append(hist.astype(np.float32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        gh = np.histogram(ang, bins=8, range=(0, 360), weights=mag)[0].astype(np.float32)
        gh = gh / (gh.sum() + 1e-6)
        feats.append(gh)
        visual = np.concatenate(feats).astype(np.float32)
    if radar is None:
        radar = np.zeros(8, dtype=np.float32)
    radar = np.asarray(radar, dtype=np.float32).ravel()
    if radar.size < 8:
        radar = np.pad(radar, (0, 8 - radar.size))
    else:
        radar = radar[:8]
    out = np.concatenate([visual, radar]).astype(np.float32)
    if out.size < 64:
        out = np.pad(out, (0, 64 - out.size))
    return out[:64]


@dataclass
class HDCMatch:
    class_name: str | None
    score: float
    margin: float
    scores: dict[str, float]


class ItemMemory:
    def __init__(
        self,
        dim: int = 10000,
        seed: int = 7,
        min_cosine: float = 0.12,
        embed_fn=None,
        feat_dim: int | None = None,
    ) -> None:
        self.embed_fn = embed_fn
        dim_feat = int(feat_dim) if feat_dim else (256 if embed_fn is not None else 64)
        self.encoder = HDEncoder(dim=dim, feat_dim=dim_feat, seed=seed)
        self.min_cosine = float(min_cosine)
        self.prototypes: dict[str, np.ndarray] = {}

    def _visual_feat(self, crop: np.ndarray, radar: np.ndarray | None = None) -> np.ndarray:
        if self.embed_fn is not None:
            try:
                emb = self.embed_fn(crop)
            except Exception:
                emb = None
            if emb is not None:
                e = np.asarray(emb, dtype=np.float32).ravel()
                need = self.encoder.feat_dim
                if e.size < need:
                    e = np.pad(e, (0, need - e.size))
                return e[:need]
        return crop_features(crop, radar)

    def fit_gallery(self, gallery_dir: str | Path) -> int:
        root = Path(gallery_dir)
        if not root.exists():
            return 0
        n = 0
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            hvs = []
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                feat = self._visual_feat(img)
                hvs.append(self.encoder.encode(feat))
                n += 1
            if hvs:
                self.prototypes[class_dir.name] = self.encoder.bundle(hvs)
        return n

    def add(self, class_name: str, crop: np.ndarray, radar: np.ndarray | None = None) -> None:
        hv = self.encode_observation(crop, radar)
        if class_name in self.prototypes:
            self.prototypes[class_name] = self.encoder.bundle([self.prototypes[class_name], hv])
        else:
            self.prototypes[class_name] = hv

    def encode_observation(self, crop: np.ndarray, radar: np.ndarray | None = None) -> np.ndarray:
        feat = self._visual_feat(crop, radar)
        visual = self.encoder.encode(feat)
        return self.encoder.bind_radar(visual, radar)

    def match(self, crop: np.ndarray, radar: np.ndarray | None = None) -> HDCMatch:
        if not self.prototypes:
            return HDCMatch(None, 0.0, 0.0, {})
        hv = self.encode_observation(crop, radar)
        scores = {name: cosine(hv, proto) for name, proto in self.prototypes.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - second
        if best < self.min_cosine:
            return HDCMatch(None, best, margin, scores)
        return HDCMatch(best_name, best, margin, scores)

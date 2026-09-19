from __future__ import annotations

import numpy as np

from tacyolo.hdc.gallery import seed_default_gallery
from tacyolo.hdc.memory import ItemMemory


def test_hdc_rank1_on_color_gallery(tmp_path) -> None:
    gallery = seed_default_gallery(tmp_path / "gallery")
    mem = ItemMemory(dim=512, seed=3, min_cosine=0.05)
    assert mem.fit_gallery(gallery) >= 6

    red = np.zeros((48, 64, 3), dtype=np.uint8)
    red[:] = (40, 40, 220)
    blue = np.zeros((48, 64, 3), dtype=np.uint8)
    blue[:] = (220, 40, 40)

    red_match = mem.match(red)
    blue_match = mem.match(blue)
    assert red_match.class_name == "red"
    assert blue_match.class_name == "blue"
    assert red_match.score > red_match.scores.get("blue", -1.0)
    assert red_match.score > 0.05


def test_hdc_uses_identity_embedding() -> None:
    def embed_fn(crop: np.ndarray) -> np.ndarray:
        mean = np.mean(crop.reshape(-1, 3), axis=0).astype(np.float32)
        return np.concatenate([mean, mean, mean])

    mem = ItemMemory(dim=256, seed=1, min_cosine=0.01, embed_fn=embed_fn, feat_dim=16)
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((16, 16, 3), dtype=np.uint8)
    blue[:] = (255, 0, 0)
    mem.add("red", red)
    mem.add("blue", blue)
    assert mem.match(red).class_name == "red"
    assert mem.match(blue).class_name == "blue"

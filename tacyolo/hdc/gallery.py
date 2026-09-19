from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def seed_default_gallery(root: str | Path = "data/gallery") -> Path:
    """Write a tiny few-shot color gallery so HDC works without user chips."""
    root = Path(root)
    colors = {
        "red": (40, 40, 220),
        "blue": (220, 40, 40),
        "green": (40, 180, 40),
    }
    rng = np.random.default_rng(0)
    for name, bgr in colors.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            chip = np.zeros((48, 64, 3), dtype=np.uint8)
            chip[:] = bgr
            noise = rng.integers(-12, 13, size=chip.shape, dtype=np.int16)
            chip = np.clip(chip.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.rectangle(chip, (4, 4), (60, 44), (240, 240, 240), 1)
            cv2.imwrite(str(d / f"chip_{i}.png"), chip)
    return root

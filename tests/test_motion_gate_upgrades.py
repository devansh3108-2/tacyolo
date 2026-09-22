"""Tests for GMM Background Modeling and Adaptive Illumination Filtering."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from tacyolo.optical.motion_gate import AdaptiveIlluminationFilter, MotionGate


def test_adaptive_illumination_filter() -> None:
    filt = AdaptiveIlluminationFilter(blur_kernel=15, clip_limit=2.0)
    # Create an image with an artificial illumination gradient (e.g. bright on left, dark on right)
    gray = np.zeros((100, 100), dtype=np.uint8)
    for x in range(100):
        gray[:, x] = int(x * 2.5)

    # Place a high-contrast target square in the dark region
    gray[40:60, 10:30] = 180

    filtered = filt.filter(gray)
    assert filtered.shape == (100, 100)
    assert filtered.dtype == np.uint8
    # The gradient is suppressed while local contrast is maintained
    assert filtered[40:60, 10:30].mean() > filtered[40:60, 70:90].mean()


def test_gmm_motion_gate_rejection_of_weather_noise() -> None:
    gate = MotionGate(
        method="gmm",
        downscale=1.0,
        min_energy=10.0,
        min_blob_area=15,
        use_illum_filter=True,
    )

    # Sequence of frames with simulated camera jitter / illumination noise
    rng = np.random.default_rng(123)
    bg = np.full((120, 160, 3), 60, dtype=np.uint8)

    # Step 1: Initialize background
    res0 = gate(bg.copy())
    assert res0.skipped

    # Step 2: Feed subtle illumination fluctuations (e.g. overcast cloud shimmer)
    for _ in range(5):
        noise = rng.integers(-3, 4, size=bg.shape, dtype=np.int16)
        noisy_frame = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        res = gate(noisy_frame)
        assert res.skipped, "Subtle illumination noise should be absorbed by GMM"

    # Step 3: Insert genuine moving target
    moving_frame = bg.copy()
    moving_frame[40:70, 50:85] = (220, 220, 220)  # Bright vehicle/drone
    res_target = gate(moving_frame)
    assert res_target.triggered, "Moving high-contrast target should trigger GMM gate"
    assert len(res_target.rois) > 0

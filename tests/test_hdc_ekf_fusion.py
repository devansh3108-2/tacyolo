"""Tests for HDC coupled with EKF Spatio-Temporal Motion Vectors under physical cover / occlusion."""
from __future__ import annotations

import numpy as np
import pytest

from tacyolo.track.associate import associate
from tacyolo.track.ekf import KalmanTrack, reset_track_ids
from tacyolo.track.tracker import MultiObjectTracker
from tacyolo.types import Detection


def _make_detection(x1: float, y1: float, x2: float, y2: float, class_name: str = "drone", hdc_hv: np.ndarray | None = None) -> Detection:
    crop = np.full((32, 32, 3), 150, dtype=np.uint8)
    return Detection(
        bbox=(x1, y1, x2, y2),
        conf=0.9,
        class_id=7,
        class_name=class_name,
        crop=crop,
        hdc_hv=hdc_hv,
    )


def test_ekf_hdc_cover_reidentification() -> None:
    reset_track_ids()
    tracker = MultiObjectTracker(dt=1.0, max_age=10)

    # Generate a unique HDC appearance hypervector for the target
    rng = np.random.default_rng(99)
    target_hv = rng.standard_normal(256).astype(np.float32)
    target_hv /= np.linalg.norm(target_hv)

    # Target moving linearly at +15 pixels per step along X: x0=50 -> 65 -> 80
    d0 = _make_detection(50, 50, 70, 70, hdc_hv=target_hv)
    tracker.update([d0])

    d1 = _make_detection(65, 50, 85, 70, hdc_hv=target_hv)
    tracker.predict(dt=1.0)
    tracker.update([d1])

    d2 = _make_detection(80, 50, 100, 70, hdc_hv=target_hv)
    tracker.predict(dt=1.0)
    states = tracker.update([d2])
    assert len(states) == 1
    orig_id = states[0].track_id

    # Simulate object passing behind physical cover for 2 frames (no detections)
    for _ in range(2):
        tracker.predict(dt=1.0)
        tracker.update([])

    # Re-emergence on frame 5: Expected X ~ 80 + 3*15 = 125
    # IoU with previous detection (at 80..100) and new detection (at 125..145) is exactly 0.0!
    d_reemerge = _make_detection(125, 50, 145, 70, hdc_hv=target_hv)
    tracker.predict(dt=1.0)
    final_states = tracker.update([d_reemerge])

    assert len(final_states) == 1
    # Target identity must persist!
    assert final_states[0].track_id == orig_id, f"Expected Track ID {orig_id} to persist through cover, got {final_states[0].track_id}"
    assert final_states[0].time_since_update == 0

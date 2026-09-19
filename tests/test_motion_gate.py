from __future__ import annotations

import numpy as np

from tacyolo.optical.motion_gate import MotionGate


def _static_frames(n: int = 8, w: int = 160, h: int = 120) -> list[np.ndarray]:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    return [frame.copy() for _ in range(n)]


def _moving_blob_frames(n: int = 12, w: int = 160, h: int = 120) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        x = 10 + i * 8
        cv_y = 40
        frame[cv_y : cv_y + 24, x : x + 30] = (40, 40, 220)
        frames.append(frame)
    return frames


def test_static_frames_skip_after_background_init() -> None:
    gate = MotionGate(downscale=1.0, min_energy=2.0, min_blob_area=10)
    results = [gate(f) for f in _static_frames()]
    assert results[0].skipped
    assert all(r.skipped for r in results[1:])
    assert all(r.energy < 2.0 for r in results[1:])


def test_moving_blob_triggers_and_returns_roi() -> None:
    gate = MotionGate(downscale=1.0, min_energy=1.5, min_blob_area=8, roi_pad=2)
    frames = _moving_blob_frames()
    results = [gate(f) for f in frames]
    triggered = [r for r in results if r.triggered]
    assert triggered, "moving blob should fire the gate"
    assert any(r.rois for r in triggered)
    assert triggered[-1].elapsed_ms >= 0.0

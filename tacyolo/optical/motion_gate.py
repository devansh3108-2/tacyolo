from __future__ import annotations

import time

import cv2
import numpy as np

from tacyolo.types import GateResult, ROI


class MotionGate:
    """Two-pass idle monitor: coarse frame-difference energy, then local blobs/ROIs."""

    def __init__(
        self,
        downscale: float = 0.5,
        min_energy: float = 6.0,
        min_blob_area: int = 40,
        roi_pad: int = 16,
        morph_kernel: int = 3,
        diff_history: float = 0.85,
    ) -> None:
        self.downscale = float(downscale)
        self.min_energy = float(min_energy)
        self.min_blob_area = int(min_blob_area)
        self.roi_pad = int(roi_pad)
        self.morph_kernel = int(max(1, morph_kernel))
        self.diff_history = float(diff_history)
        self._bg: np.ndarray | None = None

    def reset(self) -> None:
        self._bg = None

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        scale = self.downscale if 0.1 <= self.downscale < 1.0 else 1.0
        if scale != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return gray, scale

    def __call__(self, frame: np.ndarray) -> GateResult:
        t0 = time.perf_counter()
        gray, scale = self._prepare(frame)
        gray_f = gray.astype(np.float32)

        if self._bg is None:
            self._bg = gray_f
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GateResult(triggered=False, energy=0.0, skipped=True, elapsed_ms=elapsed)

        diff = np.abs(gray_f - self._bg)
        self._bg = self.diff_history * self._bg + (1.0 - self.diff_history) * gray_f
        motion_frac = float((diff > 16.0).mean())
        energy = float(diff.mean()) + 1000.0 * motion_frac

        if energy < self.min_energy:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GateResult(triggered=False, energy=energy, skipped=True, elapsed_ms=elapsed)

        # Pass B: local energy blobs -> ROIs
        norm = np.clip(diff / (diff.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.dilate(binary, k, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / scale
        h, w = frame.shape[:2]
        rois: list[ROI] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_blob_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            roi = ROI(
                x1=int(x * inv) - self.roi_pad,
                y1=int(y * inv) - self.roi_pad,
                x2=int((x + bw) * inv) + self.roi_pad,
                y2=int((y + bh) * inv) + self.roi_pad,
            ).clip(w, h)
            if roi.area > 0:
                rois.append(roi)

        triggered = bool(rois) or energy >= self.min_energy
        elapsed = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            triggered=triggered,
            energy=energy,
            rois=rois,
            mask=binary,
            skipped=not triggered,
            elapsed_ms=elapsed,
        )

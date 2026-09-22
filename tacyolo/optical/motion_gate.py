from __future__ import annotations

import time

import cv2
import numpy as np

from tacyolo.types import GateResult, ROI


class AdaptiveIlluminationFilter:
    """Hardware-efficient edge illumination filter to suppress weather and lighting transients.

    Separates the high-frequency reflectance component (targets) from the low-frequency
    illumination component (cloud shadows, sun angle shifts, auto-exposure, weather shimmer)
    using Retinex-inspired ratio estimation and local contrast normalization.
    """

    def __init__(self, blur_kernel: int = 31, clip_limit: float = 2.0) -> None:
        self.blur_kernel = int(blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    def filter(self, gray: np.ndarray) -> np.ndarray:
        # Step 1: CLAHE normalization of dynamic range
        enhanced = self.clahe.apply(gray)
        # Step 2: Estimate local ambient illumination field
        illumination = cv2.GaussianBlur(enhanced, (self.blur_kernel, self.blur_kernel), 0)
        # Step 3: Illumination ratio normalization (Reflectance = Image / Illumination)
        illum_f = illumination.astype(np.float32) + 1.0
        enh_f = enhanced.astype(np.float32) + 1.0
        ratio = enh_f / illum_f
        # Normalize back to 0..255 range
        norm = np.clip((ratio - 0.5) * 170.0, 0, 255).astype(np.uint8)
        return norm


class MotionGate:
    """Adaptive Background Modeling Motion Gate with hardware-accelerated GMM and Illumination Filter.

    Replaces simple frame-differencing with Gaussian Mixture Models (MOG2) combined with
    adaptive illumination filtering to filter out weather (rain, fog, wind-blown foliage)
    and lighting noise natively on edge hardware targets.
    """

    def __init__(
        self,
        downscale: float = 0.5,
        min_energy: float = 6.0,
        min_blob_area: int = 40,
        roi_pad: int = 16,
        morph_kernel: int = 3,
        diff_history: float = 0.85,
        method: str = "gmm",
        gmm_history: int = 150,
        gmm_var_threshold: float = 25.0,
        use_illum_filter: bool = True,
    ) -> None:
        self.downscale = float(downscale)
        self.min_energy = float(min_energy)
        self.min_blob_area = int(min_blob_area)
        self.roi_pad = int(roi_pad)
        self.morph_kernel = int(max(1, morph_kernel))
        self.diff_history = float(diff_history)
        self.method = str(method or "gmm").lower()
        self.gmm_history = int(gmm_history)
        self.gmm_var_threshold = float(gmm_var_threshold)
        self.use_illum_filter = bool(use_illum_filter)

        self._illum_filter = AdaptiveIlluminationFilter() if self.use_illum_filter else None
        self._gmm: cv2.BackgroundSubtractor | None = None
        self._frame_count = 0
        self._bg: np.ndarray | None = None
        self._init_subtractor()

    def _init_subtractor(self) -> None:
        if self.method == "gmm":
            # detectShadows=False significantly accelerates execution on edge GPUs/CPUs
            self._gmm = cv2.createBackgroundSubtractorMOG2(
                history=self.gmm_history,
                varThreshold=self.gmm_var_threshold,
                detectShadows=False,
            )
        else:
            self._gmm = None

    def reset(self) -> None:
        self._bg = None
        self._frame_count = 0
        self._init_subtractor()

    def _prepare(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        scale = self.downscale if 0.1 <= self.downscale < 1.0 else 1.0
        if scale != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self._illum_filter is not None:
            gray = self._illum_filter.filter(gray)
        return gray, scale

    def __call__(self, frame: np.ndarray) -> GateResult:
        t0 = time.perf_counter()
        gray, scale = self._prepare(frame)
        self._frame_count += 1

        if self._frame_count == 1:
            if self.method == "gmm" and self._gmm is not None:
                self._gmm.apply(gray, learningRate=1.0)
            else:
                self._bg = gray.astype(np.float32)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GateResult(triggered=False, energy=0.0, skipped=True, elapsed_ms=elapsed)

        if self.method == "gmm" and self._gmm is not None:
            # GMM foreground segmentation
            learning_rate = -1 if self._frame_count > 10 else 0.2
            fg_mask = self._gmm.apply(gray, learningRate=learning_rate)
            _, binary = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
            motion_frac = float((binary > 0).mean())
            energy = motion_frac * 1000.0
        else:
            # Fallback simple difference
            gray_f = gray.astype(np.float32)
            if self._bg is None:
                self._bg = gray_f
                elapsed = (time.perf_counter() - t0) * 1000.0
                return GateResult(triggered=False, energy=0.0, skipped=True, elapsed_ms=elapsed)

            diff = np.abs(gray_f - self._bg)
            self._bg = self.diff_history * self._bg + (1.0 - self.diff_history) * gray_f
            motion_frac = float((diff > 16.0).mean())
            energy = float(diff.mean()) + 1000.0 * motion_frac
            norm = np.clip(diff / (diff.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
            _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        if energy < self.min_energy:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GateResult(triggered=False, energy=energy, skipped=True, elapsed_ms=elapsed)

        # Pass B: morphological cleanup & local blobs -> ROIs
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        cleaned = cv2.dilate(cleaned, k, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        triggered = bool(rois)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return GateResult(
            triggered=triggered,
            energy=energy,
            rois=rois,
            mask=cleaned,
            skipped=not triggered,
            elapsed_ms=elapsed,
        )

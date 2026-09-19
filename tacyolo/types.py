from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def clip(self, width: int, height: int) -> "ROI":
        return ROI(
            x1=int(np.clip(self.x1, 0, width - 1)),
            y1=int(np.clip(self.y1, 0, height - 1)),
            x2=int(np.clip(self.x2, 1, width)),
            y2=int(np.clip(self.y2, 1, height)),
        )

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class GateResult:
    triggered: bool
    energy: float
    rois: list[ROI] = field(default_factory=list)
    mask: np.ndarray | None = None
    skipped: bool = False
    elapsed_ms: float = 0.0


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    conf: float
    class_id: int
    class_name: str
    hdc_class: str | None = None
    hdc_score: float | None = None
    hdc_margin: float | None = None
    crop: np.ndarray | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.bbox

    @property
    def cxcy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5


@dataclass
class RadarPeak:
    range_bin: float
    doppler_bin: float
    energy: float
    bearing: float = 0.0


@dataclass
class RFResult:
    rd_map: np.ndarray
    residual: np.ndarray
    recovered: np.ndarray
    peaks: list[RadarPeak] = field(default_factory=list)
    energy: float = 0.0
    betti0: int = 0
    persistence: list[tuple[float, float]] = field(default_factory=list)
    features: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    elapsed_ms: float = 0.0


@dataclass
class TrackState:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    class_name: str
    hdc_class: str | None
    hdc_score: float | None
    hits: int
    age: int
    time_since_update: int
    confirmed: bool
    covariance: np.ndarray | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x - self.w * 0.5, self.y - self.h * 0.5, self.x + self.w * 0.5, self.y + self.h * 0.5


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    frame: np.ndarray
    gate: GateResult
    rf: RFResult | None
    detections: list[Detection] = field(default_factory=list)
    tracks: list[TrackState] = field(default_factory=list)
    yolo_ran: bool = False
    duty_mode: str = "idle"
    duty_reason: str = ""
    yolo_frac: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

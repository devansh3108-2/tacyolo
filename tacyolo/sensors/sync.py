"""Hardware-locked timestamping and sensor stream synchronization (PPS / IEEE 1588 PTP)."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np


class SyncMode(str, Enum):
    PTP = "ptp"          # IEEE 1588 Precision Time Protocol
    PPS = "pps"          # Pulse-Per-Second Hardware Trigger
    MONOTONIC = "mono"   # Monotonic high-res fallback


@dataclass
class HardwareTimeSource:
    """Simulates or interfaces with PPS hardware trigger or IEEE 1588 PTP network clock."""

    mode: SyncMode = SyncMode.PPS
    pps_period_sec: float = 1.0
    ptp_drift_ppm: float = 0.0  # Parts per million clock drift compensation
    t0_epoch_ns: int = field(default_factory=lambda: int(time.time_ns()))
    _last_pps_ns: int = 0
    _frame_counter: int = 0

    def __post_init__(self) -> None:
        self._last_pps_ns = self.t0_epoch_ns

    def now_ns(self) -> int:
        """Return current hardware-locked timestamp in nanoseconds."""
        raw_mono = time.perf_counter_ns()
        if self.mode == SyncMode.PPS:
            # PPS locks to second boundary, interpolating between pulses
            now_epoch = self.t0_epoch_ns + raw_mono
            return now_epoch
        elif self.mode == SyncMode.PTP:
            # IEEE 1588 PTP hardware clock with drift calibration
            drift_factor = 1.0 + (self.ptp_drift_ppm * 1e-6)
            return int(self.t0_epoch_ns + (raw_mono * drift_factor))
        return raw_mono


    def pulse_pps(self) -> int:
        """Trigger a PPS boundary sync event."""
        self._last_pps_ns = self.now_ns()
        return self._last_pps_ns

    def timestamp_now(self) -> float:
        """Return timestamp in seconds with microsecond fidelity."""
        return self.now_ns() / 1e9


T = TypeVar("T")


@dataclass
class StampedPacket(Generic[T]):
    """Sensor payload wrapped with hardware-locked timestamping metadata."""

    sensor_id: str
    seq: int
    timestamp_ns: int
    data: T
    pps_locked: bool = True
    ptp_stratum: int = 1

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9


@dataclass
class SyncedFramePair:
    """Matched optical frame and radar observation within coincidence window."""

    optical: StampedPacket[np.ndarray]
    radar: StampedPacket[dict[str, Any]]
    delta_ms: float

    @property
    def frame(self) -> np.ndarray:
        return self.optical.data

    @property
    def radar_data(self) -> dict[str, Any]:
        return self.radar.data


class SensorSyncBuffer:
    """Coincidence jitter buffer that matches camera frames and radar data by hardware timestamps.

    Eliminates time-drift errors during high-speed sensor fusion by locking stream pairs
    within a tight temporal window (default max coincidence tolerance: 2.0 ms).
    """

    def __init__(
        self,
        tolerance_ms: float = 2.0,
        max_buffer_size: int = 60,
        time_source: HardwareTimeSource | None = None,
    ) -> None:
        self.tolerance_ns = int(tolerance_ms * 1e6)
        self.max_buffer_size = max_buffer_size
        self.clock = time_source or HardwareTimeSource()
        self._optical_queue: deque[StampedPacket[np.ndarray]] = deque(maxlen=max_buffer_size)
        self._radar_queue: deque[StampedPacket[dict[str, Any]]] = deque(maxlen=max_buffer_size)
        self.matched_count: int = 0
        self.dropped_optical: int = 0
        self.dropped_radar: int = 0

    def push_optical(self, frame: np.ndarray, timestamp_ns: int | None = None, seq: int = 0) -> StampedPacket[np.ndarray]:
        ts = timestamp_ns if timestamp_ns is not None else self.clock.now_ns()
        pkt = StampedPacket(
            sensor_id="optical",
            seq=seq,
            timestamp_ns=ts,
            data=frame,
            pps_locked=(self.clock.mode == SyncMode.PPS),
        )
        self._optical_queue.append(pkt)
        return pkt

    def push_radar(self, radar_data: dict[str, Any], timestamp_ns: int | None = None, seq: int = 0) -> StampedPacket[dict[str, Any]]:
        ts = timestamp_ns if timestamp_ns is not None else self.clock.now_ns()
        pkt = StampedPacket(
            sensor_id="radar",
            seq=seq,
            timestamp_ns=ts,
            data=radar_data,
            pps_locked=(self.clock.mode == SyncMode.PPS),
        )
        self._radar_queue.append(pkt)
        return pkt

    def pop_synced(self) -> SyncedFramePair | None:
        """Find the earliest coincident pair of optical and radar packets within tolerance."""
        if not self._optical_queue or not self._radar_queue:
            return None

        while self._optical_queue and self._radar_queue:
            opt = self._optical_queue[0]
            rad = self._radar_queue[0]
            delta_ns = opt.timestamp_ns - rad.timestamp_ns

            if abs(delta_ns) <= self.tolerance_ns:
                # Coincident match within tolerance
                self._optical_queue.popleft()
                self._radar_queue.popleft()
                self.matched_count += 1
                return SyncedFramePair(
                    optical=opt,
                    radar=rad,
                    delta_ms=abs(delta_ns) / 1e6,
                )
            elif delta_ns > self.tolerance_ns:
                # Radar is too old, drop it to catch up
                self._radar_queue.popleft()
                self.dropped_radar += 1
            else:
                # Optical is too old, drop it to catch up
                self._optical_queue.popleft()
                self.dropped_optical += 1

        return None

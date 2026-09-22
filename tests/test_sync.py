"""Tests for Hardware-Locked Timestamping and Stream Synchronization (PPS / IEEE 1588 PTP)."""
from __future__ import annotations

import time
import numpy as np
import pytest

from tacyolo.sensors.optical import OpticalSource
from tacyolo.sensors.radar import SyntheticRadar
from tacyolo.sensors.sync import (
    HardwareTimeSource,
    SensorSyncBuffer,
    SyncMode,
    StampedPacket,
    SyncedFramePair,
)


def test_hardware_time_source_pps_and_ptp() -> None:
    # 1. PPS Mode
    pps_clock = HardwareTimeSource(mode=SyncMode.PPS)
    t1_ns = pps_clock.now_ns()
    time.sleep(0.005)
    t2_ns = pps_clock.now_ns()
    assert t2_ns > t1_ns
    assert pps_clock.timestamp_now() > 0.0

    # 2. PTP Mode with drift compensation
    ptp_clock = HardwareTimeSource(mode=SyncMode.PTP, ptp_drift_ppm=10.0)
    ptp_t1 = ptp_clock.now_ns()
    assert ptp_t1 > 0


def test_sensor_sync_buffer_coincidence_matching() -> None:
    clock = HardwareTimeSource(mode=SyncMode.PPS)
    sync_buf = SensorSyncBuffer(tolerance_ms=2.0, time_source=clock)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_radar = {"rd_map": np.zeros((64, 32), dtype=np.float32)}

    t0_ns = clock.now_ns()

    # Push optical at t0, radar at t0 + 1.0 ms (within 2.0 ms tolerance)
    sync_buf.push_optical(dummy_frame, timestamp_ns=t0_ns, seq=1)
    sync_buf.push_radar(dummy_radar, timestamp_ns=t0_ns + int(1.0 * 1e6), seq=1)

    pair = sync_buf.pop_synced()
    assert pair is not None
    assert isinstance(pair, SyncedFramePair)
    assert pair.delta_ms <= 2.0
    assert pair.optical.seq == 1
    assert pair.radar.seq == 1
    assert sync_buf.matched_count == 1


def test_sensor_sync_buffer_drops_stale_data() -> None:
    clock = HardwareTimeSource()
    sync_buf = SensorSyncBuffer(tolerance_ms=2.0, time_source=clock)

    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_radar = {"rd_map": np.zeros((10, 10), dtype=np.float32)}

    t0_ns = clock.now_ns()

    # Push very old radar at t0 - 20ms
    sync_buf.push_radar(dummy_radar, timestamp_ns=t0_ns - int(20.0 * 1e6), seq=1)
    # Push optical at t0
    sync_buf.push_optical(dummy_frame, timestamp_ns=t0_ns, seq=2)
    # Push matching radar at t0 + 0.5ms
    sync_buf.push_radar(dummy_radar, timestamp_ns=t0_ns + int(0.5 * 1e6), seq=2)

    pair = sync_buf.pop_synced()
    assert pair is not None
    assert pair.optical.seq == 2
    assert pair.radar.seq == 2
    assert sync_buf.dropped_radar >= 1


def test_stamped_sources_integration() -> None:
    optical = OpticalSource(source="synthetic", width=160, height=120)
    radar = SyntheticRadar(n_range=32, n_doppler=16)

    opt_packet = optical.read_stamped()
    rad_packet = radar.next_stamped()

    assert opt_packet is not None
    assert isinstance(opt_packet, StampedPacket)
    assert opt_packet.data.shape == (120, 160, 3)
    assert opt_packet.timestamp_ns > 0

    assert rad_packet is not None
    assert isinstance(rad_packet, StampedPacket)
    assert "rd_map" in rad_packet.data
    assert rad_packet.timestamp_ns > 0

    optical.close()
    radar.close()

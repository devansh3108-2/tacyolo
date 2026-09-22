"""Tests for INT8 Hardened Quantization and Operational Calibration Pipeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from tacyolo.runtime.quantize import (
    OperationalCalibrationDataLoader,
    build_int8_calibration_cache,
)


def test_operational_calibration_data_loader(tmp_path: Path) -> None:
    loader = OperationalCalibrationDataLoader(
        source_dir=tmp_path / "operational_empty",
        imgsz=320,
        max_samples=10,
    )
    assert len(loader) > 0
    batches = list(loader.get_batch(batch_size=4))
    assert len(batches) >= 1
    b0 = batches[0]
    assert b0.ndim == 4
    assert b0.shape[1:] == (3, 320, 320)
    assert b0.max() <= 1.0
    assert b0.min() >= 0.0


def test_build_int8_calibration_cache(tmp_path: Path) -> None:
    weights_path = tmp_path / "tactical_test.pt"
    weights_path.write_bytes(b"dummy_weights")
    out_dir = tmp_path / "out"

    cache_file = build_int8_calibration_cache(
        weights_path=weights_path,
        calib_dir=tmp_path / "calib",
        output_dir=out_dir,
        imgsz=320,
        max_samples=8,
    )
    assert cache_file.exists()
    assert cache_file.stat().st_size > 0
    meta_file = out_dir / "tactical_test_int8_meta.json"
    assert meta_file.exists()

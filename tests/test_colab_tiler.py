"""Tests for Google Drive Layout, Image Tiler, and Batch Purge Pipeline."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from scripts.colab_tile_train import (
    DriveLayout,
    ImageTiler,
    ColabBatchStreamingTrainer,
)


def test_drive_layout_setup(tmp_path: Path) -> None:
    drive_root = tmp_path / "MyDrive" / "TACYOLO"
    layout = DriveLayout(drive_root)
    layout.setup()

    assert layout.working_testing.exists()
    assert layout.raw_data.exists()
    assert layout.tiled_batches.exists()
    assert layout.trained_weights.exists()
    assert layout.calibration_data.exists()
    assert layout.metrics_logs.exists()

    state = layout.load_state()
    assert "completed_batches" in state
    state["completed_batches"].append("batch_0")
    layout.save_state(state)
    assert "batch_0" in layout.load_state()["completed_batches"]


def test_image_tiler_slicing_and_box_reprojection(tmp_path: Path) -> None:
    tiler = ImageTiler(tile_size=400, overlap=0.2)

    # Create a 700x700 image
    img = np.full((700, 700, 3), 120, dtype=np.uint8)
    img_path = tmp_path / "aerial_test.jpg"
    cv2.imwrite(str(img_path), img)

    # Add a bounding box around (x=100..150, y=100..150) -> normalized xc=0.178, yc=0.178, w=0.071, h=0.071
    lbl_path = tmp_path / "aerial_test.txt"
    lbl_path.write_text("7 0.178571 0.178571 0.071429 0.071429\n", encoding="utf-8")

    out_img_dir = tmp_path / "tiled_img"
    out_lbl_dir = tmp_path / "tiled_lbl"
    out_img_dir.mkdir()
    out_lbl_dir.mkdir()

    n_tiles = tiler.tile_image_and_labels(img_path, lbl_path, out_img_dir, out_lbl_dir)
    assert n_tiles > 1
    tiled_imgs = list(out_img_dir.glob("*.jpg"))
    assert len(tiled_imgs) == n_tiles

    # Check that at least one tile received the re-projected bounding box
    tiled_lbls = [p for p in out_lbl_dir.glob("*.txt") if p.stat().st_size > 0]
    assert len(tiled_lbls) >= 1
    content = tiled_lbls[0].read_text(encoding="utf-8").strip()
    parts = content.split()
    assert parts[0] == "7"


def test_purge_batch_data(tmp_path: Path) -> None:
    layout = DriveLayout(tmp_path / "TACYOLO")
    trainer = ColabBatchStreamingTrainer(layout=layout, tile_size=320)

    raw_dir = layout.raw_data / "test_tag"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sample.txt").write_text("test", encoding="utf-8")

    tiled_dir = layout.tiled_batches / "test_tag"
    tiled_dir.mkdir(parents=True)
    (tiled_dir / "tile.txt").write_text("test", encoding="utf-8")

    assert raw_dir.exists()
    assert tiled_dir.exists()

    trainer.purge_batch_data(raw_dir, "test_tag")

    assert not raw_dir.exists()
    assert not tiled_dir.exists()


def test_get_dir_size_gb(tmp_path: Path) -> None:
    from scripts.colab_tile_train import get_dir_size_gb

    d = tmp_path / "sized_dir"
    d.mkdir()
    # 1MB file
    f = d / "data.bin"
    f.write_bytes(b"\x00" * 1024 * 1024)

    gb = get_dir_size_gb(d)
    assert 0.0 < gb < 0.01


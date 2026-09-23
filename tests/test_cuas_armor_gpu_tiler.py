"""Tests for GPU Tiler & Unified C-UAS and Military Armor Dataset Pipeline."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import yaml

from scripts.prepare_cuas_armor_gpu import (
    CUASArmorGPUTiler,
    DATASET_CONFIGS,
    UNIFIED_CLASSES,
    initialize_unified_folders,
    UNIFIED_DATASET_DIR,
)


def test_master_data_yaml_structure() -> None:
    initialize_unified_folders()
    yaml_file = UNIFIED_DATASET_DIR / "data.yaml"
    assert yaml_file.exists()

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["nc"] == 12
    assert len(data["names"]) == 12
    assert data["names"][0] == "person"
    assert data["names"][1] == "bird"
    assert data["names"][2] == "drone"
    assert data["names"][6] == "tank"
    assert data["names"][7] == "armored_vehicle"
    assert data["names"][9] == "artillery"


def test_annotation_parsers_for_all_formats(tmp_path: Path) -> None:
    tiler = CUASArmorGPUTiler(tile_size=640, device="cpu")

    # 1. Test YOLO format (Seraphim: 0 -> 2, 1 -> 3)
    lbl_yolo = tmp_path / "seraphim.txt"
    lbl_yolo.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n", encoding="utf-8")
    boxes = tiler.parse_annotations(lbl_yolo, 1000, 1000, dataset_key="seraphim")
    assert len(boxes) == 2
    assert boxes[0][0] == 2  # mapped to drone
    assert boxes[1][0] == 3  # mapped to fixed_wing_uav

    # 2. Test VisDrone format (comma-separated, category 1: pedestrian -> 0)
    lbl_vis = tmp_path / "visdrone.txt"
    lbl_vis.write_text("100,200,50,80,1,1,0,0\n300,400,60,60,1,4,0,0\n", encoding="utf-8")
    boxes = tiler.parse_annotations(lbl_vis, 1000, 1000, dataset_key="visdrone")
    assert len(boxes) == 2
    assert boxes[0][0] == 0   # pedestrian -> person
    assert boxes[1][0] == 10  # car -> civilian_vehicle

    # 3. Test DOTA 8-point polygon format
    lbl_dota = tmp_path / "dota.txt"
    lbl_dota.write_text("100 100 200 100 200 200 100 200 large-vehicle 0\n", encoding="utf-8")
    boxes = tiler.parse_annotations(lbl_dota, 1000, 1000, dataset_key="dota_v2")
    assert len(boxes) == 1
    assert boxes[0][0] == 8  # large-vehicle -> military_truck / heavy

    # 4. Test Drone-vs-Bird format (0: bird -> 1, 1: drone -> 2)
    lbl_dvb = tmp_path / "drone_bird.txt"
    lbl_dvb.write_text("0 0.4 0.4 0.05 0.05\n1 0.8 0.8 0.08 0.08\n", encoding="utf-8")
    boxes = tiler.parse_annotations(lbl_dvb, 1000, 1000, dataset_key="drone_vs_bird")
    assert len(boxes) == 2
    assert boxes[0][0] == 1  # bird
    assert boxes[1][0] == 2  # drone


def test_gpu_cuda_tiler_slicing_and_reprojection(tmp_path: Path) -> None:
    # Use CUDA if available, fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiler = CUASArmorGPUTiler(tile_size=640, overlap=0.2, device=device)

    # Create a 1920x1080 high-res image
    img = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    # Draw a distinct rectangle
    cv2.rectangle(img, (200, 200), (300, 300), (255, 0, 0), -1)

    img_path = tmp_path / "test_aerial_4k.jpg"
    cv2.imwrite(str(img_path), img)

    # Box centered at (250, 250), width 100, height 100 -> norm xc=250/1920, yc=250/1080, w=100/1920, h=100/1080
    xc = 250.0 / 1920.0
    yc = 250.0 / 1080.0
    bw = 100.0 / 1920.0
    bh = 100.0 / 1080.0
    lbl_path = tmp_path / "test_aerial_4k.txt"
    lbl_path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    out_img_dir = tmp_path / "out_imgs"
    out_lbl_dir = tmp_path / "out_lbls"
    out_img_dir.mkdir()
    out_lbl_dir.mkdir()

    n_tiles = tiler.tile_image(img_path, lbl_path, out_img_dir, out_lbl_dir, dataset_key="seraphim")
    assert n_tiles > 1

    tiled_imgs = list(out_img_dir.glob("*.jpg"))
    assert len(tiled_imgs) == n_tiles

    # Verify tile dimensions are 640x640
    sample_tile = cv2.imread(str(tiled_imgs[0]))
    assert sample_tile.shape == (640, 640, 3)

    # Verify that at least one tile received the re-projected bounding box
    label_files = [p for p in out_lbl_dir.glob("*.txt") if p.stat().st_size > 0]
    assert len(label_files) >= 1

    for lf in label_files:
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            c_id = int(parts[0])
            # Seraphim class 0 was mapped to class 2 (drone)
            assert c_id == 2
            txc, tyc, tbw, tbh = [float(x) for x in parts[1:]]
            assert 0.0 <= txc <= 1.0
            assert 0.0 <= tyc <= 1.0
            assert 0.0 < tbw <= 1.0
            assert 0.0 < tbh <= 1.0

"""Comprehensive Tests for Multi-Teacher Ensemble Distillation Pipeline."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import yaml

from tacyolo.distillation.ensemble import (
    FrameDetections,
    MultiTeacherInference,
    TeacherDetection,
    TeacherModelSpec,
)
from tacyolo.distillation.exporter import EdgeModelExporter, LatencyBenchmarkResult
from tacyolo.distillation.fusion import FusedDetection, WeightedBoxFusionConsensus
from tacyolo.distillation.partitioner import DatasetPartitioner


def test_teacher_class_mapping() -> None:
    # Test class mapping with integer and string keys
    mapping = {0: 10, "1": 20, 2: None}
    spec = TeacherModelSpec(
        name="test_teacher",
        model_path="dummy.pt",
        class_map=mapping,
    )
    assert spec.map_class_id(0) == 10
    assert spec.map_class_id(1) == 20
    assert spec.map_class_id(2) is None
    assert spec.map_class_id(99) is None

    # Test identity mapping when class_map is None
    spec_identity = TeacherModelSpec(name="identity", model_path="dummy.pt", class_map=None)
    assert spec_identity.map_class_id(5) == 5


def test_wbf_fusion_consensus() -> None:
    fusion = WeightedBoxFusionConsensus(
        weights=[1.0, 1.2, 1.1],
        iou_thr=0.55,
        skip_box_thr=0.35,
    )

    # Simulate 3 teachers detecting a target around [0.1, 0.2, 0.4, 0.5]
    t0_dets = [
        TeacherDetection(box_xyxy_norm=[0.10, 0.20, 0.40, 0.50], confidence=0.85, class_id=2),
        TeacherDetection(box_xyxy_norm=[0.80, 0.80, 0.90, 0.90], confidence=0.20, class_id=2), # Low conf, should be skipped
    ]
    t1_dets = [
        TeacherDetection(box_xyxy_norm=[0.12, 0.21, 0.39, 0.51], confidence=0.92, class_id=2),
    ]
    t2_dets = [
        TeacherDetection(box_xyxy_norm=[0.11, 0.19, 0.41, 0.49], confidence=0.88, class_id=2),
    ]

    frame = FrameDetections(
        frame_id="frame_001",
        image_path=Path("dummy.jpg"),
        width=1920,
        height=1080,
        teacher_detections=[t0_dets, t1_dets, t2_dets],
    )

    fused = fusion.fuse_frame(frame)
    # The low confidence box (0.20 < 0.35) must be skipped, leaving 1 fused box
    assert len(fused) == 1
    det = fused[0]
    assert det.class_id == 2
    assert 0.85 <= det.confidence <= 1.0

    # Verify YOLO line serialization
    yolo_str = det.to_yolo_line()
    parts = yolo_str.split()
    assert len(parts) == 5
    assert parts[0] == "2"
    for v in parts[1:]:
        val = float(v)
        assert 0.0 <= val <= 1.0


def test_dataset_partitioner_and_yaml_generation(tmp_path: Path) -> None:
    output_dir = tmp_path / "partitioned_dataset"
    partitioner = DatasetPartitioner(
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.20,
        test_ratio=0.10,
        class_names=["person", "drone", "tank"],
    )

    # Create 10 dummy images and detections
    frame_dets: dict[str, FrameDetections] = {}
    fused_dataset: dict[str, list[FusedDetection]] = {}

    src_imgs_dir = tmp_path / "raw_src"
    src_imgs_dir.mkdir()

    for i in range(10):
        img_p = src_imgs_dir / f"test_{i:03d}.jpg"
        cv2.imwrite(str(img_p), np.full((100, 100, 3), 128, dtype=np.uint8))
        fid = str(img_p.resolve())
        frame_dets[fid] = FrameDetections(
            frame_id=fid,
            image_path=img_p,
            width=100,
            height=100,
            teacher_detections=[[]],
        )
        fused_dataset[fid] = [
            FusedDetection(
                class_id=1,
                confidence=0.90,
                xc=0.5,
                yc=0.5,
                bw=0.2,
                bh=0.2,
                x1=0.4,
                y1=0.4,
                x2=0.6,
                y2=0.6,
            )
        ]

    summary = partitioner.partition(frame_dets, fused_dataset)
    assert summary.total_images == 10
    assert summary.train_count == 7
    assert summary.val_count == 2
    assert summary.test_count == 1
    assert summary.dataset_yaml_path.exists()

    with open(summary.dataset_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["nc"] == 3
    assert cfg["names"][0] == "person"
    assert cfg["names"][1] == "drone"
    assert cfg["names"][2] == "tank"

    # Verify labels were written
    train_labels = list((output_dir / "labels" / "train").glob("*.txt"))
    assert len(train_labels) == 7
    content = train_labels[0].read_text(encoding="utf-8").strip()
    assert content.startswith("1 0.500000 0.500000")


def test_media_collector_image_and_video(tmp_path: Path) -> None:
    # 1. Test image collection
    img_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))

    # 2. Test video creation & extraction
    vid_path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
    for _ in range(15):
        frame = np.full((64, 64, 3), 200, dtype=np.uint8)
        out.write(frame)
    out.release()

    collected = MultiTeacherInference.collect_operational_media(
        input_dir=tmp_path,
        video_stride=5,
    )

    # Should contain the original sample.jpg + 3 extracted frames (0, 5, 10)
    assert len(collected) >= 4
    for p in collected:
        assert p.exists()


def test_edge_latency_benchmarking(tmp_path: Path) -> None:
    device = "0" if torch.cuda.is_available() else "cpu"
    exporter = EdgeModelExporter(device=device)

    # Run benchmark on existing YOLO model weights
    weights_path = Path("weights/yolo11s.pt")
    if not weights_path.exists():
        # Use pretrained yolo11n.pt or dummy test
        from ultralytics import YOLO
        m = YOLO("yolo11n.pt")
        weights_path = Path("yolo11n.pt")

    res = exporter.benchmark_latency(
        model_path=weights_path,
        imgsz=320,
        warmup_iters=2,
        bench_iters=5,
    )

    assert isinstance(res, LatencyBenchmarkResult)
    assert res.mean_latency_ms > 0.0
    assert res.p99_latency_ms >= res.p95_latency_ms
    assert res.fps > 0.0
    table = res.summary_table()
    assert "Mean Latency" in table
    assert "P99 Latency" in table

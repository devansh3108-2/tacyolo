"""Production CLI Runner: Multi-Teacher Ensemble Distillation Pipeline.

Integrates YOLOv8, YOLO11, and RT-DETR via Weighted Box Fusion (WBF),
distills consensus labels into YOLO11s, and exports to TensorRT FP16
with P99 latency benchmarking for embedded Jetson deployments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tacyolo.distillation import (
    EdgeModelExporter,
    EnsembleDistillationPipeline,
    PipelineConfig,
    TeacherModelSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production Multi-Teacher (YOLOv8 + YOLO11 + RT-DETR) -> YOLO11s Distillation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Media inputs
    parser.add_argument("--input-dir", type=str, required=True, help="Path to unannotated operational videos or images")
    parser.add_argument("--output-dir", type=str, default="runs/distill_dataset", help="Output directory for partitioned dataset")
    parser.add_argument("--video-stride", type=int, default=5, help="Frame extraction stride for input video files")

    # Teacher models
    parser.add_argument("--yolov8-weights", type=str, default="yolov8x.pt", help="Weights path for Teacher A (YOLOv8)")
    parser.add_argument("--yolo11-weights", type=str, default="yolo11x.pt", help="Weights path for Teacher B (YOLO11)")
    parser.add_argument("--rtdetr-weights", type=str, default="rtdetr-l.pt", help="Weights path for Teacher C (RT-DETR)")
    parser.add_argument("--weights", nargs=3, type=float, default=[1.0, 1.2, 1.1], help="WBF weights for [v8, 11, rtdetr]")

    # WBF parameters
    parser.add_argument("--iou-thr", type=float, default=0.55, help="Weighted Box Fusion IoU consensus threshold")
    parser.add_argument("--skip-box-thr", type=float, default=0.35, help="WBF confidence filter threshold")
    parser.add_argument("--taxonomy-map", type=str, default=None, help="YAML file with teacher->target class mapping")

    # Partitioning
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train split fraction")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split fraction")

    # Student training
    parser.add_argument("--student-model", type=str, default="yolo11s.pt", help="Student detector architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Student training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference and training resolution")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Epochs before training end to disable mosaic")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")

    # Hardware & export
    parser.add_argument("--device", type=str, default=None, help="CUDA device index ('0', 'cpu', etc.)")
    parser.add_argument("--skip-export", action="store_true", help="Skip TensorRT export stage")
    parser.add_argument("--benchmark-iters", type=int, default=100, help="Warm benchmark iterations for P99 latency")
    parser.add_argument("--stage", choices=["all", "ensemble", "partition", "train", "export"], default="all", help="Execution stage")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig.default_tri_teacher(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        yolov8_weights=args.yolov8v8_weights if hasattr(args, "yolov8v8_weights") else args.yolov8_weights,
        yolo11_weights=args.yolo11_weights,
        rtdetr_weights=args.rtdetr_weights,
        class_map_yaml=args.taxonomy_map,
    )
    config.wbf_weights = list(args.weights)
    config.iou_thr = args.iou_thr
    config.skip_box_thr = args.skip_box_thr
    config.train_ratio = args.train_ratio
    config.val_ratio = args.val_ratio
    config.test_ratio = args.test_ratio
    config.student_model = args.student_model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.imgsz = args.imgsz
    config.close_mosaic = args.close_mosaic
    config.amp = not args.no_amp
    config.device = args.device
    config.export_tensorrt = not args.skip_export
    config.benchmark_iters = args.benchmark_iters

    pipeline = EnsembleDistillationPipeline(config)

    if args.stage == "all":
        results = pipeline.run_full_pipeline()
        print("\n✨ PIPELINE COMPLETED SUCCESSFULLY! ✨")
        return 0

    if args.stage == "ensemble":
        frame_dets = pipeline.run_stage_1_ensemble()
        fused = pipeline.run_stage_2_wbf(frame_dets)
        summary = pipeline.run_stage_3_partition(frame_dets, fused)
        print(f"✅ Ensemble & dataset generation finished: {summary.dataset_yaml_path}")
        return 0

    if args.stage == "train":
        yaml_path = Path(args.output_dir) / "dataset.yaml"
        train_res = pipeline.run_stage_4_train(yaml_path)
        print(f"✅ Training completed: {train_res.best_weights}")
        return 0

    if args.stage == "export":
        candidate_weights = Path(args.output_dir) / "weights" / "best.pt"
        if not candidate_weights.exists():
            candidate_weights = Path(args.student_model)
        exported, bench = pipeline.run_stage_5_export(candidate_weights)
        print(f"✅ Exported: {exported}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

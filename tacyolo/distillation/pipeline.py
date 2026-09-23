"""End-to-End Orchestrator: Multi-Teacher Ensemble Distillation Pipeline.

Chains Stage 1 (Multi-Teacher Inference) -> Stage 2 (WBF Consensus) ->
Stage 3 (Auto Partitioning) -> Stage 4 (Student Training) -> Stage 5 (Edge Export).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from tacyolo.distillation.ensemble import (
    FrameDetections,
    MultiTeacherInference,
    TeacherModelSpec,
)
from tacyolo.distillation.exporter import EdgeModelExporter, LatencyBenchmarkResult
from tacyolo.distillation.fusion import FusedDetection, WeightedBoxFusionConsensus
from tacyolo.distillation.partitioner import DatasetPartitioner, PartitionSummary
from tacyolo.distillation.trainer import StudentDistillationTrainer, TrainingResult


@dataclass
class PipelineConfig:
    """End-to-end configuration for ensemble distillation."""
    input_dir: str | Path
    output_dir: str | Path = "runs/distillation_dataset"
    teachers: list[TeacherModelSpec] = field(default_factory=list)
    wbf_weights: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.1])
    iou_thr: float = 0.55
    skip_box_thr: float = 0.35
    train_ratio: float = 0.80
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    student_model: str = "yolo11s.pt"
    epochs: int = 50
    batch_size: int = 16
    imgsz: int = 640
    close_mosaic: int = 10
    amp: bool = True
    device: str | None = None
    class_names: list[str] | dict[int, str] | None = None
    export_tensorrt: bool = True
    benchmark_iters: int = 100

    @classmethod
    def default_tri_teacher(
        cls,
        input_dir: str | Path,
        output_dir: str | Path = "runs/distillation_dataset",
        yolov8_weights: str = "yolov8x.pt",
        yolo11_weights: str = "yolo11x.pt",
        rtdetr_weights: str = "rtdetr-l.pt",
        class_map_yaml: str | Path | None = None,
    ) -> PipelineConfig:
        """Constructs default ensemble configuration featuring YOLOv8, YOLO11, and RT-DETR."""
        class_mapping: dict[int, int] | None = None
        target_names: dict[int, str] | None = None

        if class_map_yaml and Path(class_map_yaml).exists():
            with open(class_map_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                class_mapping = data.get("mapping")
                target_names = data.get("names")

        teachers = [
            TeacherModelSpec(
                name="Teacher_A_YOLOv8",
                model_path=yolov8_weights,
                model_type="yolo",
                weight=1.0,
                class_map=class_mapping,
            ),
            TeacherModelSpec(
                name="Teacher_B_YOLO11",
                model_path=yolo11_weights,
                model_type="yolo",
                weight=1.2,
                class_map=class_mapping,
            ),
            TeacherModelSpec(
                name="Teacher_C_RTDETR",
                model_path=rtdetr_weights,
                model_type="rtdetr",
                weight=1.1,
                class_map=class_mapping,
            ),
        ]

        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            teachers=teachers,
            wbf_weights=[1.0, 1.2, 1.1],
            class_names=target_names or {0: "target"},
        )


class EnsembleDistillationPipeline:
    """Production pipeline orchestrating multi-teacher distillation to edge deployment."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.inference_engine = MultiTeacherInference(
            teachers=config.teachers,
            device=config.device,
            batch_size=config.batch_size,
        )
        self.fusion_engine = WeightedBoxFusionConsensus(
            weights=config.wbf_weights,
            iou_thr=config.iou_thr,
            skip_box_thr=config.skip_box_thr,
        )
        self.partitioner = DatasetPartitioner(
            output_dir=self.output_dir,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            class_names=config.class_names,
        )
        self.trainer = StudentDistillationTrainer(
            student_model=config.student_model,
            device=config.device,
            imgsz=config.imgsz,
            close_mosaic=config.close_mosaic,
            amp=config.amp,
        )
        self.exporter = EdgeModelExporter(device=config.device)

    def run_stage_1_ensemble(self) -> dict[str, FrameDetections]:
        """Stage 1: Media ingestion and multi-teacher batched inference."""
        print("\n=======================================================")
        print("🚀 STAGE 1: MULTI-TEACHER ENSEMBLE INFERENCE")
        print("=======================================================")
        images = self.inference_engine.collect_operational_media(self.config.input_dir)
        print(f"📸 Total Operational Frames to Process: {len(images)}")
        return self.inference_engine.run_teacher_inference(images)

    def run_stage_2_wbf(
        self,
        frame_detections: dict[str, FrameDetections],
    ) -> dict[str, list[FusedDetection]]:
        """Stage 2: Weighted Box Fusion consensus aggregation."""
        print("\n=======================================================")
        print("⚖️ STAGE 2: WEIGHTED BOX FUSION (WBF) CONSENSUS")
        print("=======================================================")
        fused = self.fusion_engine.fuse_all_frames(frame_detections)
        total_boxes = sum(len(v) for v in fused.values())
        print(f"🎯 Total Fused Consensus Bounding Boxes: {total_boxes}")
        return fused

    def run_stage_3_partition(
        self,
        frame_detections: dict[str, FrameDetections],
        fused_dataset: dict[str, list[FusedDetection]],
    ) -> PartitionSummary:
        """Stage 3: Partitioning into images/{train,val,test} and dataset.yaml."""
        print("\n=======================================================")
        print("📁 STAGE 3: AUTO-DATASET PARTITIONING")
        print("=======================================================")
        return self.partitioner.partition(frame_detections, fused_dataset)

    def run_stage_4_train(self, dataset_yaml: Path) -> TrainingResult:
        """Stage 4: Student distillation training (YOLO11s)."""
        print("\n=======================================================")
        print("🎓 STAGE 4: SINGLE STUDENT DISTILLATION TRAINING")
        print("=======================================================")
        return self.trainer.train(
            dataset_yaml=dataset_yaml,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
        )

    def run_stage_5_export(
        self,
        weights_path: Path,
    ) -> tuple[Path, LatencyBenchmarkResult | None]:
        """Stage 5: Edge export (TensorRT FP16) and latency benchmarking."""
        print("\n=======================================================")
        print("⚡ STAGE 5: ZERO-OVERHEAD EDGE EXPORT & BENCHMARKING")
        print("=======================================================")
        exported_path = self.exporter.export_tensorrt(
            weights_path=weights_path,
            imgsz=self.config.imgsz,
            batch_size=1,
            half=True,
        )
        benchmark_res = None
        if self.config.benchmark_iters > 0:
            benchmark_target = exported_path if exported_path.suffix in {".engine", ".pt"} else weights_path
            benchmark_res = self.exporter.benchmark_latency(
                model_path=benchmark_target,
                imgsz=self.config.imgsz,
                batch_size=1,
                warmup_iters=20,
                bench_iters=self.config.benchmark_iters,
            )
        return exported_path, benchmark_res

    def run_full_pipeline(self) -> dict[str, Any]:
        """Executes all 5 stages end-to-end."""
        # Stage 1
        frame_detections = self.run_stage_1_ensemble()
        # Stage 2
        fused_dataset = self.run_stage_2_wbf(frame_detections)
        # Stage 3
        partition_summary = self.run_stage_3_partition(frame_detections, fused_dataset)
        # Stage 4
        train_result = self.run_stage_4_train(partition_summary.dataset_yaml_path)
        # Stage 5
        best_weights = train_result.best_weights or train_result.last_weights
        if not best_weights or not best_weights.exists():
            raise RuntimeError("Student training did not produce valid weights.")

        exported_model, benchmark_result = self.run_stage_5_export(best_weights)

        return {
            "partition_summary": partition_summary,
            "train_result": train_result,
            "exported_model": exported_model,
            "benchmark_result": benchmark_result,
        }

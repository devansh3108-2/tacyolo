"""TACYOLO Multi-Teacher Ensemble Distillation & Edge Jetson Pipeline.

Unifies YOLOv8, YOLO11, and RT-DETR teacher models using Weighted Box Fusion (WBF),
distills consensus labels into a compact YOLO11s student detector, and exports
to TensorRT FP16 for embedded NVIDIA Jetson deployments.
"""
from __future__ import annotations

from tacyolo.distillation.ensemble import MultiTeacherInference, TeacherModelSpec
from tacyolo.distillation.exporter import EdgeModelExporter, LatencyBenchmarkResult
from tacyolo.distillation.fusion import WeightedBoxFusionConsensus
from tacyolo.distillation.partitioner import DatasetPartitioner
from tacyolo.distillation.pipeline import EnsembleDistillationPipeline, PipelineConfig
from tacyolo.distillation.trainer import StudentDistillationTrainer

__all__ = [
    "MultiTeacherInference",
    "TeacherModelSpec",
    "WeightedBoxFusionConsensus",
    "DatasetPartitioner",
    "StudentDistillationTrainer",
    "EdgeModelExporter",
    "LatencyBenchmarkResult",
    "EnsembleDistillationPipeline",
    "PipelineConfig",
]

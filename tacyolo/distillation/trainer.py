"""Stage 4: Single Student Distillation Training.

Initializes a compact student detector (YOLO11s) and executes training with
hyperparameters tuned for edge transfer (close_mosaic=10, amp=True, imgsz=640).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


@dataclass
class TrainingResult:
    """Artifacts and metrics produced by student distillation training."""
    best_weights: Path | None
    last_weights: Path | None
    save_dir: Path
    metrics: dict[str, float]


class StudentDistillationTrainer:
    """Trains a compact student model (YOLO11s) on WBF-distilled consensus labels."""

    def __init__(
        self,
        student_model: str = "yolo11s.pt",
        device: str | None = None,
        imgsz: int = 640,
        close_mosaic: int = 10,
        amp: bool = True,
    ) -> None:
        self.student_model = student_model
        self.imgsz = imgsz
        self.close_mosaic = close_mosaic
        self.amp = amp

        if device is not None:
            self.device = str(device)
        else:
            self.device = "0" if torch.cuda.is_available() else "cpu"

    def train(
        self,
        dataset_yaml: str | Path,
        epochs: int = 50,
        batch_size: int = 16,
        project: str | Path = "runs/distill",
        name: str = "yolo11s_student",
        workers: int = 4,
        patience: int = 20,
        extra_args: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Executes student distillation training."""
        yaml_path = Path(dataset_yaml).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml not found at: {yaml_path}")

        print(f"🎓 Initializing Student Detector: {self.student_model} on device: {self.device}...")
        model = YOLO(self.student_model)

        train_kwargs: dict[str, Any] = {
            "data": str(yaml_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": self.imgsz,
            "device": self.device,
            "project": str(project),
            "name": name,
            "exist_ok": True,
            "close_mosaic": self.close_mosaic,
            "amp": self.amp,
            "workers": workers,
            "patience": patience,
            "verbose": True,
        }

        if extra_args:
            train_kwargs.update(extra_args)

        print(f"🚀 Launching student training: {epochs} epochs, batch={batch_size}, close_mosaic={self.close_mosaic}, amp={self.amp}")
        results = model.train(**train_kwargs)

        save_dir = Path(model.trainer.save_dir if hasattr(model, "trainer") and model.trainer else Path(project) / name)
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        metrics_dict: dict[str, float] = {}
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            metrics_dict = {str(k): float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}

        # Write metrics summary
        summary_file = save_dir / "training_summary.json"
        summary_payload = {
            "student_model": self.student_model,
            "dataset_yaml": str(yaml_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": self.imgsz,
            "close_mosaic": self.close_mosaic,
            "amp": self.amp,
            "best_weights": str(best_pt) if best_pt.exists() else None,
            "last_weights": str(last_pt) if last_pt.exists() else None,
            "metrics": metrics_dict,
        }
        summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        print(f"✅ Distillation Training Complete! Best Weights: {best_pt if best_pt.exists() else last_pt}")
        return TrainingResult(
            best_weights=best_pt if best_pt.exists() else None,
            last_weights=last_pt if last_pt.exists() else None,
            save_dir=save_dir,
            metrics=metrics_dict,
        )

"""Stage 1: Multi-Teacher Ensemble Inference.

Loads YOLOv8, YOLO11, and RT-DETR teacher models, extracts frames from operational
video streams or image directories, normalizes taxonomy class spaces, and extracts
batched bounding box candidates with GPU memory isolation.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO


@dataclass
class TeacherModelSpec:
    """Specification for a teacher detector model in the ensemble."""
    name: str
    model_path: str | Path
    model_type: str = "yolo"  # "yolo" or "rtdetr"
    weight: float = 1.0
    conf: float = 0.25
    iou: float = 0.70
    imgsz: int = 640
    class_map: dict[int | str, int] | None = None  # maps source class_id -> target class_id

    def load_model(self, device: str = "0") -> Any:
        """Instantiates the detector on target device."""
        path_str = str(self.model_path)
        if self.model_type.lower() == "rtdetr":
            model = RTDETR(path_str)
        else:
            model = YOLO(path_str)
        return model

    def map_class_id(self, raw_class_id: int) -> int | None:
        """Maps teacher raw class index to unified target class index."""
        if self.class_map is None:
            return raw_class_id
        # Check int key
        if raw_class_id in self.class_map:
            return self.class_map[raw_class_id]
        # Check str key
        str_key = str(raw_class_id)
        if str_key in self.class_map:
            return self.class_map[str_key]
        return None


@dataclass
class TeacherDetection:
    """Individual candidate detection from a teacher model."""
    box_xyxy_norm: list[float]  # [x1, y1, x2, y2] normalized to [0, 1]
    confidence: float
    class_id: int


@dataclass
class FrameDetections:
    """All detections from all teachers for a specific frame."""
    frame_id: str
    image_path: Path
    width: int
    height: int
    # List of detections grouped by teacher index: [teacher_0_dets, teacher_1_dets, ...]
    teacher_detections: list[list[TeacherDetection]] = field(default_factory=list)


class MultiTeacherInference:
    """Orchestrates multi-teacher inference across image folders or video files."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}

    def __init__(
        self,
        teachers: Sequence[TeacherModelSpec],
        device: str | None = None,
        batch_size: int = 16,
        memory_efficient: bool = True,
    ) -> None:
        self.teachers = list(teachers)
        self.batch_size = max(1, batch_size)
        self.memory_efficient = memory_efficient

        if device is not None:
            self.device = str(device)
        else:
            self.device = "0" if torch.cuda.is_available() else "cpu"

    @classmethod
    def collect_operational_media(
        cls,
        input_dir: str | Path,
        staging_dir: str | Path | None = None,
        video_stride: int = 5,
    ) -> list[Path]:
        """Discovers operational image files and extracts keyframes from video files."""
        in_path = Path(input_dir).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {in_path}")

        collected_images: list[Path] = []
        raw_files = [p for p in in_path.rglob("*") if p.is_file()]

        # Collect native images
        for f in raw_files:
            if f.suffix.lower() in cls.IMAGE_EXTENSIONS:
                collected_images.append(f)

        # Process videos
        videos = [f for f in raw_files if f.suffix.lower() in cls.VIDEO_EXTENSIONS]
        if videos:
            target_stage = Path(staging_dir).resolve() if staging_dir else in_path / "extracted_frames"
            target_stage.mkdir(parents=True, exist_ok=True)

            for vid in videos:
                cap = cv2.VideoCapture(str(vid))
                if not cap.isOpened():
                    continue

                frame_idx = 0
                saved_idx = 0
                vid_stem = vid.stem

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % video_stride == 0:
                        out_name = f"{vid_stem}_f{frame_idx:06d}.jpg"
                        out_path = target_stage / out_name
                        cv2.imwrite(str(out_path), frame)
                        collected_images.append(out_path)
                        saved_idx += 1
                    frame_idx += 1

                cap.release()

        return sorted(list(dict.fromkeys(collected_images)))

    def run_teacher_inference(
        self,
        image_paths: Sequence[Path],
    ) -> dict[str, FrameDetections]:
        """Runs all teacher models over the images with VRAM memory isolation."""
        if not image_paths:
            return {}

        results: dict[str, FrameDetections] = {}
        for img_path in image_paths:
            h, w = self._probe_image_size(img_path)
            frame_id = str(img_path.resolve())
            results[frame_id] = FrameDetections(
                frame_id=frame_id,
                image_path=img_path,
                width=w,
                height=h,
                teacher_detections=[[] for _ in range(len(self.teachers))],
            )

        # Run teachers sequentially to guarantee zero OOM risk on single-GPU
        for teacher_idx, teacher in enumerate(self.teachers):
            print(f"🔬 Running Teacher [{teacher_idx+1}/{len(self.teachers)}]: {teacher.name} ({teacher.model_path}) on {self.device}...")
            model = teacher.load_model(device=self.device)

            # Batched processing
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]
                path_strs = [str(p) for p in batch_paths]

                preds = model(
                    path_strs,
                    conf=teacher.conf,
                    iou=teacher.iou,
                    imgsz=teacher.imgsz,
                    device=self.device,
                    verbose=False,
                )

                for img_p, pred in zip(batch_paths, preds):
                    frame_id = str(img_p.resolve())
                    w, h = results[frame_id].width, results[frame_id].height

                    if pred.boxes is None or len(pred.boxes) == 0:
                        continue

                    boxes_xyxy = pred.boxes.xyxy.cpu().numpy()
                    confs = pred.boxes.conf.cpu().numpy()
                    classes = pred.boxes.cls.cpu().numpy().astype(int)

                    t_dets: list[TeacherDetection] = []
                    for (x1, y1, x2, y2), conf, raw_c in zip(boxes_xyxy, confs, classes):
                        target_c = teacher.map_class_id(int(raw_c))
                        if target_c is None:
                            continue

                        # Normalize to [0, 1] and clip strictly to bounds
                        x1_n = float(np.clip(x1 / max(1, w), 0.0, 1.0))
                        y1_n = float(np.clip(y1 / max(1, h), 0.0, 1.0))
                        x2_n = float(np.clip(x2 / max(1, w), 0.0, 1.0))
                        y2_n = float(np.clip(y2 / max(1, h), 0.0, 1.0))

                        if x2_n > x1_n and y2_n > y1_n:
                            t_dets.append(
                                TeacherDetection(
                                    box_xyxy_norm=[x1_n, y1_n, x2_n, y2_n],
                                    confidence=float(conf),
                                    class_id=int(target_c),
                                )
                            )

                    results[frame_id].teacher_detections[teacher_idx] = t_dets

            # Teardown model and flush GPU VRAM
            del model
            self._flush_gpu_memory()

        return results

    @staticmethod
    def _probe_image_size(path: Path) -> tuple[int, int]:
        """Probes (height, width) of image."""
        img = cv2.imread(str(path))
        if img is not None:
            return img.shape[0], img.shape[1]
        return 640, 640

    @staticmethod
    def _flush_gpu_memory() -> None:
        """Flushes CUDA cache and runs garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

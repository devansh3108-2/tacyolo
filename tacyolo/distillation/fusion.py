"""Stage 2: Weighted Box Fusion (WBF) Consensus.

Uses the `ensemble-boxes` library to fuse candidate detections from all three teachers
(YOLOv8, YOLO11, RT-DETR) with configurable per-model weighting, IoU consensus thresholds,
and confidence filtering into standardized YOLO format text annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from tacyolo.distillation.ensemble import FrameDetections, TeacherDetection


@dataclass
class FusedDetection:
    """Consensus detection produced by Weighted Box Fusion."""
    class_id: int
    confidence: float
    xc: float
    yc: float
    bw: float
    bh: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_yolo_line(self) -> str:
        """Converts to normalized YOLO format: <class_id> <xc> <yc> <bw> <bh>."""
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.bw:.6f} {self.bh:.6f}"


class WeightedBoxFusionConsensus:
    """Fuses multi-teacher detections using Weighted Box Fusion (WBF)."""

    def __init__(
        self,
        weights: Sequence[float] | None = None,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.35,
        conf_type: str = "avg",
        allows_overflow: bool = False,
    ) -> None:
        self.weights = list(weights) if weights is not None else None
        self.iou_thr = float(iou_thr)
        self.skip_box_thr = float(skip_box_thr)
        self.conf_type = str(conf_type)
        self.allows_overflow = allows_overflow

    def fuse_frame(
        self,
        frame_dets: FrameDetections,
        weights: Sequence[float] | None = None,
    ) -> list[FusedDetection]:
        """Fuses candidate detections from all teachers for a single frame."""
        effective_weights = list(weights) if weights is not None else self.weights
        num_teachers = len(frame_dets.teacher_detections)

        if effective_weights is None:
            effective_weights = [1.0] * num_teachers
        elif len(effective_weights) != num_teachers:
            raise ValueError(
                f"Weights count ({len(effective_weights)}) must match teacher count ({num_teachers})."
            )

        boxes_list: list[list[list[float]]] = []
        scores_list: list[list[float]] = []
        labels_list: list[list[int]] = []

        total_input_boxes = 0
        for t_dets in frame_dets.teacher_detections:
            t_boxes: list[list[float]] = []
            t_scores: list[float] = []
            t_labels: list[int] = []

            for d in t_dets:
                # Ensure strictly valid [0, 1] range for ensemble-boxes
                x1, y1, x2, y2 = d.box_xyxy_norm
                x1 = float(np.clip(x1, 0.0, 1.0))
                y1 = float(np.clip(y1, 0.0, 1.0))
                x2 = float(np.clip(x2, 0.0, 1.0))
                y2 = float(np.clip(y2, 0.0, 1.0))
                if x2 > x1 and y2 > y1 and d.confidence > 0.0:
                    t_boxes.append([x1, y1, x2, y2])
                    t_scores.append(float(d.confidence))
                    t_labels.append(int(d.class_id))

            boxes_list.append(t_boxes)
            scores_list.append(t_scores)
            labels_list.append(t_labels)
            total_input_boxes += len(t_boxes)

        if total_input_boxes == 0:
            return []

        f_boxes, f_scores, f_labels = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=effective_weights,
            iou_thr=self.iou_thr,
            skip_box_thr=self.skip_box_thr,
            conf_type=self.conf_type,
            allows_overflow=self.allows_overflow,
        )

        fused_results: list[FusedDetection] = []
        for box, score, label in zip(f_boxes, f_scores, f_labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0.0 or bh <= 0.0:
                continue

            xc = x1 + bw / 2.0
            yc = y1 + bh / 2.0
            cid = int(round(float(label)))

            fused_results.append(
                FusedDetection(
                    class_id=cid,
                    confidence=float(score),
                    xc=float(np.clip(xc, 0.0, 1.0)),
                    yc=float(np.clip(yc, 0.0, 1.0)),
                    bw=float(np.clip(bw, 0.0, 1.0)),
                    bh=float(np.clip(bh, 0.0, 1.0)),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return fused_results

    def fuse_all_frames(
        self,
        frame_detections: dict[str, FrameDetections],
        weights: Sequence[float] | None = None,
    ) -> dict[str, list[FusedDetection]]:
        """Fuses all frames in a dictionary index."""
        fused_dataset: dict[str, list[FusedDetection]] = {}
        for fid, fd in frame_detections.items():
            fused_dataset[fid] = self.fuse_frame(fd, weights=weights)
        return fused_dataset

    @staticmethod
    def save_yolo_labels(
        fused_dets: list[FusedDetection],
        out_txt_path: Path,
    ) -> None:
        """Writes fused detections to a YOLO annotation text file."""
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [d.to_yolo_line() for d in fused_dets]
        text_content = "\n".join(lines) + ("\n" if lines else "")
        out_txt_path.write_text(text_content, encoding="utf-8")

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tacyolo.types import FrameResult


def _colormap_rd(rd: np.ndarray, size: tuple[int, int] = (160, 96)) -> np.ndarray:
    mag = np.abs(rd)
    mag = mag / (mag.max() + 1e-6)
    img = (np.clip(mag, 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)


def annotate(result: FrameResult, show_gate: bool = True, rd_inset: bool = True) -> np.ndarray:
    frame = result.frame.copy()
    if show_gate and result.gate.rois:
        for roi in result.gate.rois:
            cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), (80, 80, 80), 1)

    for det in result.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)

    for trk in result.tracks:
        if not trk.confirmed and trk.hits < 2:
            continue
        x1, y1, x2, y2 = [int(v) for v in trk.bbox]
        color = (40, 220, 40) if trk.confirmed else (180, 180, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"id{trk.track_id} {trk.hdc_class or trk.class_name}"
        if trk.hdc_score is not None:
            label += f" {trk.hdc_score:.2f}"
        cv2.putText(frame, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        vx, vy = trk.vx, trk.vy
        cx, cy = int(trk.x), int(trk.y)
        cv2.arrowedLine(frame, (cx, cy), (int(cx + 4 * vx), int(cy + 4 * vy)), color, 1, tipLength=0.3)

    for trk in result.extras.get("tentative") or []:
        x1, y1, x2, y2 = [int(v) for v in trk.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

    status = result.duty_mode.upper() if result.duty_mode else ("YOLO" if result.yolo_ran else "IDLE")
    energy = result.gate.energy
    txt = (
        f"{status}  yolo={int(result.yolo_ran)}  frac={result.yolo_frac:.2f}  "
        f"gate={energy:.1f}  pub={len(result.tracks)}"
    )
    cv2.putText(frame, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    if rd_inset and result.rf is not None:
        inset = _colormap_rd(result.rf.residual if result.rf.residual.size else result.rf.rd_map)
        h, w = inset.shape[:2]
        frame[8 : 8 + h, frame.shape[1] - 8 - w : frame.shape[1] - 8] = inset
        cv2.rectangle(
            frame,
            (frame.shape[1] - 8 - w, 8),
            (frame.shape[1] - 8, 8 + h),
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "RD",
            (frame.shape[1] - 8 - w, 8 + h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return frame


def write_video(path: str | Path, frames: list[np.ndarray], fps: float = 30.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path

from __future__ import annotations

import time
from pathlib import Path

from tacyolo.runtime.backend import InferenceBackend, create_backend
from tacyolo.types import Detection, ROI
import numpy as np


class Detector:
    def __init__(
        self,
        weights: str = "dummy",
        backend: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "auto",
        on_crops: bool = False,
        providers: list[str] | None = None,
        world_classes: list[str] | None = None,
    ) -> None:
        self.on_crops = on_crops
        self.backend: InferenceBackend = create_backend(
            weights=weights,
            backend=backend,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            providers=providers,
            world_classes=world_classes,
        )

    @property
    def name(self) -> str:
        return self.backend.name

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI] | None = None,
        force_crops: bool | None = None,
    ) -> list[Detection]:
        t0 = time.perf_counter()
        use_crops = bool(force_crops) if force_crops is not None else self.on_crops
        if use_crops and rois:
            dets: list[Detection] = []
            h, w = frame.shape[:2]
            for roi in rois:
                x1, y1, x2, y2 = roi.clip(w, h).as_xyxy()
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_roi = ROI(0, 0, int(crop.shape[1]), int(crop.shape[0]))
                local = self.backend.infer(crop, rois=[crop_roi])
                for det in local:
                    bx1, by1, bx2, by2 = det.bbox
                    det.bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                    dets.append(det)
            _ = (time.perf_counter() - t0) * 1000.0
            return dets
        dets = self.backend.infer(frame, rois=rois)
        return dets

    def export(self, fmt: str, int8: bool = False, out_dir: str | Path = "runs") -> Path:
        return self.backend.export(fmt, int8=int8, out_dir=out_dir)

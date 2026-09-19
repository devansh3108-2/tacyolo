from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tacyolo.types import Detection, ROI


@dataclass
class InferResult:
    detections: list[Detection]
    backend: str
    elapsed_ms: float = 0.0


class InferenceBackend:
    name = "base"

    def infer(self, frame: np.ndarray, rois: list[ROI] | None = None) -> list[Detection]:
        raise NotImplementedError

    def embed_crop(self, crop: np.ndarray) -> np.ndarray | None:
        """Identity embedding for a detection chip. None = use color HDC fallback."""
        return None

    def export(self, fmt: str, int8: bool = False, out_dir: str | Path = "runs") -> Path:
        raise NotImplementedError("Export is not supported by this backend")


class DummyBackend(InferenceBackend):
    """Turns motion ROIs into detections so the fusion stack runs without YOLO weights."""

    name = "dummy"

    def infer(self, frame: np.ndarray, rois: list[ROI] | None = None) -> list[Detection]:
        dets: list[Detection] = []
        if not rois:
            return dets
        h, w = frame.shape[:2]
        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = roi.clip(w, h).as_xyxy()
            crop = frame[y1:y2, x1:x2]
            name = _color_name(crop)
            dets.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    conf=0.6,
                    class_id=i,
                    class_name=name,
                    crop=crop.copy() if crop.size else None,
                )
            )
        return dets


class UltralyticsBackend(InferenceBackend):
    name = "pytorch"

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        world_classes: list[str] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        stem = Path(weights).name.lower()
        if "world" in stem:
            try:
                from ultralytics import YOLOWorld

                self.model = YOLOWorld(weights)
            except Exception:
                self.model = YOLO(weights)
            from tacyolo.weights import DEFAULT_WORLD_CLASSES

            classes = world_classes or DEFAULT_WORLD_CLASSES
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(classes)
        else:
            self.model = YOLO(weights)
        names = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {}

    def infer(self, frame: np.ndarray, rois: list[ROI] | None = None) -> list[Detection]:
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        dets: list[Detection] = []
        if not results:
            return dets
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        h, w = frame.shape[:2]
        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = [float(v) for v in box]
            if rois and not _overlaps_any((x1, y1, x2, y2), rois):
                continue
            x1c, y1c = int(max(0, x1)), int(max(0, y1))
            x2c, y2c = int(min(w, x2)), int(min(h, y2))
            crop = frame[y1c:y2c, x1c:x2c]
            dets.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    conf=float(conf),
                    class_id=int(cls_id),
                    class_name=self.names.get(int(cls_id), str(int(cls_id))),
                    crop=crop.copy() if crop.size else None,
                )
            )
        return dets

    def embed_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        try:
            embs = self.model.embed(crop, verbose=False)
        except Exception:
            return None
        if not embs:
            return None
        vec = embs[0]
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().numpy()
        out = np.asarray(vec, dtype=np.float32).ravel()
        n = float(np.linalg.norm(out))
        if n > 0:
            out = out / n
        return out
        import shutil

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt in {"engine", "tensorrt", "trt"}:
            raise NotImplementedError(
                "TensorRT .engine export is deferred until a Jetson/NVIDIA TensorRT toolchain is available."
            )
        kwargs = {"format": fmt, "imgsz": self.imgsz}
        if int8 and fmt in {"onnx", "openvino"}:
            kwargs["int8"] = True
        path = Path(str(self.model.export(**kwargs)))
        dest = out_dir / path.name
        if dest.resolve() != path.resolve():
            shutil.copy2(path, dest)
            return dest
        return path


class OnnxBackend(InferenceBackend):
    name = "onnx"

    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.imgsz = imgsz
        self.conf = conf
        avail = ort.get_available_providers()
        requested = [
            p
            for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
            if p != "TensorrtExecutionProvider"
        ]
        chosen = [p for p in requested if p in avail]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(weights, providers=chosen)
        self.providers = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        print(f"onnxruntime providers={self.providers}")

    def infer(self, frame: np.ndarray, rois: list[ROI] | None = None) -> list[Detection]:
        blob = _letterbox(frame, self.imgsz)
        outputs = self.session.run(None, {self.input_name: blob})
        # Ultralytics ONNX layouts vary; Dummy-like fallback if parse fails.
        try:
            dets = _parse_yolo_onnx(outputs[0], frame.shape, self.imgsz, self.conf)
        except Exception:
            dets = []
        if rois:
            dets = [d for d in dets if _overlaps_any(d.bbox, rois)]
        h, w = frame.shape[:2]
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            det.crop = frame[int(max(0, y1)) : int(min(h, y2)), int(max(0, x1)) : int(min(w, x2))].copy()
        return dets


class TensorRTBackend(InferenceBackend):
    name = "tensorrt"

    def __init__(self, weights: str, **_: object) -> None:
        raise NotImplementedError(
            "TensorRT runtime is a Jetson export path. Use backend=dummy, pytorch, or onnx on desktop."
        )


def _color_name(crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return "object"
    mean = crop.reshape(-1, 3).mean(axis=0)
    b, g, r = mean
    if r >= g and r >= b and r > 80:
        return "red"
    if b >= g and b >= r and b > 80:
        return "blue"
    if g >= r and g >= b and g > 80:
        return "green"
    return "object"


def _overlaps_any(box: tuple[float, float, float, float], rois: list[ROI], min_iou: float = 0.05) -> bool:
    x1, y1, x2, y2 = box
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi.as_xyxy()
        ix1, iy1 = max(x1, rx1), max(y1, ry1)
        ix2, iy2 = min(x2, rx2), min(y2, ry2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > 0 and inter / area >= min_iou:
            return True
    return False


def _letterbox(frame: np.ndarray, imgsz: int) -> np.ndarray:
    import cv2

    img = frame
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((imgsz, imgsz, 3), dtype=np.float32)
    canvas[:nh, :nw] = resized
    blob = np.transpose(canvas, (2, 0, 1))[None, ...]
    return blob


def _parse_yolo_onnx(out: np.ndarray, hw: tuple[int, ...], imgsz: int, conf: float) -> list[Detection]:
    arr = np.squeeze(out)
    if arr.ndim != 2:
        return []
    if arr.shape[0] in (84, 85) or arr.shape[0] < arr.shape[1]:
        arr = arr.T
    dets: list[Detection] = []
    h, w = hw[:2]
    scale = max(h, w) / float(imgsz)
    for row in arr:
        if row.shape[0] < 6:
            continue
        cx, cy, bw, bh = row[:4]
        scores = row[4:]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < conf:
            continue
        x1 = (cx - bw / 2) * scale
        y1 = (cy - bh / 2) * scale
        x2 = (cx + bw / 2) * scale
        y2 = (cy + bh / 2) * scale
        dets.append(
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                conf=score,
                class_id=class_id,
                class_name=str(class_id),
            )
        )
    return dets


def create_backend(
    weights: str = "auto",
    backend: str = "auto",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    providers: list[str] | None = None,
    world_classes: list[str] | None = None,
) -> InferenceBackend:
    from tacyolo.weights import default_device, resolve_weights

    backend = (backend or "auto").lower()
    device = default_device(device)
    weights = resolve_weights(weights)
    path = Path(str(weights))

    if backend == "dummy" or str(weights).lower() in {"dummy", "none", ""}:
        return DummyBackend()
    if backend == "tensorrt":
        return TensorRTBackend(str(weights))
    if backend == "onnx" or path.suffix.lower() == ".onnx":
        return OnnxBackend(str(weights), imgsz=imgsz, conf=conf, providers=providers)
    if backend == "pytorch":
        return UltralyticsBackend(
            str(weights),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            world_classes=world_classes,
        )

    if path.suffix.lower() == ".onnx" and path.exists():
        return OnnxBackend(str(path), imgsz=imgsz, conf=conf, providers=providers)
    if path.suffix.lower() == ".engine":
        return TensorRTBackend(str(path))
    if path.exists() and path.suffix.lower() in {".pt", ".pth"}:
        try:
            return UltralyticsBackend(
                str(weights),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                world_classes=world_classes,
            )
        except Exception:
            return DummyBackend()
    return DummyBackend()

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"

MILITARY_PT = WEIGHTS_DIR / "military_yolov8s.pt"
YOLO11N_PT = WEIGHTS_DIR / "yolo11n.pt"
YOLO11S_PT = WEIGHTS_DIR / "yolo11s.pt"
WORLD_PT = WEIGHTS_DIR / "yolov8s-worldv2.pt"
TACTICAL_PT = WEIGHTS_DIR / "tactical_yolo11s.pt"
TACTICAL_N_PT = WEIGHTS_DIR / "tactical_yolo11n.pt"

DEFAULT_WORLD_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "tank",
    "armored vehicle",
    "drone",
    "uav",
    "helicopter",
    "airplane",
    "boat",
]


def default_device(requested: str | None = None) -> str:
    if requested and str(requested).lower() not in {"auto", "", "none"}:
        return str(requested)
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def resolve_weights(name: str | None) -> str:
    raw = (name or "auto").strip()
    lowered = raw.lower()
    if lowered in {"dummy", "none"}:
        return "dummy"
    if lowered in {"military", "tactical"}:
        if lowered == "tactical" and TACTICAL_PT.exists():
            return str(TACTICAL_PT)
        return str(MILITARY_PT if MILITARY_PT.exists() else raw)
    if lowered in {"world", "open-vocab", "openvocab"}:
        return str(WORLD_PT if WORLD_PT.exists() else "yolov8s-worldv2.pt")
    if lowered in {"auto", "default", ""}:
        for candidate in (TACTICAL_PT, TACTICAL_N_PT, MILITARY_PT, ROOT / "yolo11n.pt", YOLO11N_PT):
            if candidate.exists():
                return str(candidate)
        return "yolo11n.pt"
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    for folder in (WEIGHTS_DIR, ROOT):
        alt = folder / path.name
        if alt.exists():
            return str(alt)
    return raw

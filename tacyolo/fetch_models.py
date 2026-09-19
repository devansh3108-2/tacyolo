"""Download the detectors that are actually useful on this desktop."""

from __future__ import annotations

import shutil
from pathlib import Path

from tacyolo.weights import MILITARY_PT, ROOT, WEIGHTS_DIR, WORLD_PT, YOLO11N_PT, YOLO11S_PT


_HF_REPOS = (
    ("mmoz-root/military-vehicle-detection_yolov8", "best.pt"),
    ("mmoz-root/military-vehicle-detection-yolov8", "best.pt"),
)


def _copy_if_present(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.resolve() != dest.resolve():
        shutil.copy2(src, dest)


def fetch_yolo11n() -> Path:
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")
    src = Path(getattr(model, "ckpt_path", None) or ROOT / "yolo11n.pt")
    if not src.exists():
        src = ROOT / "yolo11n.pt"
    _copy_if_present(src, YOLO11N_PT)
    return YOLO11N_PT if YOLO11N_PT.exists() else src


def fetch_yolo11s() -> Path:
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11s.pt")
    src = Path(getattr(model, "ckpt_path", None) or ROOT / "yolo11s.pt")
    if not src.exists():
        src = ROOT / "yolo11s.pt"
    _copy_if_present(src, YOLO11S_PT)
    return YOLO11S_PT if YOLO11S_PT.exists() else src


def fetch_world() -> Path:
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8s-worldv2.pt")
    src = Path(getattr(model, "ckpt_path", None) or ROOT / "yolov8s-worldv2.pt")
    if not src.exists():
        src = ROOT / "yolov8s-worldv2.pt"
    _copy_if_present(src, WORLD_PT)
    return WORLD_PT if WORLD_PT.exists() else src


def fetch_military() -> Path:
    from huggingface_hub import hf_hub_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for repo_id, filename in _HF_REPOS:
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copy2(downloaded, MILITARY_PT)
            return MILITARY_PT
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not download military weights: {last_error}")


def fetch_all() -> dict[str, str]:
    out: dict[str, str] = {}
    out["yolo11n"] = str(fetch_yolo11n())
    try:
        out["yolo11s"] = str(fetch_yolo11s())
    except Exception as exc:  # noqa: BLE001
        out["yolo11s"] = f"FAILED: {exc}"
    try:
        out["military"] = str(fetch_military())
    except Exception as exc:  # noqa: BLE001
        out["military"] = f"FAILED: {exc}"
    try:
        out["world"] = str(fetch_world())
    except Exception as exc:  # noqa: BLE001
        out["world"] = f"FAILED: {exc}"
    return out


def main() -> int:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    results = fetch_all()
    for key, value in results.items():
        print(f"{key}: {value}")
    return 0 if all(not str(v).startswith("FAILED") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

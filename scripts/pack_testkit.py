"""Build a sendable test zip: code + weights, no training data."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / "Desktop" / "TACYOLO_TestKit.zip"
STAGE = ROOT / "_testkit_stage"

INCLUDE_DIRS = ("tacyolo", "examples", "tests", "configs")
INCLUDE_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "HOW_TO_TEST.txt",
    "bus.jpg",
)
WEIGHTS = (
    "yolo11n.pt",
    "yolo11s.pt",
    "military_yolov8s.pt",
    "yolov8s-worldv2.pt",
)
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def _copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(src)
        dest = dst / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def build() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    kit = STAGE / "TACYOLO_TestKit"
    kit.mkdir(parents=True)

    for name in INCLUDE_DIRS:
        _copy_tree(ROOT / name, kit / name)
    for name in INCLUDE_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, kit / name)

    wdir = kit / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    for name in WEIGHTS:
        for folder in (ROOT / "weights", ROOT):
            src = folder / name
            if src.exists():
                shutil.copy2(src, wdir / name)
                break

    pitch = ROOT / "pitch" / "TACYOLO_Company_Pitch.pdf"
    if pitch.exists():
        docs = kit / "docs"
        docs.mkdir(exist_ok=True)
        shutil.copy2(pitch, docs / "TACYOLO_Company_Pitch.pdf")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in kit.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE).as_posix())
    shutil.rmtree(STAGE)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")

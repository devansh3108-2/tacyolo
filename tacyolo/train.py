"""Fine-tune YOLO on the merged tactical dataset, including part-by-part curriculum."""

from __future__ import annotations

import shutil
from pathlib import Path

from tacyolo.trainset import CLASS_NAMES, DATA_ROOT, prepare
from tacyolo.weights import ROOT, WEIGHTS_DIR, YOLO11S_PT, default_device

CURRICULUM = [
    {"name": "1_core", "sources": ["visdrone", "coco128", "hf_drone", "tank"], "epochs": 20},
    {"name": "2_drones", "sources": ["seraphim", "aerial"], "epochs": 20},
    {"name": "3_military", "sources": ["military", "aircraft"], "epochs": 15},
    {"name": "4_google_oi", "sources": ["open_images"], "epochs": 20},
    {"name": "5_mix", "sources": [], "epochs": 25},
]


def _ensure_base_weights(model: str | None) -> Path:
    if model:
        path = Path(model)
        if path.exists():
            return path
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if YOLO11S_PT.exists():
        return YOLO11S_PT
    from ultralytics import YOLO

    detector = YOLO("yolo11s.pt")
    src = Path(getattr(detector, "ckpt_path", None) or ROOT / "yolo11s.pt")
    if src.exists() and src.resolve() != YOLO11S_PT.resolve():
        shutil.copy2(src, YOLO11S_PT)
    if YOLO11S_PT.exists():
        return YOLO11S_PT
    if src.exists():
        return src
    return ROOT / "yolo11s.pt"


def _copy_best(best: Path) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest_s = WEIGHTS_DIR / "tactical_yolo11s.pt"
    dest_n = WEIGHTS_DIR / "tactical_yolo11n.pt"
    if best.exists():
        shutil.copy2(best, dest_s)
        shutil.copy2(best, dest_n)
        print(f"trained weights: {dest_s}")
        print(f"classes: {CLASS_NAMES}")
        return dest_s
    raise FileNotFoundError(f"Training finished but best.pt was not found at {best}")


def train(
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 8,
    device: str | None = None,
    model: str | None = None,
    name: str = "tactical",
) -> Path:
    yaml_path = DATA_ROOT / "tactical.yaml"
    if not yaml_path.exists() or not (DATA_ROOT / "images" / "train").exists():
        yaml_path = prepare()

    weights = _ensure_base_weights(model)
    device = default_device(device)

    from ultralytics import YOLO
    from tacyolo.gpu_cap import attach_ultralytics

    detector = YOLO(str(weights))
    attach_ultralytics(detector)
    try:
        results = detector.train(
            data=str(yaml_path),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            device=device,
            project=str(ROOT / "runs" / "train"),
            name=name,
            exist_ok=True,
            pretrained=True,
            workers=1,
            patience=max(8, int(epochs) // 3),
            plots=False,
            val=True,
            verbose=True,
            amp=True,
            close_mosaic=min(5, max(1, int(epochs) // 5)),
            cos_lr=True,
            mixup=0.05,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or batch <= 2:
            raise
        print(f"CUDA OOM at batch={batch}, retrying with batch={batch // 2}")
        detector = YOLO(str(weights))
        attach_ultralytics(detector)
        results = detector.train(
            data=str(yaml_path),
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=max(2, batch // 2),
            device=device,
            project=str(ROOT / "runs" / "train"),
            name=name,
            exist_ok=True,
            pretrained=True,
            workers=1,
            patience=max(8, int(epochs) // 3),
            plots=False,
            val=True,
            verbose=True,
            amp=True,
            close_mosaic=min(5, max(1, int(epochs) // 5)),
            cos_lr=True,
        )
    best = Path(getattr(results, "save_dir", ROOT / "runs" / "train" / name)) / "weights" / "best.pt"
    if not best.exists():
        best = ROOT / "runs" / "train" / name / "weights" / "best.pt"
    return _copy_best(best)


def train_curriculum(
    epochs: int | None = None,
    imgsz: int = 640,
    batch: int = 8,
    device: str | None = None,
    model: str | None = None,
) -> Path:
    last = _ensure_base_weights(model)
    print(f"curriculum start weights={last}")
    for stage in CURRICULUM:
        sources = list(stage["sources"])
        stage_epochs = int(epochs) if epochs and epochs > 0 else int(stage["epochs"])
        print(f"\n===== STAGE {stage['name']} sources={sources or 'already-merged'} epochs={stage_epochs} =====")
        if sources:
            prepare(sources)
        last = train(
            epochs=stage_epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            model=str(last),
            name=f"tactical_{stage['name']}",
        )
        print(f"stage {stage['name']} done -> {last}")
    return last


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train tactical YOLO11s")
    parser.add_argument("--epochs", type=int, default=0, help="0 = use curriculum defaults")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--curriculum", action="store_true", help="Train part-by-part, adding a new dataset after each stage")
    parser.add_argument("--sources", default="", help="comma-separated sources for prepare-only")
    args = parser.parse_args(argv)
    if args.prepare_only:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None
        prepare(sources)
        return 0
    if args.curriculum or args.epochs == 0:
        train_curriculum(
            epochs=args.epochs if args.epochs > 0 else None,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            model=args.model,
        )
        return 0
    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

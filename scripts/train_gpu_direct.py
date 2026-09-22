"""Stage 2: Blazing-Fast Pure GPU Training Directly from Google Drive.

Runs on A100 GPU in Colab:
1. Reads the already-prepared 640x640 tiles directly from Google Drive.
2. ZERO copy time, zero download time, zero CPU wait!
3. Maximum GPU utilization with batch=32/64 on A100.
4. Checkpoints best.pt directly into Google Drive trained_weights/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from scripts.colab_tile_train import DriveLayout
except ImportError:
    from colab_tile_train import DriveLayout



def train_gpu_direct(
    drive_root: str | Path = "/content/drive/MyDrive/TACYOLO",
    epochs: int = 100,
    batch_size: int = 32,
    model_name: str = "yolo11s.pt",
    imgsz: int = 640,
) -> Path:
    layout = DriveLayout(drive_root)
    data_yaml = layout.tiled_batches / "data.yaml"

    if not data_yaml.exists():
        import yaml
        try:
            from scripts.colab_tile_train import CLASS_NAMES
        except ImportError:
            from colab_tile_train import CLASS_NAMES

        train_img_dir = layout.tiled_batches / "images" / "train"
        if train_img_dir.exists() and any(train_img_dir.glob("*.*")):
            print(f"⚡ Detected tiled images in {train_img_dir}! Auto-generating missing data.yaml...")
            yaml_content = {
                "path": str(layout.tiled_batches.resolve()),
                "train": "images/train",
                "val": "images/val" if (layout.tiled_batches / "images" / "val").exists() else "images/train",
                "names": {i: name for i, name in enumerate(CLASS_NAMES)},
                "nc": len(CLASS_NAMES),
            }
            layout.tiled_batches.mkdir(parents=True, exist_ok=True)
            data_yaml.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")
            print(f"✅ Generated {data_yaml}")
        else:
            found_yamls = list(layout.root.rglob("data.yaml")) if layout.root.exists() else []
            if found_yamls:
                data_yaml = found_yamls[0]
                print(f"✅ Discovered dataset YAML at: {data_yaml}")
            else:
                print(f"❌ Error: Master dataset not found at {data_yaml}!")
                print(f"📁 Diagnostic inspect of {layout.root}:")
                if layout.root.exists():
                    for item in layout.root.iterdir():
                        print(f"   - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
                        if item.is_dir():
                            sub_items = list(item.glob("*"))[:5]
                            print(f"     Sample files: {[p.name for p in sub_items]}")
                else:
                    print(f"   ⚠️ {layout.root} does not exist. Is Google Drive mounted?")
                print("\n👉 Please run scripts/prepare_tiles_cpu.py first to generate the tiles on Google Drive.")
                sys.exit(1)

    # Resume from existing best.pt if present on Drive, else base model
    best_on_drive = layout.trained_weights / "best.pt"
    starting_weights = str(best_on_drive) if best_on_drive.exists() else model_name

    print("=================================================================")
    print("🔥 STAGE 2: PURE A100 GPU TRAINING (DIRECT DRIVE STREAMING)")
    print(f"📁 Dataset YAML: {data_yaml}")
    print(f"🏋️ Starting Weights: {starting_weights}")
    print(f"⚡ Batch Size: {batch_size} | Epochs: {epochs} | Resolution: {imgsz}x{imgsz}")
    print("=================================================================\n")

    from ultralytics import YOLO

    if "world" in str(starting_weights).lower():
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(starting_weights)
        except Exception:
            model = YOLO(starting_weights)
    else:
        model = YOLO(starting_weights)

    project_dir = layout.working_testing / "runs"
    project_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=str(project_dir),
        name="gpu_direct_train",
        exist_ok=True,
        device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        workers=8,
        cache="disk",  # Cache directly for high-speed epochs
    )

    # Save to trained_weights on Drive
    trained_best = project_dir / "gpu_direct_train" / "weights" / "best.pt"
    if trained_best.exists():
        shutil.copy2(trained_best, layout.trained_weights / "best.pt")
        shutil.copy2(trained_best, layout.trained_weights / "tactical_yolo11s.pt")
        print("\n=================================================================")
        print(f"🎉 TRAINING COMPLETE! Best weights saved to:")
        print(f"⭐ {layout.trained_weights / 'best.pt'}")
        print("=================================================================")

    return layout.trained_weights / "best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="TACYOLO Direct-Drive Pure GPU Training")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/TACYOLO", help="Google Drive Root Path")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for A100 GPU (default: 32)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tile resolution (default: 640)")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model (e.g. yolo11s.pt or yolov8s-worldv2.pt)")
    args = parser.parse_args()

    train_gpu_direct(
        drive_root=args.drive_root,
        epochs=args.epochs,
        batch_size=args.batch,
        model_name=args.model,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

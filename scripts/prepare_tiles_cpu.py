"""Stage 1: CPU-Only Data Preparation & Tiling.

Runs on free/cheap CPU runtime in Colab:
1. Downloads all 4 Pillars (Drones, CCTV, OpenImages, Thermal) directly into Google Drive.
2. Slices them into 640x640 overlapping tiles inside Google Drive.
3. Generates the master data.yaml.
4. Purges raw files to save Drive space.
ZERO GPU usage, saving 100% of your A100 credits for training!
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from scripts.colab_tile_train import (
    CLASS_NAMES,
    UNIFIED_4_IN_1_DATASETS,
    DriveLayout,
    ImageTiler,
    get_dir_size_gb,
)


def prepare_and_tile_all(drive_root: str | Path = "/content/drive/MyDrive/TACYOLO", tile_size: int = 640) -> Path:
    layout = DriveLayout(drive_root)
    layout.setup()
    tiler = ImageTiler(tile_size=tile_size)

    # Master tiled dataset directories
    master_train_imgs = layout.tiled_batches / "images" / "train"
    master_train_lbls = layout.tiled_batches / "labels" / "train"
    master_val_imgs = layout.tiled_batches / "images" / "val"
    master_val_lbls = layout.tiled_batches / "labels" / "val"

    for p in [master_train_imgs, master_train_lbls, master_val_imgs, master_val_lbls]:
        p.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("🚀 STAGE 1: CPU-ONLY DATA PREPARATION & TILING (ZERO GPU WASTED)")
    print(f"📁 Target Drive Root: {layout.root}")
    print("=================================================================\n")

    total_tiles = 0
    t0_all = time.time()

    for idx, ds in enumerate(UNIFIED_4_IN_1_DATASETS, 1):
        ref = ds["ref"]
        tag = ds["tag"]
        cls_map = ds.get("cls_map")

        stage_raw_dir = layout.raw_data / tag
        stage_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(UNIFIED_4_IN_1_DATASETS)}] ⬇️ Downloading {ref} directly to Drive...")
        try:
            import kagglehub
            raw_path = Path(kagglehub.dataset_download(ref))
            for item in raw_path.iterdir():
                dst = stage_raw_dir / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
        except Exception as e:
            print(f"kagglehub failed ({e}), trying Kaggle CLI...")
            os.system(f"kaggle datasets download -d {ref} -p '{stage_raw_dir}' --unzip")

        raw_size = get_dir_size_gb(stage_raw_dir)
        print(f"[{idx}/{len(UNIFIED_4_IN_1_DATASETS)}] 🔪 Tiling {raw_size:.2f} GB of images for {tag}...")

        # Discover images
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in stage_raw_dir.rglob("*") if p.suffix.lower() in img_extensions]

        count = 0
        for i, img_path in enumerate(image_files):
            is_val = (i % 10 == 0)
            target_img_dir = master_val_imgs if is_val else master_train_imgs
            target_lbl_dir = master_val_lbls if is_val else master_train_lbls

            found_lbl = None
            cand = img_path.parent / (img_path.stem + ".txt")
            if cand.exists():
                found_lbl = cand
            elif len(img_path.parents) > 1:
                cand2 = img_path.parents[1] / "labels" / (img_path.stem + ".txt")
                if cand2.exists():
                    found_lbl = cand2

            n = tiler.tile_image_and_labels(img_path, found_lbl, target_img_dir, target_lbl_dir, cls_map)
            count += n

        total_tiles += count
        print(f"[{idx}/{len(UNIFIED_4_IN_1_DATASETS)}] ✅ Generated {count} tiles (Running total: {total_tiles} tiles)")

        # Immediately purge the raw archive to keep Drive clean
        print(f"[{idx}/{len(UNIFIED_4_IN_1_DATASETS)}] 🧹 Purging raw archive for {tag}...")
        if stage_raw_dir.exists():
            shutil.rmtree(stage_raw_dir, ignore_errors=True)

    # Write unified master data.yaml
    yaml_path = layout.tiled_batches / "data.yaml"
    import yaml
    yaml_content = {
        "path": str(layout.tiled_batches.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    yaml_path.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")

    elapsed_min = (time.time() - t0_all) / 60.0
    print("\n=================================================================")
    print(f"🎉 PREPARATION COMPLETE in {elapsed_min:.1f} minutes!")
    print(f"📊 Total Tiles Generated on Google Drive: {total_tiles}")
    print(f"📄 Master YAML for GPU Training: {yaml_path}")
    print("👉 Now you can switch Colab runtime to A100 GPU and run scripts/train_gpu_direct.py!")
    print("=================================================================")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="TACYOLO CPU Pre-Tiling Stage")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/TACYOLO", help="Google Drive Root Path")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile resolution")
    args = parser.parse_args()

    prepare_and_tile_all(drive_root=args.drive_root, tile_size=args.tile_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())

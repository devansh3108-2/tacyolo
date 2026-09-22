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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from scripts.colab_tile_train import (
        CLASS_NAMES,
        UNIFIED_4_IN_1_DATASETS,
        DriveLayout,
        ImageTiler,
        discover_kaggle_datasets,
        get_dir_size_gb,
    )
except ImportError:
    from colab_tile_train import (
        CLASS_NAMES,
        UNIFIED_4_IN_1_DATASETS,
        DriveLayout,
        ImageTiler,
        discover_kaggle_datasets,
        get_dir_size_gb,
    )



def prepare_and_tile_all(
    drive_root: str | Path = "/content/drive/MyDrive/TACYOLO",
    tile_size: int = 640,
    auto_discover: int | None = None,
) -> Path:
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

    datasets_to_process = UNIFIED_4_IN_1_DATASETS
    if auto_discover:
        print(f"🌐 Querying Kaggle for {auto_discover} tactical datasets across all 4 pillars...")
        datasets_to_process = discover_kaggle_datasets(count=auto_discover)

    # Load persistent pipeline state to never repeat already-completed datasets
    state = layout.load_state()
    completed_datasets = set(state.get("completed_datasets", []))

    # Auto-seed completed datasets if existing tiles are found from the previous run
    existing_tile_count = sum(1 for _ in master_train_imgs.glob("*.*"))
    if existing_tile_count > 1000 and not completed_datasets:
        print(f"📦 Detected {existing_tile_count} existing tiles from previous run. Auto-registering initial 14 datasets as completed...")
        for b_ds in UNIFIED_4_IN_1_DATASETS:
            completed_datasets.add(b_ds["ref"])
            completed_datasets.add(b_ds.get("tag", b_ds["ref"].replace("/", "_")))
        state["completed_datasets"] = list(completed_datasets)
        layout.save_state(state)

    print("=================================================================")
    print("🚀 STAGE 1: CPU-ONLY DATA PREPARATION & TILING (ZERO GPU WASTED)")
    print(f"📁 Target Drive Root: {layout.root}")
    print(f"📦 Total Datasets in Schedule: {len(datasets_to_process)}")
    print(f"✅ Already Completed: {len(completed_datasets) // 2} datasets")
    print(f"📊 Existing Tiles on Drive: {existing_tile_count}")
    print("=================================================================\n")

    total_tiles = existing_tile_count
    t0_all = time.time()

    for idx, ds in enumerate(datasets_to_process, 1):
        ref = ds["ref"]
        tag = ds.get("tag", ref.replace("/", "_"))
        cls_map = ds.get("cls_map")

        # Skip if already finished!
        if ref in completed_datasets or tag in completed_datasets:
            print(f"[{idx}/{len(datasets_to_process)}] ⏭️ Skipping already completed dataset: {ref}")
            continue

        stage_raw_dir = layout.raw_data / tag
        stage_raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(datasets_to_process)}] ⬇️ Downloading {ref} directly to Drive...")
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
        print(f"[{idx}/{len(datasets_to_process)}] 🔪 Tiling {raw_size:.2f} GB of images for {tag}...")

        # Discover images and index labels
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in stage_raw_dir.rglob("*") if p.suffix.lower() in img_extensions]
        label_index = {p.stem: p for p in stage_raw_dir.rglob("*.txt") if "label" in str(p).lower() or p.parent.name in ("train", "val", "labels", "default")}

        count = 0
        for i, img_path in enumerate(image_files):
            is_val = (i % 10 == 0)
            target_img_dir = master_val_imgs if is_val else master_train_imgs
            target_lbl_dir = master_val_lbls if is_val else master_train_lbls

            found_lbl = label_index.get(img_path.stem)
            if not found_lbl:
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
        print(f"[{idx}/{len(datasets_to_process)}] ✅ Generated {count} tiles (Running total: {total_tiles} tiles)")

        # Immediately purge the raw archive to keep Drive clean
        print(f"[{idx}/{len(datasets_to_process)}] 🧹 Purging raw archive for {tag}...")
        if stage_raw_dir.exists():
            shutil.rmtree(stage_raw_dir, ignore_errors=True)

        # Mark this dataset as completed so it is NEVER repeated
        completed_datasets.add(ref)
        completed_datasets.add(tag)
        state["completed_datasets"] = list(completed_datasets)
        state["total_tiles_generated"] = total_tiles
        layout.save_state(state)

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
    parser.add_argument("--auto-discover", type=int, default=None, help="Auto-discover N tactical datasets from Kaggle (e.g. 500)")
    args = parser.parse_args()

    prepare_and_tile_all(drive_root=args.drive_root, tile_size=args.tile_size, auto_discover=args.auto_discover)
    return 0


if __name__ == "__main__":
    sys.exit(main())

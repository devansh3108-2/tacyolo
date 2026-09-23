"""Download, Save, & GPU-Tile ALL 6 C-UAS and Military Armor Datasets Into ONE Folder.

This script:
1. Downloads all 6 datasets (Seraphim, Anti-UAV, Drone-vs-Bird, Military Vehicles,
   VisDrone, DOTA-v2) into datasets/tactical_cuas_armor/sources/<dataset_name>/
2. GPU-tiles every image into 640x640 YOLO-ready tiles with bounding-box re-projection
3. Outputs everything into a single unified folder:
   datasets/tactical_cuas_armor/images/{train,val,test}
   datasets/tactical_cuas_armor/labels/{train,val,test}
   datasets/tactical_cuas_armor/data.yaml

Run:
    python scripts/download_6_datasets.py
    python scripts/download_6_datasets.py --dataset seraphim
    python scripts/download_6_datasets.py --skip-tile   # download only, tile later
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UNIFIED_DIR = ROOT / "datasets" / "tactical_cuas_armor"
SOURCES_DIR = UNIFIED_DIR / "sources"
IMAGES_DIR = UNIFIED_DIR / "images"
LABELS_DIR = UNIFIED_DIR / "labels"
STATE_FILE = UNIFIED_DIR / "download_state.json"

# ──────────────────────────────────────────────────────────────────────────────
# The 6 Datasets: download sources, class mappings, and ingestion methods
# ──────────────────────────────────────────────────────────────────────────────
DATASETS: dict[str, dict[str, Any]] = {
    # ─── A. Counter-UAS & Aerial Threats ──────────────────────────────────────
    "seraphim": {
        "name": "Seraphim Drone Detection Dataset (83K images, YOLO 640x640)",
        "dest": SOURCES_DIR / "seraphim",
        "download_method": "huggingface",
        "hf_repo": "lgrzybowski/seraphim-drone-detection-dataset",
        "git_repo": "https://github.com/Seraphim-Defence-Systems/seraphim-drone-detection-dataset.git",
        "kaggle_fallback": None,
        "class_map": {0: 2, 1: 3, 2: 2},  # rotary->drone, fixed-wing->fixed_wing_uav, hybrid->drone
    },
    "anti_uav": {
        "name": "Anti-UAV Benchmark (Thermal IR + RGB multi-spectral)",
        "dest": SOURCES_DIR / "anti_uav",
        "download_method": "git",
        "hf_repo": None,
        "git_repo": "https://github.com/nvhuynh16/YOLO-applied-to-Anti-UAV.git",
        "kaggle_fallback": None,
        "class_map": {0: 2},  # drone
    },
    "drone_vs_bird": {
        "name": "Drone-vs-Bird Challenge (Bird/Drone discrimination)",
        "dest": SOURCES_DIR / "drone_vs_bird",
        "download_method": "roboflow",
        "hf_repo": None,
        "git_repo": None,
        "kaggle_fallback": "beniroy/drone-vs-bird-detection",
        "roboflow_workspace": "drone-vs-bird",
        "roboflow_project": "drone-vs-bird",
        "class_map": {0: 1, 1: 2},  # bird->1, drone->2
    },
    # ─── B. Armor, Military Vehicles & Dismounts ─────────────────────────────
    "military_vehicles": {
        "name": "Military Vehicle Detection (MBTs, APCs, Artillery)",
        "dest": SOURCES_DIR / "military_vehicles",
        "download_method": "kaggle",
        "hf_repo": None,
        "git_repo": None,
        "kaggle_fallback": "simuletic/uav-and-aerial-view-battle-tank-detection-dataset",
        "kaggle_extras": [
            "caferfatihgltekin/air-defense-object-detection-dataset-yolov8",
            "sudipchakrabarty/kiit-mita",
        ],
        "class_map": {0: 6, 1: 7, 2: 8, 3: 9},  # tank, APC, truck, artillery
    },
    "visdrone": {
        "name": "VisDrone-DET (Aerial Dismounts & Civilian Vehicles)",
        "dest": SOURCES_DIR / "visdrone",
        "download_method": "kaggle",
        "hf_repo": None,
        "git_repo": None,
        "kaggle_fallback": "banuprasadb/visdrone-dataset",
        "class_map": {
            0: None, 1: 0, 2: 0, 3: 10, 4: 10,
            5: 10, 6: 8, 7: 10, 8: 10, 9: 10, 10: 10,
        },
    },
    "dota_v2": {
        "name": "DOTA-v2.0 (Aerial Reconnaissance - Aircraft, Vehicles, Installations)",
        "dest": SOURCES_DIR / "dota_v2",
        "download_method": "kaggle",
        "hf_repo": None,
        "git_repo": None,
        "kaggle_fallback": "chandlertimm/dota-data",
        "class_map": None,  # handled by text_class_map
        "text_class_map": {
            "plane": 5, "ship": 11, "storage-tank": 6, "harbor": 11,
            "large-vehicle": 8, "small-vehicle": 10, "helicopter": 4,
            "container-crane": 8, "airport": 5, "helipad": 4, "roundabout": 10,
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Download Engine
# ──────────────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": [], "total_images_downloaded": 0}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _download_via_kaggle(ref: str, dest: Path) -> bool:
    """Downloads a Kaggle dataset to dest folder."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f".done_{ref.replace('/', '_')}"
    if marker.exists():
        print(f"    [CACHED] {ref}")
        return True

    # Method 1: kagglehub
    try:
        import kagglehub
        cache_path = Path(kagglehub.dataset_download(ref))
        print(f"    [kagglehub] Downloaded {ref} -> {cache_path}")
        for item in cache_path.iterdir():
            target = dest / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        marker.write_text("OK", encoding="utf-8")
        return True
    except Exception as e1:
        print(f"    [kagglehub failed: {e1}] Trying CLI fallback...")

    # Method 2: kaggle CLI
    try:
        ret = subprocess.run(
            ["kaggle", "datasets", "download", "-d", ref, "-p", str(dest), "--unzip"],
            capture_output=True, text=True, timeout=600,
        )
        if ret.returncode == 0:
            marker.write_text("OK", encoding="utf-8")
            print(f"    [kaggle CLI] Downloaded {ref}")
            return True
        else:
            print(f"    [kaggle CLI stderr] {ret.stderr[:200]}")
    except Exception as e2:
        print(f"    [kaggle CLI failed: {e2}]")

    return False


def _download_via_git(url: str, dest: Path) -> bool:
    """Clones a git repository to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".done_git"
    if marker.exists():
        print(f"    [CACHED] {url}")
        return True

    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True, text=True, timeout=600,
        )
        marker.write_text("OK", encoding="utf-8")
        print(f"    [git clone] Cloned {url}")
        return True
    except Exception as e:
        print(f"    [git clone failed: {e}]")
        return False


def _download_via_huggingface(repo_id: str, dest: Path) -> bool:
    """Downloads from Hugging Face Datasets."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".done_hf"
    if marker.exists():
        print(f"    [CACHED] HF:{repo_id}")
        return True

    # Method 1: huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        marker.write_text("OK", encoding="utf-8")
        print(f"    [huggingface_hub] Downloaded {repo_id}")
        return True
    except Exception as e1:
        print(f"    [huggingface_hub failed: {e1}] Trying git-lfs fallback...")

    # Method 2: git clone from HF
    hf_url = f"https://huggingface.co/datasets/{repo_id}"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", hf_url, str(dest)],
            capture_output=True, text=True, timeout=1200,
        )
        marker.write_text("OK", encoding="utf-8")
        print(f"    [git clone HF] Cloned {hf_url}")
        return True
    except Exception as e2:
        print(f"    [HF git clone failed: {e2}]")

    return False


def download_single_dataset(key: str, cfg: dict) -> tuple[str, bool, float]:
    """Downloads a single dataset using configured method, with fallbacks."""
    t0 = time.time()
    dest = cfg["dest"]
    method = cfg.get("download_method", "kaggle")

    print(f"  [{key}] {cfg['name']}")
    print(f"    Method: {method} -> {dest}")

    success = False

    if method == "huggingface" and cfg.get("hf_repo"):
        success = _download_via_huggingface(cfg["hf_repo"], dest)
        if not success and cfg.get("git_repo"):
            success = _download_via_git(cfg["git_repo"], dest)

    elif method == "git" and cfg.get("git_repo"):
        success = _download_via_git(cfg["git_repo"], dest)

    elif method == "kaggle" and cfg.get("kaggle_fallback"):
        success = _download_via_kaggle(cfg["kaggle_fallback"], dest)
        # Also download extras (multiple Kaggle sources into same folder)
        for extra_ref in cfg.get("kaggle_extras", []):
            extra_dest = dest / extra_ref.replace("/", "_")
            _download_via_kaggle(extra_ref, extra_dest)

    elif method == "roboflow":
        # Try Kaggle fallback for Drone-vs-Bird
        if cfg.get("kaggle_fallback"):
            success = _download_via_kaggle(cfg["kaggle_fallback"], dest)
        if not success:
            print(f"    [INFO] Roboflow datasets require manual download or API key.")
            print(f"    To download from Roboflow CLI:")
            print(f"      pip install roboflow")
            print(f"      roboflow download {cfg.get('roboflow_workspace', '')} -f yolov8 -l {dest}")

    # Final fallback: try Kaggle if available
    if not success and cfg.get("kaggle_fallback") and method != "kaggle":
        print(f"    [Fallback] Trying Kaggle: {cfg['kaggle_fallback']}")
        success = _download_via_kaggle(cfg["kaggle_fallback"], dest)

    elapsed = time.time() - t0
    status = "OK" if success else "NEEDS_MANUAL"
    print(f"    [{status}] {key} in {elapsed:.1f}s\n")
    return key, success, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry
# ──────────────────────────────────────────────────────────────────────────────

def download_all_6(
    target_dataset: str | None = None,
    workers: int = 2,
    skip_tile: bool = False,
    tile_size: int = 640,
) -> None:
    """Downloads all 6 datasets into one unified folder, then optionally GPU-tiles."""
    # Create unified folder structure
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    for cfg in DATASETS.values():
        cfg["dest"].mkdir(parents=True, exist_ok=True)

    state = _load_state()
    completed = set(state.get("completed", []))

    targets = {target_dataset: DATASETS[target_dataset]} if target_dataset and target_dataset in DATASETS else DATASETS

    print("=" * 72)
    print("  TACYOLO: DOWNLOADING ALL 6 DATASETS INTO ONE UNIFIED FOLDER")
    print("=" * 72)
    print(f"  Unified Destination: {UNIFIED_DIR}")
    print(f"  Datasets to Download: {len(targets)}")
    print(f"  Workers: {workers}")
    print("=" * 72 + "\n")

    t0_all = time.time()

    # Download sequentially (safer for large datasets) or in parallel
    if workers <= 1:
        for key, cfg in targets.items():
            if key in completed:
                print(f"  [{key}] SKIPPED (already completed)\n")
                continue
            k, ok, elapsed = download_single_dataset(key, cfg)
            if ok:
                completed.add(k)
                state["completed"] = list(completed)
                _save_state(state)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for key, cfg in targets.items():
                if key in completed:
                    print(f"  [{key}] SKIPPED (already completed)\n")
                    continue
                futures[pool.submit(download_single_dataset, key, cfg)] = key

            for fut in as_completed(futures):
                k, ok, elapsed = fut.result()
                if ok:
                    completed.add(k)
                    state["completed"] = list(completed)
                    _save_state(state)

    total_elapsed = time.time() - t0_all

    # Count what we got
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    total_images = sum(1 for _ in SOURCES_DIR.rglob("*") if _.suffix.lower() in img_exts)
    state["total_images_downloaded"] = total_images
    _save_state(state)

    print("=" * 72)
    print(f"  DOWNLOAD COMPLETE: {total_images} images across 6 datasets")
    print(f"  Total Time: {total_elapsed / 60:.1f} minutes")
    print(f"  Saved To: {SOURCES_DIR}")
    print("=" * 72)

    # GPU Tile
    if not skip_tile and total_images > 0:
        print("\n  Now GPU-tiling all downloaded images into unified train/val/test...\n")
        try:
            from scripts.prepare_cuas_armor_gpu import run_pipeline
            generated = run_pipeline(
                tile_size=tile_size,
                dataset_key=target_dataset,
            )
            print(f"\n  GPU Tiling Done: {generated} tiles generated in {UNIFIED_DIR}")
        except Exception as e:
            print(f"  [WARN] GPU tiling encountered an error: {e}")
            print(f"  You can tile manually later: python scripts/prepare_cuas_armor_gpu.py")
    elif skip_tile:
        print("\n  --skip-tile active. To tile later, run:")
        print(f"    python scripts/prepare_cuas_armor_gpu.py")

    # Write master data.yaml
    import yaml
    yaml_path = UNIFIED_DIR / "data.yaml"
    class_names = [
        "person", "bird", "drone", "fixed_wing_uav", "helicopter",
        "airplane", "tank", "armored_vehicle", "military_truck",
        "artillery", "civilian_vehicle", "boat",
    ]
    yaml_content = {
        "path": str(UNIFIED_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")
    print(f"\n  Master data.yaml: {yaml_path}")
    print(f"  DONE!\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ALL 6 C-UAS & Military Armor Datasets Into ONE Unified Folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Download a single specific dataset instead of all 6",
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel download threads")
    parser.add_argument("--skip-tile", action="store_true", help="Download only, skip GPU tiling")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile resolution for GPU tiling")
    args = parser.parse_args()

    download_all_6(
        target_dataset=args.dataset,
        workers=args.workers,
        skip_tile=args.skip_tile,
        tile_size=args.tile_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

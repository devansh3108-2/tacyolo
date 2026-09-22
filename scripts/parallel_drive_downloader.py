"""High-speed parallel multi-threaded downloader from Kaggle directly into Google Drive.

Downloads all 4 pillars simultaneously (Drones, CCTV, OpenImages, Thermal)
using concurrent worker threads to maximize network bandwidth and save directly to Drive.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# The 4 Unified Pillars of Datasets
UNIFIED_PILLARS = {
    "Pillar_1_Drones_Aerial": [
        "banuprasadb/visdrone-dataset",
        "sshikamaru/drone-yolo-detection",
        "muki2003/yolo-drone-detection-dataset",
        "troykueh/multi-class-drone-detection-dataset-yolov8-ready",
        "caferfatihgltekin/air-defense-object-detection-dataset-yolov8",
        "simuletic/uav-and-aerial-view-battle-tank-detection-dataset",
        "sudipchakrabarty/kiit-mita",
    ],
    "Pillar_2_CCTV_Surveillance": [
        "solomonk/berkeley-deepdrive-bdd100k-yolo",
        "gaweshgomes/llvip-rgb-thermal-yolo-format",
        "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset",
        "niteshc7r/datasets-for-object-detection-night-and-thermal",
        "kausthubkannan/thermal-image-people-detection",
    ],
    "Pillar_3_Universal_OpenImages": [
        "ashishjangra27/open-images-dataset-v7-validation",
        "arashnic/open-images-dataset-v6-sample",
    ],
    "Pillar_4_Tactical_Thermal": [
        "sikdermdsaiful/thermal-images-for-human-detection",
        "mustafayngl/kaist-dataset-yolo26-early-fusion-preview-set",
        "animeshmahajan/thermal-image-dataset",
    ],
}


def download_single_dataset(ref: str, pillar_name: str, drive_raw_root: Path) -> dict:
    """Worker task: downloads a single dataset directly to Google Drive."""
    t0 = time.time()
    dest_dir = drive_raw_root / pillar_name / ref.replace("/", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)

    marker = dest_dir / ".download_complete"
    if marker.exists():
        print(f"⏩ [Already Downloaded] {ref} in {dest_dir.name}")
        return {"ref": ref, "status": "CACHED", "elapsed_sec": 0.0}

    print(f"⬇️  [Downloading] {ref} ({pillar_name}) -> {dest_dir}...")
    try:
        import kagglehub
        downloaded_cache = Path(kagglehub.dataset_download(ref))
        for item in downloaded_cache.iterdir():
            target = dest_dir / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        marker.write_text("OK", encoding="utf-8")
        elapsed = time.time() - t0
        print(f"✅ [Finished] {ref} in {elapsed:.1f}s")
        return {"ref": ref, "status": "SUCCESS", "elapsed_sec": elapsed}
    except Exception as e:
        # Fallback to CLI command
        try:
            cmd = f"kaggle datasets download -d {ref} -p '{dest_dir}' --unzip"
            ret = os.system(cmd)
            if ret == 0:
                marker.write_text("OK", encoding="utf-8")
                elapsed = time.time() - t0
                print(f"✅ [Finished CLI] {ref} in {elapsed:.1f}s")
                return {"ref": ref, "status": "SUCCESS", "elapsed_sec": elapsed}
        except Exception:
            pass
        print(f"❌ [Failed] {ref}: {e}")
        return {"ref": ref, "status": "FAILED", "error": str(e), "elapsed_sec": time.time() - t0}


def parallel_download_all(drive_root: str | Path = "/content/drive/MyDrive/TACYOLO", max_workers: int = 4) -> None:
    raw_root = Path(drive_root) / "raw_data"
    raw_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for pillar, datasets in UNIFIED_PILLARS.items():
        for ref in datasets:
            tasks.append((ref, pillar))

    total = len(tasks)
    print(f"🚀 Launching Parallel Direct-to-Drive Downloader for {total} datasets across 4 pillars!")
    print(f"📁 Target Google Drive Destination: {raw_root}")
    print(f"⚡ Simultaneous Workers: {max_workers}\n")

    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(download_single_dataset, ref, pillar, raw_root): ref
            for ref, pillar in tasks
        }
        for fut in as_completed(future_map):
            ref = future_map[fut]
            completed += 1
            res = fut.result()
            print(f"📊 Progress: [{completed}/{total}] {res['status']} | {ref}")

    total_time = time.time() - t_start
    print(f"\n🎉 All 4 Pillars downloaded simultaneously directly to Google Drive in {total_time / 60:.1f} minutes!")


def main() -> int:
    parser = argparse.ArgumentParser(description="TACYOLO Parallel Multi-Threaded Kaggle to Drive Downloader")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/TACYOLO", help="Google Drive Root Path")
    parser.add_argument("--workers", type=int, default=4, help="Number of simultaneous download threads (default: 4)")
    args = parser.parse_args()

    parallel_download_all(drive_root=args.drive_root, max_workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

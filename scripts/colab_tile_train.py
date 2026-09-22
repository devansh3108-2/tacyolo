"""TACYOLO Google Drive & Colab 3TB Tiled Streaming Pipeline.

Orchestrates multi-terabyte training directly on Google Drive:
1. Manages Drive directory hierarchy: working_testing, raw_data, tiled_batches, trained_weights.
2. Direct-to-Drive Kaggle dataset downloads (zero local workstation storage consumption).
3. High-resolution tactical image slicing/tiling with bounding-box recalculation.
4. Batch-wise training loop: train on batch -> checkpoint to Drive -> purge raw/tiles -> next batch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

# High-value defense and tactical datasets on Kaggle
DEFAULT_KAGGLE_TACTICAL_DATASETS = [
    {"ref": "sshikamaru/drone-yolo-detection", "tag": "drone_yolo", "cls_map": {0: 7}},
    {"ref": "muki2003/yolo-drone-detection-dataset", "tag": "drone_muki", "cls_map": {0: 7}},
    {"ref": "sudipchakrabarty/kiit-mita", "tag": "kiit_mita", "cls_map": {0: 8, 1: 7, 2: 0, 3: 6, 4: 10, 5: 9}},
    {"ref": "troykueh/multi-class-drone-detection-dataset-yolov8-ready", "tag": "drone_mcd", "cls_map": {0: 7}},
    {"ref": "caferfatihgltekin/air-defense-object-detection-dataset-yolov8", "tag": "air_defense", "cls_map": {0: 6, 1: 10, 2: 7, 3: 6}},
    {"ref": "simuletic/uav-and-aerial-view-battle-tank-detection-dataset", "tag": "tank_uav", "cls_map": {0: 8, 1: 9}},
    {"ref": "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset", "tag": "thermal_hit", "cls_map": {0: 0, 1: 2, 2: 1, 3: 5}},
    {"ref": "gaweshgomes/llvip-rgb-thermal-yolo-format", "tag": "thermal_llvip", "cls_map": {0: 0}},
    {"ref": "kausthubkannan/thermal-image-people-detection", "tag": "thermal_ppl", "cls_map": {0: 0}},
]

# UNIFIED 4-IN-1: Drones + CCTV Surveillance + Google Open Images + Foundation Vocabulary
UNIFIED_4_IN_1_DATASETS = [
    # --- Pillar 1: Drone & Overhead Aerial (DOTA / VisDrone / UAV Military) ---
    {"ref": "banuprasadb/visdrone-dataset", "tag": "p1_visdrone_aerial", "cls_map": {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 5, 8: 4, 9: 3}},
    {"ref": "sshikamaru/drone-yolo-detection", "tag": "p1_drone_yolo", "cls_map": {0: 7}},
    {"ref": "muki2003/yolo-drone-detection-dataset", "tag": "p1_drone_muki", "cls_map": {0: 7}},
    {"ref": "troykueh/multi-class-drone-detection-dataset-yolov8-ready", "tag": "p1_drone_mcd", "cls_map": {0: 7}},
    {"ref": "caferfatihgltekin/air-defense-object-detection-dataset-yolov8", "tag": "p1_air_defense", "cls_map": {0: 6, 1: 10, 2: 7, 3: 6}},
    {"ref": "simuletic/uav-and-aerial-view-battle-tank-detection-dataset", "tag": "p1_tank_uav", "cls_map": {0: 8, 1: 9}},
    {"ref": "sudipchakrabarty/kiit-mita", "tag": "p1_kiit_mita", "cls_map": {0: 8, 1: 7, 2: 0, 3: 6, 4: 10, 5: 9}},

    # --- Pillar 2: CCTV & Adverse Weather Surveillance (BDD100K / Night FLIR / Crowds) ---
    {"ref": "solomonk/berkeley-deepdrive-bdd100k-yolo", "tag": "p2_bdd100k_cctv", "cls_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}},
    {"ref": "gaweshgomes/llvip-rgb-thermal-yolo-format", "tag": "p2_thermal_llvip", "cls_map": {0: 0}},
    {"ref": "pandrii000/hituav-a-highaltitude-infrared-thermal-dataset", "tag": "p2_thermal_hit", "cls_map": {0: 0, 1: 2, 2: 1, 3: 5}},
    {"ref": "niteshc7r/datasets-for-object-detection-night-and-thermal", "tag": "p2_night_thermal", "cls_map": {0: 0, 1: 2, 2: 5}},
    {"ref": "kausthubkannan/thermal-image-people-detection", "tag": "p2_thermal_ppl", "cls_map": {0: 0}},

    # --- Pillar 3: Universal Object Detection (Google Open Images / Broad Classes) ---
    {"ref": "ashishjangra27/open-images-dataset-v7-validation", "tag": "p3_openimages_v7", "cls_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}},
    {"ref": "arashnic/open-images-dataset-v6-sample", "tag": "p3_openimages_sample", "cls_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}},
]


CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "bus",
    "truck", "airplane", "drone", "tank", "armored_car", "helicopter"
]


class DriveLayout:
    """Manages the prescribed folder structure inside Google Drive."""

    def __init__(self, drive_root: str | Path = "/content/drive/MyDrive/TACYOLO") -> None:
        self.root = Path(drive_root)
        self.working_testing = self.root / "working_testing"
        self.raw_data = self.root / "raw_data"
        self.tiled_batches = self.root / "tiled_batches"
        self.trained_weights = self.root / "trained_weights"
        self.calibration_data = self.root / "calibration_data"
        self.metrics_logs = self.root / "metrics_logs"
        self.state_file = self.root / "pipeline_state.json"

    def setup(self) -> None:
        for p in [
            self.root,
            self.working_testing,
            self.raw_data,
            self.tiled_batches,
            self.trained_weights,
            self.calibration_data,
            self.metrics_logs,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        print(f"[DriveLayout] Initialized Drive structure at: {self.root}")

    def load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"completed_batches": [], "current_batch_index": 0, "best_map50": 0.0}

    def save_state(self, state: dict) -> None:
        self.state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


class ImageTiler:
    """Slices large operational & aerial images into overlapping tiles with box re-projection."""

    def __init__(self, tile_size: int = 640, overlap: float = 0.2, min_box_visibility: float = 0.3) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.step = int(tile_size * (1.0 - overlap))
        self.min_box_visibility = min_box_visibility

    def tile_image_and_labels(
        self,
        image_path: Path,
        label_path: Path | None,
        out_img_dir: Path,
        out_lbl_dir: Path,
        class_map: dict[int, int] | None = None,
    ) -> int:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
        h, w = img.shape[:2]

        boxes: list[tuple[int, float, float, float, float]] = []
        if label_path and label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                    if class_map and cls_id in class_map:
                        cls_id = class_map[cls_id]
                    boxes.append((cls_id, xc, yc, bw, bh))

        # If already equal or smaller than tile size, copy directly
        if h <= self.tile_size and w <= self.tile_size:
            out_img = out_img_dir / image_path.name
            out_lbl = out_lbl_dir / (image_path.stem + ".txt")
            cv2.imwrite(str(out_img), img)
            if boxes:
                lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in boxes]
                out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return 1

        tiles_generated = 0
        y_starts = list(range(0, max(1, h - self.tile_size), self.step))
        if not y_starts or y_starts[-1] + self.tile_size < h:
            y_starts.append(max(0, h - self.tile_size))
        x_starts = list(range(0, max(1, w - self.tile_size), self.step))
        if not x_starts or x_starts[-1] + self.tile_size < w:
            x_starts.append(max(0, w - self.tile_size))

        for y0 in y_starts:
            y1 = min(h, y0 + self.tile_size)
            y0_adj = max(0, y1 - self.tile_size)
            for x0 in x_starts:
                x1 = min(w, x0 + self.tile_size)
                x0_adj = max(0, x1 - self.tile_size)

                tile = img[y0_adj:y1, x0_adj:x1]
                tile_name = f"{image_path.stem}_tile_{y0_adj}_{x0_adj}"

                tile_img_path = out_img_dir / f"{tile_name}.jpg"
                tile_lbl_path = out_lbl_dir / f"{tile_name}.txt"

                tile_boxes = []
                for cls_id, xc, yc, bw, bh in boxes:
                    # Convert normalized coords to image pixels
                    bx1 = (xc - bw / 2.0) * w
                    by1 = (yc - bh / 2.0) * h
                    bx2 = (xc + bw / 2.0) * w
                    by2 = (yc + bh / 2.0) * h

                    # Clip to current tile
                    ix1 = max(bx1, float(x0_adj))
                    iy1 = max(by1, float(y0_adj))
                    ix2 = min(bx2, float(x1))
                    iy2 = min(by2, float(y1))

                    orig_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                    inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                    if orig_area > 0 and (inter_area / orig_area) >= self.min_box_visibility:
                        # Re-project to tile coordinates
                        tw = float(x1 - x0_adj)
                        th = float(y1 - y0_adj)
                        txc = ((ix1 + ix2) / 2.0 - x0_adj) / tw
                        tyc = ((iy1 + iy2) / 2.0 - y0_adj) / th
                        tbw = (ix2 - ix1) / tw
                        tbh = (iy2 - iy1) / th
                        if tbw > 0.005 and tbh > 0.005:
                            tile_boxes.append((cls_id, txc, tyc, tbw, tbh))

                cv2.imwrite(str(tile_img_path), tile)
                if tile_boxes:
                    lines = [f"{c} {txc:.6f} {tyc:.6f} {tbw:.6f} {tbh:.6f}" for c, txc, tyc, tbw, tbh in tile_boxes]
                    tile_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                tiles_generated += 1

        return tiles_generated


def get_dir_size_gb(path: Path) -> float:
    """Calculate recursive directory size in gigabytes (GB)."""
    if not path.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_bytes / (1024 ** 3)


class ColabBatchStreamingTrainer:

    """Manages downloading Kaggle datasets directly to Drive, tiling, training, and purging."""

    def __init__(self, layout: DriveLayout, tile_size: int = 640) -> None:
        self.layout = layout
        self.tiler = ImageTiler(tile_size=tile_size)
        self.layout.setup()

    def download_kaggle_batch(self, ref: str, dest_dir: Path) -> Path:
        """Download Kaggle dataset directly to Google Drive destination."""
        print(f"[Kaggle-Direct] Downloading {ref} -> {dest_dir}...")
        try:
            import kagglehub
            raw_path = Path(kagglehub.dataset_download(ref))
            # Move or link into Drive raw_data folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in raw_path.iterdir():
                dst = dest_dir / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
            return dest_dir
        except Exception as e:
            # Fallback to standard kaggle CLI command
            print(f"[Kaggle-Direct] kagglehub failed ({e}), falling back to kaggle CLI...")
            os.system(f"kaggle datasets download -d {ref} -p '{dest_dir}' --unzip")
            return dest_dir

    def prepare_tiled_batch(self, raw_dir: Path, tag: str, cls_map: dict[int, int] | None = None) -> Path:
        """Tile images from raw_dir into tiled_batches."""
        batch_dir = self.layout.tiled_batches / tag
        train_img_dir = batch_dir / "images" / "train"
        train_lbl_dir = batch_dir / "labels" / "train"
        val_img_dir = batch_dir / "images" / "val"
        val_lbl_dir = batch_dir / "labels" / "val"

        for p in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Discover all image/label pairs in raw_dir
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in img_extensions]

        count = 0
        for i, img_path in enumerate(image_files):
            # Split roughly 90/10 train/val
            is_val = (i % 10 == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            # Attempt to find corresponding YOLO label
            possible_lbl_names = [img_path.stem + ".txt"]
            found_lbl: Path | None = None
            for parent in [img_path.parent, img_path.parents[1] / "labels" if len(img_path.parents) > 1 else None]:
                if parent and parent.exists():
                    cand = parent / (img_path.stem + ".txt")
                    if cand.exists():
                        found_lbl = cand
                        break

            n = self.tiler.tile_image_and_labels(img_path, found_lbl, target_img_dir, target_lbl_dir, cls_map)
            count += n

        print(f"[Tiler] Generated {count} tiles for batch {tag}")

        # Write data.yaml for this batch
        yaml_content = {
            "path": str(batch_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            "nc": len(CLASS_NAMES),
        }
        yaml_path = batch_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")
        return yaml_path

    def purge_batch_data(self, raw_dir: Path, tag: str) -> None:
        """Purge raw data and tiles to preserve storage across 3TB of training."""
        print(f"[Purge] Cleaning up raw download and tile buffer for {tag} to free Drive quota...")
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        tiled_dir = self.layout.tiled_batches / tag
        if tiled_dir.exists():
            shutil.rmtree(tiled_dir, ignore_errors=True)
        print(f"[Purge] Finished cleanup for {tag}.")

    def train_batch(self, data_yaml: Path, epochs: int = 15, batch_size: int = 16, resume_weights: str | None = None) -> Path:
        """Execute YOLO training on the active batch."""
        from ultralytics import YOLO

        model_path = resume_weights or "yolo11s.pt"
        best_pt = self.layout.trained_weights / "best.pt"
        if best_pt.exists() and resume_weights is None:
            model_path = str(best_pt)

        print(f"[Training] Training on {data_yaml} starting from {model_path}...")
        model = YOLO(model_path)
        project_dir = self.layout.working_testing / "runs"
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=self.tiler.tile_size,
            project=str(project_dir),
            name="batch_train",
            exist_ok=True,
            device="cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
        )

        # Copy resulting weights to trained_weights directory
        last_trained_best = project_dir / "batch_train" / "weights" / "best.pt"
        if last_trained_best.exists():
            shutil.copy2(last_trained_best, self.layout.trained_weights / "best.pt")
            shutil.copy2(last_trained_best, self.layout.trained_weights / "tactical_yolo11s.pt")
            print(f"[Training] Updated best weights at: {self.layout.trained_weights / 'best.pt'}")
        return self.layout.trained_weights / "best.pt"

    def run_pipeline(
        self,
        max_batches: int | None = None,
        epochs_per_batch: int = 5,
        manifest_path: str | Path | None = None,
        auto_discover: int | None = None,
        chunk_gb: float = 50.0,
    ) -> None:
        state = self.layout.load_state()

        if manifest_path and Path(manifest_path).exists():
            print(f"[Manifest] Loading batch list from manifest: {manifest_path}")
            datasets = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        elif auto_discover and auto_discover > 0:
            print(f"[Auto-Discover] Dynamically discovering {auto_discover} tactical datasets from Kaggle...")
            datasets = discover_kaggle_datasets(count=auto_discover)
            manifest_cache = self.layout.root / f"manifest_{len(datasets)}_batches.json"
            manifest_cache.write_text(json.dumps(datasets, indent=2), encoding="utf-8")
            print(f"[Auto-Discover] Discovered {len(datasets)} batches! Saved manifest to {manifest_cache}")
        else:
            datasets = DEFAULT_KAGGLE_TACTICAL_DATASETS

        total_to_run = len(datasets)
        ds_idx = state.get("current_batch_index", 0)
        chunk_idx = state.get("current_chunk_index", 1)

        print(f"[Pipeline] Ready to stream {total_to_run} datasets in ~{chunk_gb:.1f} GB chunks.")
        print(f"[Pipeline] Resuming from dataset index {ds_idx}, Chunk #{chunk_idx}.")

        while ds_idx < total_to_run:
            if max_batches is not None and ds_idx >= max_batches:
                break

            chunk_tag = f"chunk_{chunk_idx:03d}"
            chunk_raw_dir = self.layout.raw_data / chunk_tag
            chunk_raw_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=================================================================")
            print(f"📦 [Chunk {chunk_idx}] Staging up to {chunk_gb:.1f} GB of data directly to Drive...")
            print(f"=================================================================")

            downloaded_in_chunk = 0
            while ds_idx < total_to_run:
                if max_batches is not None and ds_idx >= max_batches:
                    break
                ds = datasets[ds_idx]
                ref = ds["ref"]
                tag = ds.get("tag", ref.replace("/", "_"))
                target_dest = chunk_raw_dir / tag

                current_gb = get_dir_size_gb(chunk_raw_dir)
                if current_gb >= chunk_gb and downloaded_in_chunk > 0:
                    print(f"🎯 [Chunk {chunk_idx}] Target reached ({current_gb:.2f} GB / {chunk_gb:.1f} GB). Commencing training!")
                    break

                print(f"[Chunk {chunk_idx}] Step {downloaded_in_chunk+1} | Staged: {current_gb:.2f} GB / {chunk_gb:.1f} GB | Downloading {ref}...")
                try:
                    self.download_kaggle_batch(ref, target_dest)
                    downloaded_in_chunk += 1
                except Exception as e:
                    print(f"[Warning] Failed to download {ref}: {e}. Skipping to next dataset...")
                ds_idx += 1

            chunk_final_gb = get_dir_size_gb(chunk_raw_dir)
            if chunk_final_gb == 0:
                print(f"[Chunk {chunk_idx}] No data staged. Ending pipeline.")
                break

            print(f"\n🚀 [Chunk {chunk_idx}] Tiling {chunk_final_gb:.2f} GB of imagery into 640x640 small-object tiles...")
            yaml_path = self.prepare_tiled_batch(chunk_raw_dir, chunk_tag)

            print(f"🏋️ [Chunk {chunk_idx}] Training YOLO model for {epochs_per_batch} epochs across {chunk_final_gb:.2f} GB chunk...")
            best_weights = self.train_batch(yaml_path, epochs=epochs_per_batch)
            print(f"✅ [Chunk {chunk_idx}] Checkpoint updated at: {best_weights}")

            # Crucial 50GB purge: Free Google Drive storage back to 0 before the next chunk
            print(f"🧹 [Chunk {chunk_idx}] Purging {chunk_final_gb:.2f} GB raw files & tiles from Drive to free space...")
            self.purge_batch_data(chunk_raw_dir, chunk_tag)
            print(f"✨ [Chunk {chunk_idx}] Drive space recycled! Ready for next {chunk_gb:.1f} GB chunk.\n")

            state["current_batch_index"] = ds_idx
            state["current_chunk_index"] = chunk_idx + 1
            state["completed_chunks"] = state.get("completed_chunks", 0) + 1
            self.layout.save_state(state)
            chunk_idx += 1

        print("\nAll requested 50 GB chunks trained and purged successfully!")



def discover_kaggle_datasets(count: int = 500) -> list[dict]:
    """Query Kaggle search API across all 4 tactical pillars to assemble up to `count` datasets.
    
    Pillars:
    1. Drone & Overhead Aerial (UAV, VisDrone, DOTA, Air Defense)
    2. CCTV & Adverse Weather Surveillance (BDD100K, Traffic, Crowds, Night)
    3. Universal Object Detection (OpenImages, COCO, Vehicles, Pedestrians)
    4. Thermal & Infrared (FLIR, HIT-UAV, LLVIP, Night Vision)
    """
    import urllib.parse
    import urllib.request

    # Curated base tactical datasets across all 4 pillars
    seen = set()
    found: list[dict] = []

    for ds in UNIFIED_4_IN_1_DATASETS:
        if ds["ref"] not in seen:
            seen.add(ds["ref"])
            found.append(ds)

    queries = [
        # Pillar 1: Drones & Aerial
        "drone yolo", "uav detection", "visdrone yolo", "aerial object detection",
        "air defense yolo", "military aircraft", "tank yolo", "anti drone",
        # Pillar 2: CCTV & Adverse Weather Surveillance
        "cctv object detection", "traffic surveillance yolo", "security camera dataset",
        "bdd100k yolo", "crowd detection yolo", "night surveillance", "pedestrian cctv",
        # Pillar 3: Universal Object Detection & Foundation
        "open images yolo", "universal object detection", "vehicle detection yolo",
        "military vehicle yolo", "tank detection yolo", "weapon detection yolo",
        # Pillar 4: Thermal & Infrared
        "thermal infrared yolo", "flir object detection", "thermal human detection",
        "llvip thermal", "infrared night vision", "military thermal vision"
    ]

    # Try official Kaggle API first
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        for q in queries:
            if len(found) >= count:
                break
            for page in range(1, 10):
                if len(found) >= count:
                    break
                try:
                    res = api.dataset_list(search=q, page=page)
                    if not res:
                        break
                    for item in res:
                        ref = getattr(item, "ref", None) or str(item)
                        if ref and ref not in seen:
                            seen.add(ref)
                            found.append({
                                "ref": ref,
                                "tag": ref.replace("/", "_"),
                                "cls_map": {0: 7},
                            })
                            if len(found) >= count:
                                break
                except Exception:
                    break
    except Exception as e:
        print(f"[discover_kaggle_datasets] KaggleApi unavailable ({e}), using web fallback...")
        for q in queries:
            if len(found) >= count:
                break
            for page in range(1, 8):
                if len(found) >= count:
                    break
                url = f"https://www.kaggle.com/api/v1/datasets/list?search={urllib.parse.quote(q)}&pageSize=50&page={page}"
                req = urllib.request.Request(url, headers={"User-Agent": "tacyolo"})
                try:
                    items = json.loads(urllib.request.urlopen(req, timeout=10).read())
                    if not items:
                        break
                    for item in items:
                        ref = item.get("ref")
                        if ref and ref not in seen:
                            seen.add(ref)
                            found.append({
                                "ref": ref,
                                "tag": ref.replace("/", "_"),
                                "cls_map": {0: 7},
                            })
                            if len(found) >= count:
                                break
                except Exception:
                    break

    print(f"🔍 Discovered {len(found)} tactical datasets across all 4 pillars!")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="TACYOLO Colab & Drive Batch Streaming Trainer")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/TACYOLO", help="Root path in Google Drive")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile resolution for small object detection")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per chunk (recommended 3-10)")
    parser.add_argument("--chunk-gb", type=float, default=50.0, help="Download buffer size in GB before training & purging (default: 50.0)")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to process in this run")
    parser.add_argument("--manifest", default=None, help="Path to JSON manifest listing custom batches")
    parser.add_argument("--auto-discover", type=int, default=None, help="Auto-discover N tactical datasets from Kaggle")
    parser.add_argument("--generate-manifest", type=str, default=None, help="Save discovered manifest to file and exit")
    args = parser.parse_args()

    if args.generate_manifest:
        target = args.auto_discover or 500
        print(f"Generating manifest of {target} Kaggle tactical datasets -> {args.generate_manifest}...")
        ds_list = discover_kaggle_datasets(count=target)
        out_p = Path(args.generate_manifest)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(ds_list, indent=2), encoding="utf-8")
        print(f"Manifest written with {len(ds_list)} batches to {out_p}")
        return 0

    layout = DriveLayout(args.drive_root)
    trainer = ColabBatchStreamingTrainer(layout=layout, tile_size=args.tile_size)
    trainer.run_pipeline(
        max_batches=args.max_batches,
        epochs_per_batch=args.epochs,
        manifest_path=args.manifest,
        auto_discover=args.auto_discover,
        chunk_gb=args.chunk_gb,
    )
    return 0



if __name__ == "__main__":
    sys.exit(main())


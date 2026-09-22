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

    def run_pipeline(self, max_batches: int | None = None, epochs_per_batch: int = 15) -> None:
        state = self.layout.load_state()
        datasets = DEFAULT_KAGGLE_TACTICAL_DATASETS
        start_idx = state.get("current_batch_index", 0)

        for i in range(start_idx, len(datasets)):
            if max_batches is not None and (i - start_idx) >= max_batches:
                break
            ds = datasets[i]
            ref = ds["ref"]
            tag = ds["tag"]
            cls_map = ds["cls_map"]

            print(f"\n==========================================")
            print(f"Starting Streaming Batch {i+1}/{len(datasets)}: {ref} [{tag}]")
            print(f"==========================================")

            raw_dir = self.layout.raw_data / tag
            self.download_kaggle_batch(ref, raw_dir)
            yaml_path = self.prepare_tiled_batch(raw_dir, tag, cls_map)

            best_weights = self.train_batch(yaml_path, epochs=epochs_per_batch)

            # Auto-purge raw and tile files
            self.purge_batch_data(raw_dir, tag)

            state["completed_batches"].append(tag)
            state["current_batch_index"] = i + 1
            self.layout.save_state(state)

        print("\nAll batches completed successfully!")


def main() -> int:
    parser = argparse.ArgumentParser(description="TACYOLO Colab & Drive Batch Streaming Trainer")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/TACYOLO", help="Root path in Google Drive")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile resolution for small object detection")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per batch")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to process in this run")
    args = parser.parse_args()

    layout = DriveLayout(args.drive_root)
    trainer = ColabBatchStreamingTrainer(layout=layout, tile_size=args.tile_size)
    trainer.run_pipeline(max_batches=args.max_batches, epochs_per_batch=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage 3: Auto-Dataset Partitioning.

Automatically partitions synthesized imagery and fused consensus labels into
a clean YOLO directory hierarchy (images/train, val, test and labels/train, val, test),
and generates a fully compliant `dataset.yaml` for training.
"""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from tacyolo.distillation.ensemble import FrameDetections
from tacyolo.distillation.fusion import FusedDetection, WeightedBoxFusionConsensus


@dataclass
class PartitionSummary:
    """Statistics on partitioned dataset."""
    total_images: int
    train_count: int
    val_count: int
    test_count: int
    dataset_yaml_path: Path
    output_dir: Path


class DatasetPartitioner:
    """Partitions images and fused consensus labels into train/val/test splits."""

    def __init__(
        self,
        output_dir: str | Path,
        train_ratio: float = 0.80,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        seed: int = 42,
        class_names: Sequence[str] | dict[int, str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.4f})")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        if class_names is None:
            self.class_names: dict[int, str] = {0: "target"}
        elif isinstance(class_names, dict):
            self.class_names = {int(k): str(v) for k, v in class_names.items()}
        else:
            self.class_names = {i: str(name) for i, name in enumerate(class_names)}

    def partition(
        self,
        frame_detections: dict[str, FrameDetections],
        fused_dataset: dict[str, list[FusedDetection]],
        copy_mode: str = "copy",  # "copy" or "symlink"
    ) -> PartitionSummary:
        """Splits data and creates directory tree + dataset.yaml."""
        img_train_dir = self.output_dir / "images" / "train"
        img_val_dir = self.output_dir / "images" / "val"
        img_test_dir = self.output_dir / "images" / "test"

        lbl_train_dir = self.output_dir / "labels" / "train"
        lbl_val_dir = self.output_dir / "labels" / "val"
        lbl_test_dir = self.output_dir / "labels" / "test"

        for p in [img_train_dir, img_val_dir, img_test_dir, lbl_train_dir, lbl_val_dir, lbl_test_dir]:
            p.mkdir(parents=True, exist_ok=True)

        frame_ids = sorted(list(frame_detections.keys()))
        rng = random.Random(self.seed)
        rng.shuffle(frame_ids)

        n_total = len(frame_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_fids = frame_ids[:n_train]
        val_fids = frame_ids[n_train : n_train + n_val]
        test_fids = frame_ids[n_train + n_val :]

        split_map: dict[str, tuple[Path, Path]] = {}
        for fid in train_fids:
            split_map[fid] = (img_train_dir, lbl_train_dir)
        for fid in val_fids:
            split_map[fid] = (img_val_dir, lbl_val_dir)
        for fid in test_fids:
            split_map[fid] = (img_test_dir, lbl_test_dir)

        for fid, (target_img_dir, target_lbl_dir) in split_map.items():
            fd = frame_detections[fid]
            src_img = fd.image_path
            dst_img = target_img_dir / src_img.name

            # Avoid name collisions
            if dst_img.exists() and dst_img.resolve() != src_img.resolve():
                dst_img = target_img_dir / f"{src_img.stem}_{abs(hash(fid)) % 10000}{src_img.suffix}"

            if copy_mode == "symlink":
                try:
                    if dst_img.exists():
                        dst_img.unlink()
                    dst_img.symlink_to(src_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)
            else:
                if not dst_img.exists() or dst_img.resolve() != src_img.resolve():
                    shutil.copy2(src_img, dst_img)

            # Write fused labels
            fused_dets = fused_dataset.get(fid, [])
            dst_lbl = target_lbl_dir / (dst_img.stem + ".txt")
            WeightedBoxFusionConsensus.save_yolo_labels(fused_dets, dst_lbl)

        # Generate dataset.yaml
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_content = {
            "path": str(self.output_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in sorted(self.class_names.items())},
            "nc": len(self.class_names),
        }
        yaml_path.write_text(yaml.dump(yaml_content, sort_keys=False), encoding="utf-8")

        summary = PartitionSummary(
            total_images=n_total,
            train_count=len(train_fids),
            val_count=len(val_fids),
            test_count=len(test_fids),
            dataset_yaml_path=yaml_path,
            output_dir=self.output_dir,
        )
        print(f"📦 Dataset Partitioning Complete: {summary.train_count} train, {summary.val_count} val, {summary.test_count} test -> {yaml_path}")
        return summary

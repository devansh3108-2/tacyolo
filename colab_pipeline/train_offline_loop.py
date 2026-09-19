"""Desktop-preferred offline train loop. No Kaggle.

Pick the next ready dataset folder under --data-root (default D:\\DET-YOLO_kaggle_datasets),
train a few epochs, archive/delete that folder, update state.json, next.

On missing labels: skip, write a replenish note, then try the next local
replenish_pool.txt folder that is already on disk.

Usage (from this folder, Windows):

    python train_offline_loop.py
    python train_offline_loop.py --data-root D:\\DET-YOLO_kaggle_datasets --epochs 8

Optional Colab: mount a Drive copy of the same folders and call pipeline_loop.run_cycle
with that path. Colab still must not use Kaggle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_loop import default_data_root, run_cycle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline DET-YOLO train from local folders (no Kaggle).")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help=r"Local datasets root (default D:\DET-YOLO_kaggle_datasets on Windows, or $DET_YOLO_DATA)",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-images", type=int, default=8000)
    p.add_argument("--max-datasets", type=int, default=None, help="Stop after this many successful trains.")
    p.add_argument(
        "--after-train",
        choices=("archive", "delete", "keep"),
        default="archive",
        help="archive moves the folder to <data-root>/_done",
    )
    p.add_argument("--no-skip-unlabeled", action="store_true", help="Train even if YOLO txt labels are missing.")
    p.add_argument("--no-replenish", action="store_true")
    p.add_argument("--no-push", action="store_true", help="Do not git commit/push state + weights.")
    args = p.parse_args(argv)

    print("train_offline_loop: Kaggle is NOT used. Training only from local folders.")
    print(f"data_root={args.data_root}")
    if not args.data_root.exists():
        print(
            f"data root does not exist: {args.data_root}\n"
            "Run download_desktop.py on Windows first, or copy folders here."
        )
        return 2

    st = run_cycle(
        args.repo,
        args.data_root,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        max_images=args.max_images,
        skip_if_no_labels=not args.no_skip_unlabeled,
        after_train=args.after_train,
        max_datasets=args.max_datasets,
        replenish=not args.no_replenish,
        push=not args.no_push,
    )
    print("done", len(st.get("done", [])), "failed", len(st.get("failed", [])), "skipped", len(st.get("skipped", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

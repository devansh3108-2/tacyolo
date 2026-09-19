"""Windows desktop parallel Kaggle download. Do NOT run on Colab.

Downloads download_queue.txt into --data-root (default D:\\DET-YOLO_kaggle_datasets)
using many workers. On download failure, immediately queues a replacement from
replenish_pool.txt. Aim for ~150 successful local folders, not 150 attempts.

Requires a local Kaggle token at %USERPROFILE%\\.kaggle\\kaggle.json (never commit it).

    pip install kagglehub
    python download_desktop.py --workers 8
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pipeline_loop import (
    default_data_root,
    find_dataset_dir,
    folder_has_images,
    load_queue,
    load_state,
    ref_to_dirname,
    save_state,
)


def _refuse_colab() -> None:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        raise SystemExit("download_desktop.py is for Windows desktop only. Colab must not use Kaggle.")
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return
    raise SystemExit("download_desktop.py is for Windows desktop only. Colab must not use Kaggle.")


def _copy_into(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if folder_has_images(dest):
            return
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)


def download_one(ref: str, data_root: Path) -> Path:
    existing = find_dataset_dir(data_root, ref)
    if existing is not None:
        print(f"already local: {ref} -> {existing}")
        return existing
    import kagglehub

    src = Path(kagglehub.dataset_download(ref))
    dest = data_root / ref_to_dirname(ref)
    _copy_into(src, dest)
    if not folder_has_images(dest):
        raise RuntimeError("download produced no images")
    print(f"saved: {ref} -> {dest}")
    return dest


def run_downloads(
    repo: Path,
    data_root: Path,
    *,
    workers: int = 8,
    replenish: bool = True,
) -> dict:
    _refuse_colab()
    data_root.mkdir(parents=True, exist_ok=True)
    state_path = repo / "state.json"
    st = load_state(state_path)
    st.setdefault("downloaded", [])
    st.setdefault("download_failed", [])
    done_train = set(st.get("done") or [])
    failed_train = set(st.get("failed") or [])
    skipped = set(st.get("skipped") or [])
    downloaded = set(st.get("downloaded") or [])
    download_failed = set(st.get("download_failed") or [])
    used = done_train | failed_train | skipped | downloaded | download_failed

    queue = load_queue(repo / "download_queue.txt")
    pool = [r for r in load_queue(repo / "replenish_pool.txt") if r not in set(queue)]
    pending = [r for r in queue if r not in used]
    pool_pending = [r for r in pool if r not in used]
    lock = threading.Lock()
    success_goal = len(queue)
    print(
        f"DESKTOP DOWNLOAD  root={data_root} workers={workers} "
        f"pending={len(pending)} pool={len(pool_pending)} goal={success_goal}"
    )

    def take_replenish() -> str | None:
        with lock:
            while pool_pending:
                r = pool_pending.pop(0)
                if r not in used:
                    used.add(r)
                    return r
        return None

    def mark(ref: str, ok: bool, err: str | None = None, replenish_for: str | None = None) -> None:
        with lock:
            if ok:
                if ref not in st.setdefault("downloaded", []):
                    st["downloaded"].append(ref)
                downloaded.add(ref)
            else:
                if ref not in st.setdefault("download_failed", []):
                    st["download_failed"].append(ref)
                download_failed.add(ref)
            rec = {
                "ref": ref,
                "ok": ok,
                "stage": "download",
                "at": datetime.now(timezone.utc).isoformat(),
            }
            if err:
                rec["error"] = err[:400]
            if replenish_for:
                rec["replenish_for"] = replenish_for
                if ok:
                    st.setdefault("replenished", []).append({"from": ref, "for": replenish_for, "stage": "download"})
            st.setdefault("history", []).append(rec)
            save_state(state_path, st)

    def job(ref: str, replenish_for: str | None = None) -> tuple[str, bool, str | None]:
        try:
            download_one(ref, data_root)
            mark(ref, True, replenish_for=replenish_for)
            return ref, True, None
        except Exception as e:
            print(f"FAIL download {ref}: {e}")
            mark(ref, False, str(e), replenish_for=replenish_for)
            return ref, False, str(e)

    # Phase 1: parallel main queue.
    extras: list[tuple[str, str]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(job, ref): ref for ref in pending}
            for fut in as_completed(futs):
                ref, ok, _ = fut.result()
                if (not ok) and replenish:
                    repl = take_replenish()
                    if repl:
                        print(f"REPLENISH after FAIL download {ref} -> {repl}")
                        extras.append((repl, ref))

    # Phase 2: parallel replenish for this wave, then more waves until goal or pool empty.
    while extras:
        wave = extras
        extras = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(job, repl, orig): (repl, orig) for repl, orig in wave}
            for fut in as_completed(futs):
                ref, ok, _ = fut.result()
                orig = futs[fut][1]
                if (not ok) and replenish:
                    nxt = take_replenish()
                    if nxt:
                        print(f"REPLENISH after FAIL download {ref} -> {nxt}")
                        extras.append((nxt, orig))

    n_ok = len(set(st.get("downloaded") or []))
    print(f"DOWNLOAD PASS COMPLETE  downloaded={n_ok} failed={len(st.get('download_failed') or [])}")
    if n_ok < success_goal:
        print(f"still short of {success_goal} folders; add slugs to replenish_pool.txt and re-run")
    return st


def main(argv: list[str] | None = None) -> int:
    _refuse_colab()
    p = argparse.ArgumentParser(description="Parallel Kaggle download on Windows desktop (not Colab).")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-replenish", action="store_true")
    args = p.parse_args(argv)
    run_downloads(args.repo, args.data_root, workers=args.workers, replenish=not args.no_replenish)
    return 0


if __name__ == "__main__":
    sys.exit(main())

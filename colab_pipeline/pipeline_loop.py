"""Offline DET-YOLO trainer. Never calls Kaggle / kagglehub.

Shared by the Windows desktop CLI (`train_offline_loop.py`) and the optional
Colab notebook (Drive-mounted copy of the same folders).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "done": [],
        "failed": [],
        "skipped": [],
        "downloaded": [],
        "download_failed": [],
        "current": None,
        "history": [],
        "replenished": [],
    }


def save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def load_queue(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def git_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return p


def default_data_root() -> Path:
    env = os.environ.get("DET_YOLO_DATA")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(r"D:\DET-YOLO_kaggle_datasets")
    return Path.home() / "DET-YOLO_kaggle_datasets"


def ref_to_dirname(ref: str) -> str:
    owner, _, slug = ref.partition("/")
    return f"{owner}--{slug}" if slug else owner


def folder_has_images(root: Path) -> bool:
    if not root.is_dir():
        return False
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            return True
    return False


def find_dataset_dir(data_root: Path, ref: str) -> Path | None:
    """Locate an already-local folder for owner/slug. No network."""
    data_root = Path(data_root)
    if not data_root.exists():
        return None
    owner, _, slug = ref.partition("/")
    candidates = [
        data_root / f"{owner}--{slug}",
        data_root / f"{owner}_{slug}",
        data_root / owner / slug,
        data_root / slug,
    ]
    for cand in candidates:
        if folder_has_images(cand):
            return cand
    return None


def find_pairs(root: Path):
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]
    pairs = []
    for img in images:
        lab = img.with_suffix(".txt")
        if not lab.exists():
            parts = list(img.parts)
            try:
                i = next(i for i, x in enumerate(parts) if x.lower() in ("images", "imgs", "img"))
                alt = Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt")
                if alt.exists():
                    lab = alt
            except StopIteration:
                pass
        pairs.append((img, lab if lab.exists() else None))
    return pairs


def build_yolo_set(raw_dir: Path, out_dir: Path, max_images: int = 8000):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_out = out_dir / "images" / "train"
    lab_out = out_dir / "labels" / "train"
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)
    pairs = find_pairs(raw_dir)
    labeled = [(i, l) for i, l in pairs if l is not None]
    unlabeled = [(i, l) for i, l in pairs if l is None]
    n = 0
    for img, lab in labeled + unlabeled:
        if n >= max_images:
            break
        try:
            dest = img_out / f"{n:06d}{img.suffix.lower()}"
            shutil.copy2(img, dest)
            if lab is not None:
                shutil.copy2(lab, lab_out / f"{n:06d}.txt")
            else:
                (lab_out / f"{n:06d}.txt").write_text("")
            n += 1
        except OSError:
            continue
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/train",
        "names": {i: f"c{i}" for i in range(80)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    n_lab = sum(1 for p in lab_out.glob("*.txt") if p.stat().st_size > 0)
    return yaml_path, n, n_lab


def delete_path(p: Path | None) -> None:
    if p is None or not p.exists():
        return
    if p.is_file():
        p.unlink(missing_ok=True)
    else:
        shutil.rmtree(p, ignore_errors=True)


def archive_or_delete(raw: Path, data_root: Path, after_train: str) -> None:
    if after_train == "keep":
        return
    if after_train == "delete":
        delete_path(raw)
        print("deleted local dataset folder")
        return
    dest_root = Path(data_root) / "_done"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / raw.name
    if dest.exists():
        delete_path(dest)
    try:
        shutil.move(str(raw), str(dest))
        print(f"archived -> {dest}")
    except OSError:
        delete_path(raw)
        print("archive failed; deleted local dataset folder")


def _seed_base_weights(weights: Path) -> None:
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() and weights.stat().st_size > 0:
        return
    print("seeding yolov8n -> weights/DET-YOLO.pt")
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    candidates: list[Path] = []
    for attr in ("ckpt_path", "pt_path"):
        val = getattr(model, attr, None)
        if val:
            candidates.append(Path(val))
    candidates.extend(
        [
            Path("yolov8n.pt"),
            Path.cwd() / "yolov8n.pt",
            Path("/content/yolov8n.pt"),
        ]
    )
    src = next((p for p in candidates if p.exists() and p.is_file()), None)
    if src is None:
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            downloaded = attempt_download_asset("yolov8n.pt")
            if downloaded:
                src = Path(downloaded)
        except Exception:
            src = None
    if src is not None and src.exists():
        shutil.copy2(src, weights)
        return
    save = getattr(model, "save", None)
    if callable(save):
        save(str(weights))
        if weights.exists() and weights.stat().st_size > 0:
            return
    raise FileNotFoundError("could not locate yolov8n.pt after download")


def git_push(repo: Path, message: str, force_add: list[Path] | None = None) -> None:
    root = git_root(repo)

    def run(cmd: list[str]) -> None:
        subprocess.check_call(cmd, cwd=str(root))

    run(["git", "add", "-A"])
    for extra in force_add or []:
        subprocess.run(["git", "add", "-f", "--", str(extra)], cwd=str(root), check=False)
    st = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if not (st.stdout or "").strip():
        print("nothing to push")
        return
    run(["git", "config", "user.email", "colab-pipeline@local"])
    run(["git", "config", "user.name", "DET-YOLO offline pipeline"])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", "HEAD"])


def _pop_unused(refs: list[str], used: set[str]) -> str | None:
    while refs:
        ref = refs.pop(0)
        if ref not in used:
            return ref
    return None


def _next_ready(refs: list[str], used: set[str], data_root: Path) -> tuple[str | None, Path | None]:
    while True:
        ref = _pop_unused(refs, used)
        if ref is None:
            return None, None
        found = find_dataset_dir(data_root, ref)
        if found is not None:
            return ref, found
        print(f"not on disk yet (desktop download still pending): {ref}")


def run_cycle(
    repo: Path,
    data_root: Path | None = None,
    *,
    epochs: int = 8,
    imgsz: int = 640,
    batch: int = 16,
    max_images: int = 8000,
    skip_if_no_labels: bool = True,
    after_train: str = "archive",
    max_datasets: int | None = None,
    replenish: bool = True,
    push: bool = True,
):
    """Train from folders already on disk. Does not call Kaggle."""
    from ultralytics import YOLO

    if after_train not in {"archive", "delete", "keep"}:
        raise ValueError("after_train must be archive|delete|keep")

    repo = Path(repo)
    data_root = Path(data_root) if data_root is not None else default_data_root()
    queue_path = repo / "download_queue.txt"
    pool_path = repo / "replenish_pool.txt"
    state_path = repo / "state.json"
    weights = repo / "weights" / "DET-YOLO.pt"
    work = Path(os.environ.get("DET_YOLO_WORK", str(data_root / "_work")))
    work.mkdir(parents=True, exist_ok=True)
    yolo_dir = work / "current_yolo"
    runs_dir = work / "runs"

    print(f"OFFLINE TRAIN  data_root={data_root}")
    print("Kaggle is not used in this process.")
    _seed_base_weights(weights)

    st = load_state(state_path)
    st.setdefault("replenished", [])
    st.setdefault("skipped", [])
    done = set(st.get("done") or [])
    failed = set(st.get("failed") or [])
    skipped = set(st.get("skipped") or [])
    used = set(done) | set(failed) | set(skipped)

    main_queue = load_queue(queue_path)
    main_set = set(main_queue)
    pending = [r for r in main_queue if r not in used]
    pool = [r for r in load_queue(pool_path) if r not in used and r not in main_set]

    success_goal = len(main_queue) if main_queue else 0
    if max_datasets is not None:
        success_goal = min(success_goal, len(done) + max_datasets)

    print(
        f"pending_queue={len(pending)} pool={len(pool)} "
        f"done={len(done)} failed={len(failed)} skipped={len(skipped)} "
        f"success_goal={success_goal}"
    )

    def cleanup_work() -> None:
        delete_path(yolo_dir)
        delete_path(runs_dir)

    def record_and_maybe_push(message: str, force_add: list[Path] | None = None) -> None:
        save_state(state_path, st)
        if not push:
            return
        try:
            git_push(repo, message, force_add=force_add)
        except Exception as pe:
            print("git push skipped:", pe)

    def attempt(ref: str, raw: Path, *, replenish_for: str | None = None) -> str:
        """Return 'ok', 'skip' (no labels / not usable), or 'fail'."""
        print("\n" + "=" * 60)
        tag = f"replenish for {replenish_for}" if replenish_for else "queue"
        print(f"{ref}  [{tag}]  folder={raw}  done={len(done)}/{success_goal}")
        st["current"] = ref
        save_state(state_path, st)
        t0 = time.time()
        try:
            yaml_path, n_img, n_lab = build_yolo_set(raw, yolo_dir, max_images)
            print(f"built images={n_img} labeled={n_lab}")
            if n_img < 1:
                raise RuntimeError("no images")
            if skip_if_no_labels and n_lab < 1:
                raise RuntimeError("no labels")
            model = YOLO(str(weights))
            model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=str(runs_dir),
                name="det_pipe",
                exist_ok=True,
                patience=5,
                verbose=True,
                plots=False,
                workers=0 if os.name == "nt" else 2,
            )
            best = runs_dir / "det_pipe" / "weights" / "best.pt"
            last = runs_dir / "det_pipe" / "weights" / "last.pt"
            src = best if best.exists() else last
            if not src.exists():
                raise RuntimeError("no weights after train")
            shutil.copy2(src, weights)
            st.setdefault("done", []).append(ref)
            rec = {
                "ref": ref,
                "ok": True,
                "images": n_img,
                "labeled": n_lab,
                "sec": round(time.time() - t0),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            if replenish_for:
                rec["replenish_for"] = replenish_for
                st.setdefault("replenished", []).append({"from": ref, "for": replenish_for})
            st.setdefault("history", []).append(rec)
            done.add(ref)
            used.add(ref)
            st["current"] = None
            msg = f"train: {ref} (+{n_img} imgs, {epochs} ep)"
            if replenish_for:
                msg = f"train: {ref} (replenish for {replenish_for}, +{n_img} imgs, {epochs} ep)"
            record_and_maybe_push(msg, force_add=[weights])
            return "ok"
        except Exception as e:
            err = str(e)[:400]
            no_labels = "no labels" in err.lower()
            print("SKIP" if no_labels else "FAIL", ref, e)
            if no_labels:
                print(
                    "REPLENISH NOTE: no YOLO labels in this folder. "
                    "Desktop download should pull the next unused slug from replenish_pool.txt "
                    "if it is not already on disk; this trainer will use that folder next."
                )
                st.setdefault("skipped", []).append(ref)
                skipped.add(ref)
                bucket = "skipped"
            else:
                st.setdefault("failed", []).append(ref)
                failed.add(ref)
                bucket = "failed"
            rec = {
                "ref": ref,
                "ok": False,
                "error": err,
                "at": datetime.now(timezone.utc).isoformat(),
            }
            if replenish_for:
                rec["replenish_for"] = replenish_for
            st.setdefault("history", []).append(rec)
            used.add(ref)
            st["current"] = None
            record_and_maybe_push(f"{'skip' if no_labels else 'fail'}: {ref}")
            return "skip" if no_labels else "fail"
        finally:
            try:
                archive_or_delete(raw, data_root, after_train)
            except Exception as ce:
                print("post-train folder cleanup skipped:", ce)
            cleanup_work()
            st["current"] = None
            save_state(state_path, st)

    def try_replenish(failed_orig: str) -> str:
        if not replenish:
            return "empty"
        while True:
            repl, rdir = _next_ready(pool, used, data_root)
            if repl is None:
                print(f"no local replenish folder after {failed_orig}")
                return "empty"
            print(f"REPLENISH after {failed_orig} -> {repl}")
            result = attempt(repl, rdir, replenish_for=failed_orig)
            if result == "ok":
                return "ok"

    while len(done) < success_goal:
        ref, raw = _next_ready(pending, used, data_root)
        replenish_for = None
        if ref is None and replenish:
            ref, raw = _next_ready(pool, used, data_root)
            replenish_for = None
        if ref is None or raw is None:
            print("no more ready local folders (download more on desktop, then re-run)")
            break
        result = attempt(ref, raw, replenish_for=replenish_for)
        if result != "ok":
            try_replenish(ref)

    print("PIPELINE PASS COMPLETE", f"done={len(done)} failed={len(failed)} skipped={len(skipped)}")
    return st

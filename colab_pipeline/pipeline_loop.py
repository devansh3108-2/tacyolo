"""DET-YOLO one-in/one-out trainer. Used by Colab notebook."""
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
    return {"done": [], "failed": [], "current": None, "history": []}


def save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")


def load_queue(path: Path) -> list[str]:
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


def _seed_base_weights(weights: Path) -> None:
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() and weights.stat().st_size > 0:
        return
    print("seeding yolov8n → weights/DET-YOLO.pt")
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
    run(["git", "config", "user.name", "DET-YOLO Colab Pipeline"])
    run(["git", "commit", "-m", message])
    run(["git", "push", "origin", "HEAD"])


def run_cycle(
    repo: Path,
    *,
    epochs: int = 8,
    imgsz: int = 640,
    batch: int = 16,
    max_images: int = 8000,
    skip_if_no_labels: bool = False,
    delete_after: bool = True,
    max_datasets: int | None = None,
):
    import kagglehub
    from ultralytics import YOLO

    repo = Path(repo)
    queue_path = repo / "download_queue.txt"
    state_path = repo / "state.json"
    weights = repo / "weights" / "DET-YOLO.pt"
    work = Path(os.environ.get("DET_YOLO_WORK", "/content/det_yolo_work"))
    work.mkdir(parents=True, exist_ok=True)
    yolo_dir = work / "current_yolo"
    runs_dir = work / "runs"

    _seed_base_weights(weights)

    st = load_state(state_path)
    done = set(st.get("done") or [])
    failed = set(st.get("failed") or [])
    pending = [r for r in load_queue(queue_path) if r not in done and r not in failed]
    if max_datasets is not None:
        pending = pending[:max_datasets]
    print(f"pending={len(pending)} done={len(done)} failed={len(failed)}")

    for idx, ref in enumerate(pending, 1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(pending)}] {ref}")
        st["current"] = ref
        save_state(state_path, st)
        raw = None
        t0 = time.time()
        try:
            print("download…")
            raw = Path(kagglehub.dataset_download(ref))
            yaml_path, n_img, n_lab = build_yolo_set(raw, yolo_dir, max_images)
            print(f"built images={n_img} labeled={n_lab}")
            if skip_if_no_labels and n_lab < 1:
                raise RuntimeError("no labels")
            if n_img < 1:
                raise RuntimeError("no images")
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
                workers=2,
            )
            best = runs_dir / "det_pipe" / "weights" / "best.pt"
            last = runs_dir / "det_pipe" / "weights" / "last.pt"
            src = best if best.exists() else last
            if not src.exists():
                raise RuntimeError("no weights after train")
            shutil.copy2(src, weights)
            st.setdefault("done", []).append(ref)
            st.setdefault("history", []).append(
                {
                    "ref": ref,
                    "ok": True,
                    "images": n_img,
                    "labeled": n_lab,
                    "sec": round(time.time() - t0),
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            done.add(ref)
            st["current"] = None
            save_state(state_path, st)
            git_push(repo, f"train: {ref} (+{n_img} imgs, {epochs} ep)", force_add=[weights])
        except Exception as e:
            print("FAIL", ref, e)
            st.setdefault("failed", []).append(ref)
            st.setdefault("history", []).append(
                {
                    "ref": ref,
                    "ok": False,
                    "error": str(e)[:400],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            failed.add(ref)
            st["current"] = None
            save_state(state_path, st)
            try:
                git_push(repo, f"fail: {ref}")
            except Exception as pe:
                print("push after fail skipped:", pe)
        finally:
            if delete_after:
                if raw is not None:
                    delete_path(raw)
                    # kagglehub often nests the extract under versions/N — drop the slug cache too
                    try:
                        if raw.parent.name.startswith("versions"):
                            delete_path(raw.parent.parent)
                    except Exception:
                        pass
                delete_path(yolo_dir)
                delete_path(runs_dir)
                print("deleted local dataset")
            st["current"] = None
            save_state(state_path, st)

    print("PIPELINE PASS COMPLETE")
    return st

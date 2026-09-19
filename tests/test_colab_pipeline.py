from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab_pipeline"))

from pipeline_loop import (  # noqa: E402
    build_yolo_set,
    find_pairs,
    git_root,
    load_queue,
    load_state,
    run_cycle,
    save_state,
)


def _tiny_jpeg(path: Path) -> None:
    # 1x1 JPEG, no extra deps
    path.write_bytes(
        bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000"
            "ffdb004300010101010101010101010101010101"
            "0101010101010101010101010101010101010101"
            "0101010101010101010101010101010101010101"
            "01010101010101ffc0000b080001000101011100"
            "ffc4001410010000000000000000000000000000"
            "00ffda00080001000100003f00fbffd9"
        )
    )


def test_load_and_save_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    assert load_state(path) == {"done": [], "failed": [], "current": None, "history": []}
    st = {"done": ["a/b"], "failed": [], "current": None, "history": []}
    save_state(path, st)
    assert load_state(path)["done"] == ["a/b"]


def test_load_queue_skips_comments_and_blank(tmp_path: Path) -> None:
    q = tmp_path / "download_queue.txt"
    q.write_text("# header\n\nowner/slug\n  other/ds  \n# skip\n", encoding="utf-8")
    assert load_queue(q) == ["owner/slug", "other/ds"]


def test_repo_queue_has_150_unique_refs() -> None:
    refs = load_queue(ROOT / "colab_pipeline" / "download_queue.txt")
    assert len(refs) == 150
    assert len(set(refs)) == 150
    assert all("/" in r and not r.startswith("#") for r in refs)


def test_find_pairs_images_and_labels_dirs(tmp_path: Path) -> None:
    img = tmp_path / "images" / "train" / "a.jpg"
    lab = tmp_path / "labels" / "train" / "a.txt"
    img.parent.mkdir(parents=True)
    lab.parent.mkdir(parents=True)
    _tiny_jpeg(img)
    lab.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][1] == lab


def test_build_yolo_set_caps_and_empty_unlabeled(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "imgs").mkdir(parents=True)
    for i in range(3):
        _tiny_jpeg(raw / "imgs" / f"{i}.jpg")
    (raw / "labels").mkdir()
    (raw / "labels" / "0.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    yaml_path, n, n_lab = build_yolo_set(raw, tmp_path / "yolo", max_images=2)
    assert yaml_path.exists()
    assert n == 2
    assert n_lab == 1
    labels = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2
    assert sum(1 for p in labels if p.stat().st_size > 0) == 1


def test_git_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "colab_pipeline"
    nested.mkdir()
    assert git_root(nested) == tmp_path.resolve()


class _DummyYOLO:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not Path(path).exists() and Path(path).name.endswith(".pt"):
            Path(path).write_bytes(b"seed-weights")

    def train(self, **kwargs):
        out = Path(kwargs["project"]) / kwargs["name"] / "weights"
        out.mkdir(parents=True, exist_ok=True)
        (out / "best.pt").write_bytes(b"best-weights")

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"seed-weights")


def test_run_cycle_skips_done_failed_and_pushes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "colab_pipeline"
    repo.mkdir()
    (repo / "download_queue.txt").write_text(
        "owner/done-ds\nowner/fail-ds\nowner/next-ds\n", encoding="utf-8"
    )
    save_state(
        repo / "state.json",
        {"done": ["owner/done-ds"], "failed": ["owner/fail-ds"], "current": None, "history": []},
    )
    (repo / "weights").mkdir()

    raw = tmp_path / "kaggle" / "owner" / "next-ds" / "versions" / "1"
    (raw / "images").mkdir(parents=True)
    _tiny_jpeg(raw / "images" / "x.jpg")
    (raw / "images" / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    work = tmp_path / "work"
    monkeypatch.setenv("DET_YOLO_WORK", str(work))
    monkeypatch.setitem(sys.modules, "kagglehub", type("K", (), {"dataset_download": staticmethod(lambda ref: str(raw))})())
    monkeypatch.setitem(sys.modules, "ultralytics", type("U", (), {"YOLO": _DummyYOLO})())

    pushes: list[str] = []

    def fake_push(repo_path: Path, message: str, force_add=None) -> None:
        pushes.append(message)

    monkeypatch.setattr("pipeline_loop.git_push", fake_push)
    monkeypatch.setattr("pipeline_loop.YOLO", _DummyYOLO, raising=False)

    # run_cycle imports YOLO/kagglehub inside the function
    import pipeline_loop as pl

    monkeypatch.setattr(pl, "_seed_base_weights", lambda weights: Path(weights).write_bytes(b"seed-weights"))

    st = run_cycle(repo, epochs=1, max_datasets=5, delete_after=True)
    assert "owner/done-ds" in st["done"]
    assert "owner/fail-ds" in st["failed"]
    assert "owner/next-ds" in st["done"]
    assert st["current"] is None
    assert any(m.startswith("train: owner/next-ds") for m in pushes)
    assert (repo / "weights" / "DET-YOLO.pt").read_bytes() == b"best-weights"
    assert not (work / "current_yolo").exists()
    assert json.loads((repo / "state.json").read_text(encoding="utf-8"))["done"][-1] == "owner/next-ds"

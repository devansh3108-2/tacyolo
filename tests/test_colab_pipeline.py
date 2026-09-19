from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "colab_pipeline"))

from pipeline_loop import (  # noqa: E402
    build_yolo_set,
    find_dataset_dir,
    find_pairs,
    git_root,
    load_queue,
    load_state,
    ref_to_dirname,
    run_cycle,
    save_state,
)


def _tiny_jpeg(path: Path) -> None:
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


def _labeled_ds(root: Path) -> Path:
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    _tiny_jpeg(img_dir / "x.jpg")
    (img_dir / "x.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    return root


def _unlabeled_ds(root: Path) -> Path:
    img_dir = root / "images"
    img_dir.mkdir(parents=True)
    _tiny_jpeg(img_dir / "x.jpg")
    return root


def test_load_and_save_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    empty = load_state(path)
    assert empty["done"] == []
    assert empty["skipped"] == []
    assert empty["downloaded"] == []
    st = {"done": ["a/b"], "failed": [], "current": None, "history": [], "replenished": []}
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


def test_replenish_pool_is_disjoint_alternate_slugs() -> None:
    queue = set(load_queue(ROOT / "colab_pipeline" / "download_queue.txt"))
    pool = load_queue(ROOT / "colab_pipeline" / "replenish_pool.txt")
    assert 50 <= len(pool) <= 100
    assert len(set(pool)) == len(pool)
    assert queue.isdisjoint(pool)
    assert all("/" in r for r in pool)


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


def test_git_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "colab_pipeline"
    nested.mkdir()
    assert git_root(nested) == tmp_path.resolve()


def test_find_dataset_dir_owner_slug_layouts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    dashed = _labeled_ds(data / "acme--visdrone")
    assert find_dataset_dir(data, "acme/visdrone") == dashed
    nested = _labeled_ds(data / "owner" / "slug")
    assert find_dataset_dir(data, "owner/slug") == nested
    assert find_dataset_dir(data, "missing/ds") is None
    assert ref_to_dirname("acme/visdrone") == "acme--visdrone"


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


def _install_train_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    work = tmp_path / "work"
    monkeypatch.setenv("DET_YOLO_WORK", str(work))
    monkeypatch.setitem(sys.modules, "ultralytics", type("U", (), {"YOLO": _DummyYOLO})())
    import pipeline_loop as pl

    monkeypatch.setattr(pl, "_seed_base_weights", lambda weights: Path(weights).write_bytes(b"seed-weights"))
    pushes: list[str] = []
    monkeypatch.setattr(pl, "git_push", lambda *a, **k: pushes.append(a[1] if len(a) > 1 else ""))
    return work, pushes


def test_run_cycle_trains_local_folder_not_kaggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "colab_pipeline"
    repo.mkdir()
    (repo / "download_queue.txt").write_text(
        "owner/done-ds\nowner/fail-ds\nowner/next-ds\nowner/not-ready\n", encoding="utf-8"
    )
    (repo / "replenish_pool.txt").write_text("owner/pool-a\n", encoding="utf-8")
    save_state(
        repo / "state.json",
        {"done": ["owner/done-ds"], "failed": ["owner/fail-ds"], "current": None, "history": []},
    )
    (repo / "weights").mkdir()
    data = tmp_path / "data"
    _labeled_ds(data / "owner--next-ds")
    work, pushes = _install_train_fakes(tmp_path, monkeypatch)

    st = run_cycle(
        repo,
        data,
        epochs=1,
        max_datasets=5,
        after_train="delete",
        push=True,
    )
    assert "owner/next-ds" in st["done"]
    assert "owner/not-ready" not in st["done"]
    assert "owner/not-ready" not in st["failed"]
    assert any(m.startswith("train: owner/next-ds") for m in pushes)
    assert (repo / "weights" / "DET-YOLO.pt").read_bytes() == b"best-weights"
    assert not (data / "owner--next-ds").exists()
    assert not (work / "current_yolo").exists()
    assert json.loads((repo / "state.json").read_text(encoding="utf-8"))["done"][-1] == "owner/next-ds"


def test_missing_labels_skip_and_replenish_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "colab_pipeline"
    repo.mkdir()
    (repo / "download_queue.txt").write_text("owner/no-lab\nowner/later\n", encoding="utf-8")
    (repo / "replenish_pool.txt").write_text("owner/good-repl\n", encoding="utf-8")
    save_state(repo / "state.json", load_state(repo / "state.json"))
    (repo / "weights").mkdir()
    data = tmp_path / "data"
    _unlabeled_ds(data / "owner--no-lab")
    _labeled_ds(data / "owner--good-repl")
    _labeled_ds(data / "owner--later")
    _install_train_fakes(tmp_path, monkeypatch)

    st = run_cycle(repo, data, epochs=1, max_datasets=1, after_train="keep", skip_if_no_labels=True)
    assert "owner/no-lab" in st["skipped"]
    assert "owner/good-repl" in st["done"]
    assert "owner/later" not in st["done"]
    assert st["replenished"] == [{"from": "owner/good-repl", "for": "owner/no-lab"}]


def test_train_fail_replenishes_until_local_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "colab_pipeline"
    repo.mkdir()
    (repo / "download_queue.txt").write_text("owner/bad-a\n", encoding="utf-8")
    (repo / "replenish_pool.txt").write_text("owner/bad-b\nowner/good-repl\n", encoding="utf-8")
    save_state(repo / "state.json", load_state(repo / "state.json"))
    (repo / "weights").mkdir()
    data = tmp_path / "data"
    _labeled_ds(data / "owner--bad-a")
    _labeled_ds(data / "owner--bad-b")
    _labeled_ds(data / "owner--good-repl")
    _install_train_fakes(tmp_path, monkeypatch)

    orig_train = _DummyYOLO.train

    def flaky_train(self, **kwargs):
        n = getattr(flaky_train, "n", 0)
        flaky_train.n = n + 1
        if n < 2:
            raise RuntimeError("cuda boom")
        return orig_train(self, **kwargs)

    monkeypatch.setattr(_DummyYOLO, "train", flaky_train)

    st = run_cycle(repo, data, epochs=1, max_datasets=1, after_train="keep")
    assert "owner/bad-a" in st["failed"]
    assert "owner/bad-b" in st["failed"]
    assert "owner/good-repl" in st["done"]


def test_download_desktop_refuses_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    import download_desktop as dd

    monkeypatch.setenv("COLAB_RELEASE_TAG", "1")
    with pytest.raises(SystemExit, match="Colab"):
        dd._refuse_colab()


def test_download_desktop_replenish_on_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import download_desktop as dd

    repo = tmp_path / "colab_pipeline"
    repo.mkdir()
    (repo / "download_queue.txt").write_text("owner/bad-a\nowner/ok-b\n", encoding="utf-8")
    (repo / "replenish_pool.txt").write_text("owner/good-repl\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()

    def fake_download(ref: str, data_root: Path) -> Path:
        if "bad" in ref:
            raise RuntimeError("kaggle 404")
        return _labeled_ds(data_root / ref_to_dirname(ref))

    monkeypatch.setattr(dd, "download_one", fake_download)
    monkeypatch.setattr(dd, "_refuse_colab", lambda: None)
    st = dd.run_downloads(repo, data, workers=2, replenish=True)
    assert "owner/bad-a" in st["download_failed"]
    assert "owner/ok-b" in st["downloaded"]
    assert "owner/good-repl" in st["downloaded"]

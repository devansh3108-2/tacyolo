from __future__ import annotations

import json
from pathlib import Path

from tacyolo.cli import main
from tacyolo.hdc.gallery import seed_default_gallery
from tacyolo.model import TacticalYOLO


def test_synthetic_track_writes_jsonl_and_skips_some_frames(tmp_path) -> None:
    gallery = seed_default_gallery(tmp_path / "gallery")
    log_path = tmp_path / "tracks.jsonl"
    model = TacticalYOLO(
        "dummy",
        hdc={"dim": 512, "gallery": str(gallery), "min_cosine": 0.05},
        detector={"weights": "dummy", "backend": "dummy"},
    )
    results = model.track(
        source="synthetic",
        radar="synthetic",
        max_frames=25,
        log=log_path,
        show=False,
        save=None,
    )
    assert len(results) == 25
    skipped = sum(1 for r in results if r.gate.skipped)
    inferred = sum(1 for r in results if r.yolo_ran)
    assert skipped >= 1
    assert inferred >= 1
    assert any(r.tracks for r in results[2:])
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 25
    assert "gate" in rows[0] and "tracks" in rows[0]
    assert "duty" in rows[0]
    assert any(r.duty_mode == "coast" for r in results)
    assert results[-1].yolo_frac < 1.0
    assert results[0].extras.get("tracker") == "midus"


def test_predict_does_not_require_radar(tmp_path) -> None:
    seed_default_gallery(tmp_path / "gallery")
    model = TacticalYOLO("dummy", hdc={"gallery": str(tmp_path / "gallery"), "dim": 256})
    results = model.predict(source="synthetic", radar="off", max_frames=6, log=False, show=False)
    assert len(results) == 6
    assert all(r.rf is None for r in results)


def test_track_from_video_file(tmp_path) -> None:
    import cv2

    from tacyolo.sensors.sim import SceneSimulator

    gallery = seed_default_gallery(tmp_path / "gallery")
    sim = SceneSimulator(n_targets=2)
    frames = []
    for _ in range(16):
        sim.step()
        frames.append(sim.render_frame())
    video_path = tmp_path / "clip.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()

    model = TacticalYOLO(
        "dummy",
        hdc={"dim": 256, "gallery": str(gallery), "min_cosine": 0.05},
        detector={"weights": "dummy", "backend": "dummy"},
    )
    results = model.track(
        source=str(video_path),
        radar="synthetic",
        max_frames=12,
        log=False,
        show=False,
    )
    assert len(results) == 12
    assert any(r.rf is not None for r in results)
    assert any(r.yolo_ran for r in results)


def test_cli_track(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    seed_default_gallery(tmp_path / "data" / "gallery")
    code = main(
        [
            "track",
            "--source",
            "synthetic",
            "--radar",
            "synthetic",
            "--weights",
            "dummy",
            "--max-frames",
            "8",
            "--log",
            str(tmp_path / "t.jsonl"),
        ]
    )
    assert code == 0
    assert Path(tmp_path / "t.jsonl").exists()

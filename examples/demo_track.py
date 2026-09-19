"""Run a 90-frame synthetic track demo and write overlay + JSONL."""

from __future__ import annotations

from pathlib import Path

from tacyolo import TacticalYOLO
from tacyolo.hdc.gallery import seed_default_gallery


def main() -> None:
    seed_default_gallery()
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TacticalYOLO("dummy")
    results = model.track(
        source="synthetic",
        radar="synthetic",
        save=out_dir / "demo.mp4",
        max_frames=90,
        log=out_dir / "tracks.jsonl",
        show=False,
    )
    skipped = sum(1 for r in results if r.gate.skipped)
    inferred = sum(1 for r in results if r.yolo_ran)
    n_tracks = max((len(r.tracks) for r in results), default=0)
    print(f"demo frames={len(results)} skipped={skipped} inferred={inferred} max_tracks={n_tracks}")
    print(f"wrote {out_dir / 'demo.mp4'} and {out_dir / 'tracks.jsonl'}")


if __name__ == "__main__":
    main()

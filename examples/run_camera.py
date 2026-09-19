"""Track the webcam with the best available detector (military YOLO if present)."""

from __future__ import annotations

from pathlib import Path

from tacyolo import TacticalYOLO


def main() -> None:
    out = Path("runs")
    out.mkdir(parents=True, exist_ok=True)
    model = TacticalYOLO("auto")
    print(f"backend={model.detector.name} device={getattr(model.detector.backend, 'device', '?')}")
    names = getattr(model.detector.backend, "names", {})
    if names:
        print("classes:", ", ".join(str(v) for v in list(names.values())[:12]))
    model.track(
        source=0,
        radar="synthetic",
        show=True,
        save=out / "webcam.mp4",
        log=out / "webcam.jsonl",
    )


if __name__ == "__main__":
    main()

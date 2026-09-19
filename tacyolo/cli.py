"""YOLO-style CLI: tacyolo predict | track | export."""

from __future__ import annotations

import argparse
from pathlib import Path

from tacyolo.hdc.gallery import seed_default_gallery
from tacyolo.model import TacticalYOLO


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="synthetic", help="video path, camera index, image folder, or 'synthetic'")
    p.add_argument("--radar", default="synthetic", help="synthetic | off | path/to/rd.npy")
    p.add_argument("--weights", default="auto", help="auto | dummy | military | world | path/to.pt")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", default=None, help="optional output mp4 path")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--log", default="runs/tracks.jsonl")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--backend", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tacyolo", description="Gated YOLO fusion: detect, HDC match, EKF/MIDUS track")
    sub = parser.add_subparsers(dest="cmd", required=True)

    predict = sub.add_parser("predict", help="Gated detection without a tracker")
    _add_common(predict)
    predict.set_defaults(radar="off")

    track = sub.add_parser("track", help="Gated detect + HDC + MIDUS or EKF tracker")
    _add_common(track)
    track.add_argument("--tracker", default=None, choices=["midus", "ekf", "hungarian"], help="ID tracker (default midus)")

    export = sub.add_parser("export", help="Export detector (ONNX now; TensorRT later)")
    export.add_argument("--weights", default="auto")
    export.add_argument("--format", default="onnx")
    export.add_argument("--int8", action="store_true")
    export.add_argument("--out-dir", default="runs")

    fetch = sub.add_parser("fetch", help="Download YOLO11n, military YOLOv8s, and YOLO-World weights")
    fetch.set_defaults(cmd="fetch")

    train = sub.add_parser("train", help="Fine-tune YOLO11s part-by-part on drone/tank/aircraft data")
    train.add_argument("--epochs", type=int, default=0, help="0 uses curriculum stage lengths")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default="auto")
    train.add_argument("--model", default=None)
    train.add_argument("--prepare-only", action="store_true")
    train.add_argument("--curriculum", action="store_true")
    train.add_argument("--sources", default="")
    return parser


def _model_from_args(args: argparse.Namespace) -> TacticalYOLO:
    overrides = {}
    if getattr(args, "conf", None) is not None:
        overrides["conf"] = args.conf
    if getattr(args, "backend", None):
        overrides["backend"] = args.backend
    if getattr(args, "tracker", None):
        overrides["tracker"] = {"backend": args.tracker}
    return TacticalYOLO(weights=args.weights, config=args.config, **overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    seed_default_gallery()

    if args.cmd == "train":
        from tacyolo.train import main as train_main

        argv = ["--epochs", str(args.epochs), "--imgsz", str(args.imgsz), "--batch", str(args.batch), "--device", str(args.device)]
        if args.model:
            argv.extend(["--model", args.model])
        if args.prepare_only:
            argv.append("--prepare-only")
        if args.curriculum:
            argv.append("--curriculum")
        if args.sources:
            argv.extend(["--sources", args.sources])
        return train_main(argv)

    if args.cmd == "fetch":
        from tacyolo.fetch_models import main as fetch_main

        return fetch_main()

    if args.cmd == "export":
        model = TacticalYOLO(weights=args.weights, backend="pytorch" if str(args.weights).endswith(".pt") else "auto")
        path = model.export(format=args.format, int8=args.int8, out_dir=args.out_dir)
        print(path)
        return 0

    model = _model_from_args(args)
    kwargs = dict(
        source=args.source,
        radar=args.radar,
        show=args.show,
        save=args.save,
        max_frames=args.max_frames,
        log=args.log,
    )
    if args.cmd == "predict":
        results = model.predict(**kwargs)
    else:
        results = model.track(**kwargs)
    skipped = sum(1 for r in results if r.gate.skipped)
    ran = sum(1 for r in results if r.yolo_ran)
    print(
        f"frames={len(results)} skipped={skipped} inferred={ran} "
        f"backend={model.detector.name} tracker={model.tracker.backend}"
    )
    if args.save:
        print(f"saved={args.save}")
    if args.log:
        print(f"log={args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

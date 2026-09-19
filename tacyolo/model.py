from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from tacyolo.config import Config
from tacyolo.hdc.memory import ItemMemory
from tacyolo.optical.detector import Detector
from tacyolo.optical.duty import DutyCycle
from tacyolo.optical.motion_gate import MotionGate
from tacyolo.rf.pipeline import RFPipeline
from tacyolo.sensors.optical import OpticalSource
from tacyolo.sensors.radar import make_radar
from tacyolo.sensors.sim import SceneSimulator
from tacyolo.track.tracker import MultiObjectTracker
from tacyolo.types import Detection, FrameResult
from tacyolo.viz import annotate, write_video


class TacticalYOLO:
    """Ultralytics-style wrapper: predict, track, export, with gated fusion."""

    def __init__(
        self,
        weights: str = "auto",
        config: str | Path | Config | None = None,
        **overrides: Any,
    ) -> None:
        if isinstance(config, Config):
            self.cfg = config
        else:
            nested = {}
            if weights:
                nested["detector"] = {"weights": weights}
            if overrides:
                # allow TacticalYOLO(..., conf=0.4) style plus nested dicts
                det = nested.setdefault("detector", {})
                for key in ("conf", "iou", "imgsz", "backend", "device", "on_crops"):
                    if key in overrides:
                        det[key] = overrides.pop(key)
                tracker = overrides.pop("tracker", None)
                if isinstance(tracker, str):
                    nested.setdefault("tracker", {})["backend"] = tracker
                elif isinstance(tracker, dict):
                    nested["tracker"] = tracker
                nested.update(overrides)
            self.cfg = Config.load(config, nested if nested else None)

        gate_cfg = self.cfg.section("gate")
        det_cfg = self.cfg.section("detector")
        rf_cfg = self.cfg.section("rf")
        hdc_cfg = self.cfg.section("hdc")
        ekf_cfg = {**self.cfg.section("ekf"), **self.cfg.section("tracker")}
        runtime_cfg = self.cfg.section("runtime")

        weights = str(det_cfg.get("weights", weights or "auto"))
        world_classes = det_cfg.get("world_classes")
        if isinstance(world_classes, dict):
            world_classes = [str(v) for _, v in sorted(world_classes.items())]
        self.gate = MotionGate(**{k: gate_cfg[k] for k in gate_cfg})
        self.detector = Detector(
            weights=weights,
            backend=str(det_cfg.get("backend", "auto")),
            imgsz=int(det_cfg.get("imgsz", 640)),
            conf=float(det_cfg.get("conf", 0.25)),
            iou=float(det_cfg.get("iou", 0.45)),
            device=str(det_cfg.get("device", "auto")),
            on_crops=bool(det_cfg.get("on_crops", False)),
            providers=list(runtime_cfg.get("providers", ["CPUExecutionProvider"])),
            world_classes=list(world_classes) if world_classes else None,
        )
        self.rf = RFPipeline(
            cs_lambda=float(rf_cfg.get("cs_lambda", 0.02)),
            cs_iters=int(rf_cfg.get("cs_iters", 40)),
            cs_sample_frac=float(rf_cfg.get("cs_sample_frac", 0.35)),
            gauss_sigmas=list(rf_cfg.get("gauss_sigmas", [1.0, 2.0, 4.0])),
            clutter_rank=int(rf_cfg.get("clutter_rank", 3)),
            tda_eps=float(rf_cfg.get("tda_eps", 3.0)),
            energy_gate=float(rf_cfg.get("energy_gate", 2.5)),
        )
        self.hdc = ItemMemory(
            dim=int(hdc_cfg.get("dim", 10000)),
            seed=int(hdc_cfg.get("seed", 7)),
            min_cosine=float(hdc_cfg.get("min_cosine", 0.12)),
            embed_fn=getattr(self.detector.backend, "embed_crop", None),
        )
        gallery = hdc_cfg.get("gallery", "data/gallery")
        self.hdc.fit_gallery(gallery)
        self.tracker = MultiObjectTracker(
            dt=float(ekf_cfg.get("dt", 0.033)),
            process_var=float(ekf_cfg.get("process_var", 25.0)),
            meas_var=float(ekf_cfg.get("meas_var", 4.0)),
            radar_var=float(ekf_cfg.get("radar_var", 16.0)),
            max_age=int(ekf_cfg.get("max_age", 15)),
            min_hits=int(ekf_cfg.get("min_hits", 2)),
            iou_weight=float(ekf_cfg.get("iou_weight", 0.55)),
            mahalanobis_gate=float(ekf_cfg.get("mahalanobis_gate", 9.21)),
            backend=str(ekf_cfg.get("backend", "midus")),
            high_thresh=float(ekf_cfg.get("high_thresh", 0.5)),
            low_thresh=float(ekf_cfg.get("low_thresh", 0.1)),
        )
        self._rf_energy_gate = float(rf_cfg.get("energy_gate", 2.5))
        duty_cfg = self.cfg.section("duty")
        self.duty = DutyCycle(**{k: duty_cfg[k] for k in duty_cfg}) if duty_cfg else DutyCycle()

    def _open_sources(
        self,
        source: Any,
        radar: str = "synthetic",
        max_frames: int | None = None,
    ) -> tuple[OpticalSource, Any, SceneSimulator | None]:
        radar_cfg = self.cfg.section("radar")
        scene = None
        if source is None or (isinstance(source, str) and source in {"synthetic", "sim"}):
            scene = SceneSimulator(n_targets=int(radar_cfg.get("n_targets", 2)))
            optical = OpticalSource(scene=scene)
        else:
            optical = OpticalSource(source)
            if getattr(optical, "scene", None) is not None:
                scene = optical.scene
        radar_src = None
        if radar not in ("off", "none", None, False):
            kwargs = {
                "n_range": int(self.cfg.get("rf", "n_range", default=128)),
                "n_doppler": int(self.cfg.get("rf", "n_doppler", default=64)),
                "snr_db": float(radar_cfg.get("snr_db", 12.0)),
                "clutter_power": float(radar_cfg.get("clutter_power", 8.0)),
                "n_targets": int(radar_cfg.get("n_targets", 2)),
            }
            mode = radar if radar not in ("synthetic", "sim", True) else "synthetic"
            radar_src = make_radar(mode=str(mode), scene=scene, **kwargs)
        return optical, radar_src, scene

    def _step(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        radar_src: Any,
        use_tracker: bool,
    ) -> FrameResult:
        gate = self.gate(frame)
        rf_result = None
        radar_xy: list[tuple[float, float]] = []
        radar_packet = None
        if radar_src is not None:
            radar_packet = radar_src.next()
            rd = radar_packet["rd_map"]
            cheap = self.rf.cheap_energy(rd)
            full = gate.triggered or cheap >= self._rf_energy_gate
            rf_result = self.rf.process(rd, full=full)
            if scene_xy := _radar_to_image(radar_packet.get("truth") or []):
                radar_xy = scene_xy

        rf_hot = rf_result is not None and rf_result.energy >= self._rf_energy_gate
        prev_tracks = [t.as_state(self.tracker.min_hits) for t in self.tracker.tracks]
        decision = self.duty.decide(
            gate=gate,
            frame_shape=frame.shape,
            tracks=prev_tracks,
            rf_hot=rf_hot,
            use_tracker=use_tracker,
        )

        detections: list[Detection] = []
        yolo_ran = False
        if decision.run_detector:
            detections = self.detector.infer(
                frame,
                rois=decision.rois or None,
                force_crops=decision.force_crops,
            )
            yolo_ran = True
            if self.detector.name == "dummy" and not detections and decision.rois:
                detections = self.detector.infer(frame, rois=decision.rois, force_crops=False)
            rf_feat = rf_result.features if rf_result is not None else None
            for det in detections:
                if det.crop is None:
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    det.crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if not self.duty.hdc_needed(det.conf):
                    continue
                match = self.hdc.match(det.crop, rf_feat)
                if match.class_name:
                    det.hdc_class = match.class_name
                    det.hdc_score = match.score
                    det.hdc_margin = match.margin

        raw_tracks: list = []
        if use_tracker:
            self.tracker.predict()
            if detections or self.tracker.tracks:
                raw_tracks = self.tracker.update(detections, radar_xy if radar_xy else None)
        tracks = self.duty.publish(raw_tracks)
        tentative = [t for t in raw_tracks if all(t.track_id != p.track_id for p in tracks)]

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            frame=frame,
            gate=gate,
            rf=rf_result,
            detections=detections,
            tracks=tracks,
            yolo_ran=yolo_ran,
            duty_mode=decision.mode,
            duty_reason=decision.reason,
            yolo_frac=self.duty.yolo_frac,
            extras={
                "radar_truth": (radar_packet or {}).get("truth", []),
                "tentative": tentative,
                "tracker": self.tracker.backend,
                "duty": {
                    "mode": decision.mode,
                    "reason": decision.reason,
                    "yolo_frac": self.duty.yolo_frac,
                    "published": len(tracks),
                },
            },
        )

    def stream(
        self,
        source: Any = "synthetic",
        radar: str = "synthetic",
        max_frames: int | None = None,
        track: bool = True,
    ) -> Iterator[FrameResult]:
        optical, radar_src, _ = self._open_sources(source, radar=radar, max_frames=max_frames)
        fps = optical.fps or 30.0
        i = 0
        try:
            while True:
                if max_frames is not None and i >= max_frames:
                    break
                frame = optical.read()
                if frame is None:
                    break
                yield self._step(frame, i, i / fps, radar_src, use_tracker=track)
                i += 1
        finally:
            optical.close()
            if radar_src is not None:
                radar_src.close()

    def predict(
        self,
        source: Any = "synthetic",
        radar: str = "off",
        show: bool = False,
        save: str | Path | None = None,
        max_frames: int | None = None,
        log: str | Path | bool | None = None,
    ) -> list[FrameResult]:
        return list(
            self._run(source, radar=radar, show=show, save=save, max_frames=max_frames, log=log, track=False)
        )

    def track(
        self,
        source: Any = "synthetic",
        radar: str = "synthetic",
        show: bool = False,
        save: str | Path | None = None,
        max_frames: int | None = None,
        log: str | Path | bool | None = None,
    ) -> list[FrameResult]:
        return list(self._run(source, radar=radar, show=show, save=save, max_frames=max_frames, log=log, track=True))

    def _run(
        self,
        source: Any,
        radar: str,
        show: bool,
        save: str | Path | None,
        max_frames: int | None,
        log: str | Path | bool | None,
        track: bool,
    ) -> Iterator[FrameResult]:
        viz_cfg = self.cfg.section("viz")
        log_path = self._resolve_log(log)
        writer = log_path.open("w", encoding="utf-8") if log_path else None
        rendered: list[np.ndarray] = []
        try:
            for result in self.stream(source=source, radar=radar, max_frames=max_frames, track=track):
                vis = annotate(
                    result,
                    show_gate=bool(viz_cfg.get("show_gate", True)),
                    rd_inset=bool(viz_cfg.get("rd_inset", True)),
                )
                if writer:
                    writer.write(json.dumps(self._log_row(result)) + "\n")
                if show:
                    import cv2

                    cv2.imshow("tacyolo", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        yield result
                        break
                if save:
                    rendered.append(vis)
                yield result
        finally:
            if writer:
                writer.close()
            if show:
                import cv2

                cv2.destroyAllWindows()
            if save and rendered:
                write_video(save, rendered, fps=30.0)

    def export(self, format: str = "onnx", int8: bool = False, out_dir: str | Path = "runs") -> Path:
        return self.detector.export(format, int8=int8, out_dir=out_dir)

    def _resolve_log(self, log: str | Path | bool | None) -> Path | None:
        if log is False:
            return None
        if log is True or log is None:
            path = self.cfg.get("log", "jsonl", default=None)
            if log is None:
                return Path(path) if path else None
            path = path or "runs/tracks.jsonl"
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        p = Path(log)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _log_row(self, result: FrameResult) -> dict[str, Any]:
        return {
            "frame": result.frame_index,
            "timestamp": result.timestamp,
            "gate": {
                "skipped": result.gate.skipped,
                "triggered": result.gate.triggered,
                "energy": result.gate.energy,
                "elapsed_ms": result.gate.elapsed_ms,
            },
            "yolo_ran": result.yolo_ran,
            "duty": {
                "mode": result.duty_mode,
                "reason": result.duty_reason,
                "yolo_frac": result.yolo_frac,
                "published": len(result.tracks),
            },
            "detections": [
                {
                    "bbox": [float(x) for x in d.bbox],
                    "conf": d.conf,
                    "class_name": d.class_name,
                    "hdc_class": d.hdc_class,
                    "hdc_score": d.hdc_score,
                }
                for d in result.detections
            ],
            "tracks": [
                {
                    "id": t.track_id,
                    "bbox": [float(x) for t_bbox in [t.bbox] for x in t_bbox],
                    "xy": [t.x, t.y],
                    "v": [t.vx, t.vy],
                    "class_name": t.class_name,
                    "hdc_class": t.hdc_class,
                    "hdc_score": t.hdc_score,
                    "confirmed": t.confirmed,
                }
                for t in result.tracks
            ],
        }


def _radar_to_image(truth: list[dict]) -> list[tuple[float, float]]:
    return [(float(t["x"]), float(t["y"])) for t in truth if "x" in t and "y" in t]

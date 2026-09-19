from __future__ import annotations

from dataclasses import dataclass, field

from tacyolo.track.associate import iou_xyxy
from tacyolo.types import GateResult, ROI, TrackState


@dataclass
class ScheduleDecision:
    """What the detector is allowed to do on this frame."""

    run_detector: bool
    mode: str
    rois: list[ROI] = field(default_factory=list)
    reason: str = ""
    force_crops: bool = False


class DutyCycle:
    """Skip YOLO when idle or already tracking; crop or tile when it must run.

    EKF coasts at full frame rate. The detector runs on unexplained motion or a
    periodic keyframe. Tiny movers get a padded high-res tile. C2 only sees
    published tracks (confirmed, not one-frame flicker).
    """

    def __init__(
        self,
        enabled: bool = True,
        detect_every: int = 5,
        refresh_age: int = 8,
        tiny_area_frac: float = 0.035,
        tile_min: int = 320,
        tile_pad: int = 24,
        cover_iou: float = 0.25,
        hdc_if_conf_below: float = 1.0,
        publish_min_hits: int = 2,
        publish_max_coast: int = 12,
        require_hdc: bool = False,
        min_hdc_score: float = 0.12,
    ) -> None:
        self.enabled = bool(enabled)
        self.detect_every = max(1, int(detect_every))
        self.refresh_age = max(1, int(refresh_age))
        self.tiny_area_frac = float(tiny_area_frac)
        self.tile_min = int(tile_min)
        self.tile_pad = int(tile_pad)
        self.cover_iou = float(cover_iou)
        self.hdc_if_conf_below = float(hdc_if_conf_below)
        self.publish_min_hits = int(publish_min_hits)
        self.publish_max_coast = int(publish_max_coast)
        self.require_hdc = bool(require_hdc)
        self.min_hdc_score = float(min_hdc_score)
        self.frames = 0
        self.detects = 0
        self._since_detect = 10**9

    @property
    def yolo_frac(self) -> float:
        return float(self.detects) / float(max(1, self.frames))

    def reset(self) -> None:
        self.frames = 0
        self.detects = 0
        self._since_detect = 10**9

    def decide(
        self,
        gate: GateResult,
        frame_shape: tuple[int, ...],
        tracks: list[TrackState],
        rf_hot: bool = False,
        use_tracker: bool = True,
    ) -> ScheduleDecision:
        self.frames += 1
        self._since_detect += 1
        h, w = int(frame_shape[0]), int(frame_shape[1])
        triggered = bool(gate.triggered or rf_hot)
        rois = list(gate.rois)

        if not self.enabled:
            run = triggered
            if run:
                self.detects += 1
                self._since_detect = 0
            return ScheduleDecision(
                run_detector=run,
                mode="full" if run else "idle",
                rois=rois,
                reason="legacy-gate" if run else "legacy-idle",
                force_crops=False,
            )

        if not use_tracker:
            if not triggered:
                return ScheduleDecision(False, "idle", [], "quiet-scene", False)
            return self._commit_detect(rois, w, h, reason="predict-no-track")

        confirmed = [t for t in tracks if t.confirmed]
        pending = [t for t in tracks if not t.confirmed and t.hits >= 1]
        live = bool(confirmed or pending)
        keyframe = self._since_detect >= self.detect_every
        stale = any(t.time_since_update >= self.refresh_age for t in confirmed)
        unexplained = _unexplained_rois(rois, confirmed, self.cover_iou)

        if unexplained:
            return self._commit_detect(unexplained, w, h, reason="unexplained-motion")

        if pending and triggered:
            pending_rois = rois or [_track_roi(t, w, h, self.tile_pad) for t in pending]
            return self._commit_detect(pending_rois, w, h, reason="confirm-pending")

        if live and (keyframe or stale):
            refresh = confirmed or pending
            track_rois = [_track_roi(t, w, h, self.tile_pad) for t in refresh]
            return self._commit_detect(track_rois, w, h, reason="keyframe-refresh")

        if live:
            reason = "coast-tracked" if triggered else "coast-quiet"
            return ScheduleDecision(False, "coast", [], reason, False)

        if triggered:
            return self._commit_detect(rois, w, h, reason="triggered-full" if not rois else "unexplained-motion")

        return ScheduleDecision(False, "idle", [], "quiet-scene", False)

    def hdc_needed(self, conf: float) -> bool:
        return float(conf) < self.hdc_if_conf_below

    def publish(self, tracks: list[TrackState]) -> list[TrackState]:
        if not self.enabled:
            return [t for t in tracks if t.hits >= 1]
        out: list[TrackState] = []
        for trk in tracks:
            if trk.hits < self.publish_min_hits or not trk.confirmed:
                continue
            if trk.time_since_update > self.publish_max_coast:
                continue
            if self.require_hdc:
                score = trk.hdc_score if trk.hdc_score is not None else 0.0
                if not trk.hdc_class or score < self.min_hdc_score:
                    continue
            out.append(trk)
        return out

    def _commit_detect(
        self,
        rois: list[ROI],
        width: int,
        height: int,
        reason: str,
    ) -> ScheduleDecision:
        self.detects += 1
        self._since_detect = 0
        if not rois:
            return ScheduleDecision(True, "full", [], reason, False)
        frame_area = max(1, width * height)
        prepared: list[ROI] = []
        any_tiny = False
        for roi in rois:
            clipped = roi.clip(width, height)
            if clipped.area < self.tiny_area_frac * frame_area:
                prepared.append(expand_tile(clipped, width, height, self.tile_min, self.tile_pad))
                any_tiny = True
            else:
                prepared.append(clipped)
        mode = "tile" if any_tiny else "crop"
        return ScheduleDecision(True, mode, prepared, reason, force_crops=True)


def expand_tile(roi: ROI, width: int, height: int, min_side: int, pad: int) -> ROI:
    side = max(int(min_side), (roi.x2 - roi.x1) + 2 * pad, (roi.y2 - roi.y1) + 2 * pad)
    cx = (roi.x1 + roi.x2) // 2
    cy = (roi.y1 + roi.y2) // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return ROI(x1=max(0, x1), y1=max(0, y1), x2=min(width, x2), y2=min(height, y2)).clip(width, height)


def _track_roi(track: TrackState, width: int, height: int, pad: int) -> ROI:
    x1, y1, x2, y2 = track.bbox
    return ROI(
        int(x1) - pad,
        int(y1) - pad,
        int(x2) + pad,
        int(y2) + pad,
    ).clip(width, height)


def _unexplained_rois(rois: list[ROI], tracks: list[TrackState], cover_iou: float) -> list[ROI]:
    if not rois:
        return []
    if not tracks:
        return list(rois)
    leftover: list[ROI] = []
    for roi in rois:
        box = (float(roi.x1), float(roi.y1), float(roi.x2), float(roi.y2))
        if any(_roi_explained_by_track(box, trk.bbox, cover_iou) for trk in tracks):
            continue
        leftover.append(roi)
    return leftover


def _roi_explained_by_track(
    roi: tuple[float, float, float, float],
    track: tuple[float, float, float, float],
    cover_iou: float,
) -> bool:
    if iou_xyxy(roi, track) >= cover_iou:
        return True
    ax1, ay1, ax2, ay2 = roi
    bx1, by1, bx2, by2 = track
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    roi_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    trk_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if roi_area > 0 and inter / roi_area >= 0.5:
        return True
    if trk_area > 0 and inter / trk_area >= 0.5:
        return True
    return False

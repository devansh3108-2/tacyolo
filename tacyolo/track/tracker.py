from __future__ import annotations

from tacyolo.track.associate import associate
from tacyolo.track.ekf import KalmanTrack, reset_track_ids
from tacyolo.track.midus import midus_associate
from tacyolo.types import Detection, TrackState


class MultiObjectTracker:
    def __init__(
        self,
        dt: float = 0.033,
        process_var: float = 25.0,
        meas_var: float = 4.0,
        radar_var: float = 16.0,
        max_age: int = 15,
        min_hits: int = 2,
        iou_weight: float = 0.55,
        mahalanobis_gate: float = 9.21,
        backend: str = "midus",
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
    ) -> None:
        self.dt = dt
        self.process_var = process_var
        self.meas_var = meas_var
        self.radar_var = radar_var
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_weight = iou_weight
        self.mahalanobis_gate = mahalanobis_gate
        self.backend = str(backend or "midus").lower()
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.tracks: list[KalmanTrack] = []
        reset_track_ids()

    def predict(self, dt: float | None = None) -> None:
        for trk in self.tracks:
            trk.predict(dt)

    def update(
        self,
        detections: list[Detection],
        radar_xy: list[tuple[float, float]] | None = None,
    ) -> list[TrackState]:
        if self.backend in {"ekf", "hungarian"}:
            matches, _unmatched_tracks, unmatched_dets = associate(
                self.tracks,
                detections,
                iou_weight=self.iou_weight,
                mahalanobis_gate=self.mahalanobis_gate,
            )
        else:
            matches, _unmatched_tracks, unmatched_dets = midus_associate(
                self.tracks,
                detections,
                high_thresh=self.high_thresh,
                low_thresh=self.low_thresh,
                iou_weight=self.iou_weight,
                mahalanobis_gate=self.mahalanobis_gate,
            )
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        for di in unmatched_dets:
            self.tracks.append(
                KalmanTrack(
                    detections[di],
                    dt=self.dt,
                    process_var=self.process_var,
                    meas_var=self.meas_var,
                    radar_var=self.radar_var,
                )
            )
        if radar_xy:
            self._fuse_radar(radar_xy)

        alive: list[KalmanTrack] = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.tracks = alive
        return [t.as_state(self.min_hits) for t in self.tracks if t.hits >= 1]

    def _fuse_radar(self, radar_xy: list[tuple[float, float]]) -> None:
        if not self.tracks or not radar_xy:
            return
        used = set()
        for trk in self.tracks:
            best = None
            best_d = 1e9
            for i, xy in enumerate(radar_xy):
                if i in used:
                    continue
                d = (trk.x[0] - xy[0]) ** 2 + (trk.x[1] - xy[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = i
            if best is not None and best_d < (120.0**2):
                trk.update_radar(radar_xy[best])
                used.add(best)

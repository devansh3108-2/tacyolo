from __future__ import annotations

import numpy as np

from tacyolo.types import Detection, TrackState


class KalmanTrack:
    """Constant-velocity EKF/KF in the image plane: state [x, y, vx, vy]."""

    _next_id = 1

    def __init__(
        self,
        det: Detection,
        dt: float = 0.033,
        process_var: float = 25.0,
        meas_var: float = 4.0,
        radar_var: float = 16.0,
    ) -> None:
        self.track_id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        x, y = det.cxcy
        x1, y1, x2, y2 = det.bbox
        self.w = max(4.0, x2 - x1)
        self.h = max(4.0, y2 - y1)
        self.dt = float(dt)
        self.class_name = det.class_name
        self.hdc_class = det.hdc_class
        self.hdc_score = det.hdc_score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.diag([meas_var, meas_var, 100.0, 100.0]).astype(np.float32)
        self.Q_base = float(process_var)
        self.R = np.eye(2, dtype=np.float32) * float(meas_var)
        self.R_radar = np.eye(2, dtype=np.float32) * float(radar_var)

    def _F(self) -> np.ndarray:
        dt = self.dt
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def predict(self, dt: float | None = None) -> None:
        if dt is not None:
            self.dt = float(dt)
        F = self._F()
        q = self.Q_base
        dt = self.dt
        qv = q * dt
        qa = q * dt * dt
        Q = np.array(
            [
                [qa, 0, qv, 0],
                [0, qa, 0, qv],
                [qv, 0, q, 0],
                [0, qv, 0, q],
            ],
            dtype=np.float32,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        z = np.array(det.cxcy, dtype=np.float32)
        self._update_z(z, self.R)
        x1, y1, x2, y2 = det.bbox
        self.w = 0.8 * self.w + 0.2 * max(4.0, x2 - x1)
        self.h = 0.8 * self.h + 0.2 * max(4.0, y2 - y1)
        self.class_name = det.class_name
        if det.hdc_class:
            self.hdc_class = det.hdc_class
            self.hdc_score = det.hdc_score
        self.hits += 1
        self.time_since_update = 0

    def update_radar(self, xy: tuple[float, float]) -> None:
        z = np.array(xy, dtype=np.float32)
        self._update_z(z, self.R_radar)

    def _update_z(self, z: np.ndarray, R: np.ndarray) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

    def mahalanobis(self, det: Detection) -> float:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        z = np.array(det.cxcy, dtype=np.float32)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        try:
            return float(y.T @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            return float("inf")

    def as_state(self, min_hits: int) -> TrackState:
        return TrackState(
            track_id=self.track_id,
            x=float(self.x[0]),
            y=float(self.x[1]),
            vx=float(self.x[2]),
            vy=float(self.x[3]),
            w=float(self.w),
            h=float(self.h),
            class_name=self.class_name,
            hdc_class=self.hdc_class,
            hdc_score=self.hdc_score,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            confirmed=self.hits >= min_hits,
            covariance=self.P.copy(),
        )


def reset_track_ids() -> None:
    KalmanTrack._next_id = 1

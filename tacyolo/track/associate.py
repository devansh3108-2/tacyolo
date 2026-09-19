from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tacyolo.track.ekf import KalmanTrack
from tacyolo.types import Detection


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def associate(
    tracks: list[KalmanTrack],
    detections: list[Detection],
    iou_weight: float = 0.55,
    mahalanobis_gate: float = 9.21,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    valid = np.ones_like(cost, dtype=bool)
    for i, trk in enumerate(tracks):
        pred_box = (
            float(trk.x[0] - trk.w * 0.5),
            float(trk.x[1] - trk.h * 0.5),
            float(trk.x[0] + trk.w * 0.5),
            float(trk.x[1] + trk.h * 0.5),
        )
        for j, det in enumerate(detections):
            iou = iou_xyxy(pred_box, det.bbox)
            maha = trk.mahalanobis(det)
            gated = maha > mahalanobis_gate and iou < 0.1
            if gated:
                valid[i, j] = False
                cost[i, j] = 1e3
            else:
                maha_n = min(maha / max(mahalanobis_gate, 1e-6), 2.0)
                cost[i, j] = iou_weight * (1.0 - iou) + (1.0 - iou_weight) * maha_n

    rows, cols = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))
    for r, c in zip(rows, cols):
        if not valid[r, c] or cost[r, c] >= 1e3:
            continue
        matches.append((int(r), int(c)))
        unmatched_tracks.discard(int(r))
        unmatched_dets.discard(int(c))
    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

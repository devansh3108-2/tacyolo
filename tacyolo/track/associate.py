"""Couple HDC gallery matching with Extended Kalman Filter (EKF) spatio-temporal constraints."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tacyolo.hdc.encode import cosine
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
    iou_weight: float = 0.40,
    spatial_weight: float = 0.30,
    hdc_weight: float = 0.30,
    mahalanobis_gate: float = 9.21,
    max_coasting_gate_mult: float = 3.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Associate tracks and detections using combined spatio-temporal EKF kinematics and HDC appearance.

    Ensures target identities persist reliably even when objects pass behind temporary physical cover
    by coupling EKF motion extrapolation with HDC cosine similarity.
    """
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

        # Scale spatial gate when object has been coasting behind physical cover
        coasting_mult = min(max_coasting_gate_mult, 1.0 + 0.35 * trk.time_since_update)
        effective_gate = mahalanobis_gate * coasting_mult

        for j, det in enumerate(detections):
            iou = iou_xyxy(pred_box, det.bbox)
            maha = trk.mahalanobis(det)

            # HDC Appearance Distance
            hdc_dist = 0.5  # default neutral
            has_hdc = False
            if trk.hdc_hv is not None and det.hdc_hv is not None:
                sim = cosine(trk.hdc_hv, det.hdc_hv)
                hdc_dist = float(np.clip(1.0 - sim, 0.0, 1.0))
                has_hdc = True
            elif trk.hdc_class and det.hdc_class:
                hdc_dist = 0.1 if (trk.hdc_class == det.hdc_class) else 0.8
                has_hdc = True

            # Spatio-Temporal Cover Rule:
            # If object was occluded (time_since_update > 0), IoU may be 0, but if
            # Mahalanobis is within the expanded EKF search ellipse and HDC agrees, allow match!
            gated = (maha > effective_gate and iou < 0.05)
            if has_hdc and hdc_dist > 0.85:
                # Strong HDC mismatch rejects candidate
                gated = True

            if gated:
                valid[i, j] = False
                cost[i, j] = 1e4
            else:
                maha_norm = min(maha / max(effective_gate, 1e-6), 2.0)
                # Weighted fusion of IoU, kinematic Mahalanobis distance, and HDC appearance
                c_iou = 1.0 - iou
                c_spatial = maha_norm * 0.5
                c_hdc = hdc_dist

                if has_hdc:
                    cost[i, j] = (
                        iou_weight * c_iou +
                        spatial_weight * c_spatial +
                        hdc_weight * c_hdc
                    )
                else:
                    # Fallback without HDC
                    w_sum = iou_weight + spatial_weight
                    cost[i, j] = (iou_weight / w_sum) * c_iou + (spatial_weight / w_sum) * c_spatial

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

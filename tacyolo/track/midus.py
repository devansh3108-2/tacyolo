"""MIDUS: Multi-object ID Dual-Update Stage (ByteTrack-style, EKF + radar)."""

from __future__ import annotations

from tacyolo.track.associate import associate
from tacyolo.track.ekf import KalmanTrack
from tacyolo.types import Detection


def midus_associate(
    tracks: list[KalmanTrack],
    detections: list[Detection],
    *,
    high_thresh: float = 0.5,
    low_thresh: float = 0.1,
    iou_weight: float = 0.55,
    mahalanobis_gate: float = 9.21,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Two-stage association: high-confidence first, then low-confidence rescue.

    New tracks are born only from unmatched high-confidence detections.
    """
    if not detections:
        return [], list(range(len(tracks))), []

    high_idx = [i for i, d in enumerate(detections) if d.conf >= high_thresh]
    low_idx = [i for i, d in enumerate(detections) if low_thresh <= d.conf < high_thresh]
    high_dets = [detections[i] for i in high_idx]
    low_dets = [detections[i] for i in low_idx]

    matches: list[tuple[int, int]] = []
    m1, u_tracks, u_high_local = associate(
        tracks, high_dets, iou_weight=iou_weight, mahalanobis_gate=mahalanobis_gate
    )
    for ti, dj in m1:
        matches.append((ti, high_idx[dj]))

    rest_tracks = list(u_tracks)
    if rest_tracks and low_dets:
        rest_objs = [tracks[i] for i in rest_tracks]
        m2, u_rest_local, _u_low = associate(
            rest_objs, low_dets, iou_weight=iou_weight, mahalanobis_gate=mahalanobis_gate
        )
        for local_t, dj in m2:
            matches.append((rest_tracks[local_t], low_idx[dj]))
        u_tracks = [rest_tracks[i] for i in u_rest_local]
    unmatched_dets = [high_idx[i] for i in u_high_local]
    unmatched_tracks = list(u_tracks)
    return matches, unmatched_tracks, unmatched_dets

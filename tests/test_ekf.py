from __future__ import annotations

import numpy as np

from tacyolo.track.ekf import KalmanTrack, reset_track_ids
from tacyolo.track.tracker import MultiObjectTracker
from tacyolo.types import Detection


def _det(x: float, y: float, w: float = 20.0, h: float = 16.0) -> Detection:
    return Detection(
        bbox=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
        conf=0.9,
        class_id=0,
        class_name="red",
    )


def test_ekf_smoother_than_raw_jitter() -> None:
    reset_track_ids()
    rng = np.random.default_rng(1)
    raw = []
    filt = []
    trk = None
    for t in range(40):
        true_x = 40.0 + 3.0 * t
        meas = true_x + float(rng.normal(0, 4.0))
        det = _det(meas, 80.0)
        raw.append(meas)
        if trk is None:
            trk = KalmanTrack(det, dt=1.0, process_var=2.0, meas_var=16.0)
        else:
            trk.predict(1.0)
            trk.update(det)
        filt.append(float(trk.x[0]))
    raw_err = np.mean((np.array(raw) - (40.0 + 3.0 * np.arange(40))) ** 2)
    filt_err = np.mean((np.array(filt) - (40.0 + 3.0 * np.arange(40))) ** 2)
    assert filt_err < raw_err


def test_tracker_assigns_stable_ids() -> None:
    reset_track_ids()
    mot = MultiObjectTracker(dt=1.0, max_age=5, min_hits=1, process_var=8.0, meas_var=4.0)
    ids = []
    for t in range(12):
        mot.predict(1.0)
        det = _det(30 + 2.5 * t, 50)
        states = mot.update([det])
        ids.append(states[0].track_id)
    assert len(set(ids)) == 1


def test_midus_rescues_low_conf_and_does_not_birth_on_low() -> None:
    reset_track_ids()
    mot = MultiObjectTracker(dt=1.0, max_age=8, min_hits=1, backend="midus", high_thresh=0.5, low_thresh=0.1)
    mot.predict(1.0)
    born = mot.update([_det(30, 50)])
    tid = born[0].track_id
    mot.predict(1.0)
    weak = _det(33, 50)
    weak.conf = 0.2
    kept = mot.update([weak])
    assert kept[0].track_id == tid
    mot.predict(1.0)
    ghost = _det(200, 200)
    ghost.conf = 0.15
    states = mot.update([ghost])
    ids = {s.track_id for s in states}
    assert tid in ids
    assert len(ids) == 1


def test_ekf_backend_births_low_conf_midus_does_not() -> None:
    reset_track_ids()
    ghost = _det(80, 80)
    ghost.conf = 0.2
    midus = MultiObjectTracker(dt=1.0, min_hits=1, backend="midus")
    assert midus.update([ghost]) == []
    reset_track_ids()
    ekf = MultiObjectTracker(dt=1.0, min_hits=1, backend="ekf")
    born = ekf.update([ghost])
    assert len(born) == 1

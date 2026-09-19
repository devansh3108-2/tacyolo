from __future__ import annotations

from tacyolo.optical.duty import DutyCycle, expand_tile
from tacyolo.types import GateResult, ROI, TrackState


def _track(
    *,
    x: float = 50.0,
    y: float = 45.0,
    w: float = 80.0,
    h: float = 70.0,
    hits: int = 4,
    confirmed: bool = True,
    tsu: int = 0,
    hdc_class: str | None = None,
    hdc_score: float | None = None,
) -> TrackState:
    return TrackState(
        track_id=1,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        w=w,
        h=h,
        class_name="vehicle",
        hdc_class=hdc_class,
        hdc_score=hdc_score,
        hits=hits,
        age=hits,
        time_since_update=tsu,
        confirmed=confirmed,
    )


def test_quiet_scene_stays_idle() -> None:
    duty = DutyCycle(detect_every=5)
    gate = GateResult(triggered=False, energy=0.2, rois=[])
    decision = duty.decide(gate, (120, 160), tracks=[])
    assert decision.run_detector is False
    assert decision.mode == "idle"


def test_new_motion_runs_crop_then_coasts() -> None:
    duty = DutyCycle(detect_every=5, tiny_area_frac=0.01)
    gate = GateResult(triggered=True, energy=6.0, rois=[ROI(10, 10, 90, 80)])
    first = duty.decide(gate, (120, 160), tracks=[])
    assert first.run_detector
    assert first.mode == "crop"
    assert first.force_crops

    tracked = duty.decide(gate, (120, 160), tracks=[_track()])
    assert tracked.run_detector is False
    assert tracked.mode == "coast"
    assert tracked.reason == "coast-tracked"


def test_tiny_roi_uses_high_res_tile() -> None:
    duty = DutyCycle(tiny_area_frac=0.05, tile_min=64, tile_pad=8)
    gate = GateResult(triggered=True, energy=4.0, rois=[ROI(40, 40, 48, 48)])
    decision = duty.decide(gate, (200, 200), tracks=[])
    assert decision.mode == "tile"
    assert decision.rois
    tile = decision.rois[0]
    assert (tile.x2 - tile.x1) >= 64
    assert (tile.y2 - tile.y1) >= 64


def test_keyframe_refresh_after_coast() -> None:
    duty = DutyCycle(detect_every=3, refresh_age=99)
    gate_hot = GateResult(triggered=True, energy=5.0, rois=[ROI(10, 10, 90, 80)])
    duty.decide(gate_hot, (120, 160), tracks=[])
    coast_gate = GateResult(triggered=False, energy=0.1, rois=[])
    modes = []
    for _ in range(4):
        decision = duty.decide(coast_gate, (120, 160), tracks=[_track(tsu=1)])
        modes.append(decision.mode)
    assert "coast" in modes
    assert "crop" in modes or "tile" in modes


def test_publish_drops_one_frame_flicker() -> None:
    duty = DutyCycle(publish_min_hits=2, require_hdc=False)
    weak = _track(hits=1, confirmed=False)
    strong = _track(hits=4, confirmed=True)
    published = duty.publish([weak, strong])
    assert [t.hits for t in published] == [4]


def test_publish_can_require_gallery_match() -> None:
    duty = DutyCycle(require_hdc=True, min_hdc_score=0.2)
    no_id = _track(hdc_class=None, hdc_score=0.0)
    named = _track(hdc_class="uav", hdc_score=0.4)
    assert duty.publish([no_id]) == []
    assert duty.publish([named])[0].hdc_class == "uav"


def test_expand_tile_stays_inside_frame() -> None:
    tile = expand_tile(ROI(0, 0, 8, 8), width=100, height=80, min_side=40, pad=4)
    assert tile.x1 >= 0 and tile.y1 >= 0
    assert tile.x2 <= 100 and tile.y2 <= 80
    assert tile.area > 8 * 8

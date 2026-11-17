"""Camera geometry, frozen-model adapter contract, and missing-data behavior."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from dreamerv3_attention.observations import FrameBundle
from dreamerv3_attention.sensors import SensorFrameBuffer
from dreamerv3_attention.vision import VisionPerceptionProvider


def calibration():
    return dict(intrinsics_px=dict(fx=640., fy=640., cx=639.5, cy=359.5),
                extrinsics_ego=dict(location_m=(1.5, 0., 2.4),
                                    rotation_pitch_yaw_roll_deg=(0., 0., 0.)))


def make_provider(classes=(), *, low_confidence=False, boxes=None):
    provider = object.__new__(VisionPerceptionProvider)
    provider.device = "cpu"
    provider.seen = []
    # Project the straight line y=-0.7 onto pixels at several visible depths.
    center_line = [[639.5-640.*.7/(x-1.5), 359.5+640.*2.4/(x-1.5)]
                   for x in (6., 10., 20.)]

    def road(bgr):
        provider.seen.append(bgr.copy())
        return dict(lane_center=dict(center_line=center_line, low_confidence=low_confidence),
                    crosswalks=[dict(xyxy=[500., 500., 780., 600.])])

    provider.road = road
    xyxy = boxes if boxes is not None else [[600., 400., 680., 600.]]*len(classes)
    boxes = NS(xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
               conf=torch.full((len(classes),), .8), cls=torch.tensor(classes))
    provider.detector = NS(predict=lambda *a, **kw: [NS(boxes=boxes)])
    return provider


def bundle():
    return FrameBundle(42, 2.1, rgb=np.full((720, 1280, 3), [1, 2, 3], np.uint8),
                       valid={"rgb": True}, calibration={"sensors": {"rgb": calibration()}})


def test_provider_consumes_real_sensor_buffer_calibration_contract():
    buffer = SensorFrameBuffer()
    buffer.set_calibration("rgb", calibration())
    buffer.add("rgb", 42, 2.1, bundle().rgb)
    p = make_provider()
    out = p.predict(buffer.bundle(42))
    assert len(p.seen) == 1
    assert out.driving["observations_valid"]
    assert out.driving["lane_error_m"] == pytest.approx(.7)


def test_ground_projection_uses_camera_geometry_and_rejects_horizon():
    np.testing.assert_allclose(VisionPerceptionProvider.ground_point(639.5, 599.5, calibration()),
                               [7.9, 0., 0.])
    assert VisionPerceptionProvider.ground_point(639.5, 350., calibration()) is None
    c = calibration()
    c["extrinsics_ego"]["rotation_pitch_yaw_roll_deg"] = (5., 0., 0.)
    with pytest.raises(ValueError, match="level front camera"):
        VisionPerceptionProvider.ground_point(640., 600., c)


def test_without_depth_rows_fall_back_to_flat_road_and_ignore_radar():
    p, b = make_provider((0, 4, 5, 6)), bundle()
    b.radars = {"front": np.array([[.01, 0., 0., 0.]])}
    prediction = p.predict(b)
    prediction.validate_for(b)
    np.testing.assert_array_equal(p.seen[0][0, 0], [3, 2, 1])
    assert prediction.driving["lane_error_m"] == pytest.approx(.7)
    assert prediction.driving["heading_error_rad"] == pytest.approx(0., abs=1e-7)
    assert prediction.driving["observations_valid"]
    assert prediction.driving["crossing_occupied"]
    assert prediction.driving["speed_limit_mps"] is None
    assert prediction.driving["speed_limit_confidence"] == 0.
    assert prediction.depth_m is None
    assert prediction.child_labels_authored
    np.testing.assert_array_equal(prediction.objects[:, 6:8], [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(prediction.objects[:, 3:6], 0.)
    assert (prediction.objects[:, 10] == np.float32(.8)).all()
    assert not prediction.objects[:, 12:15].any()


def test_conflicting_lights_choose_red_and_do_not_invent_stop_line():
    prediction = make_provider((1, 2, 3)).predict(bundle())
    assert prediction.driving["red_light"]
    assert prediction.driving["red_light_distance_m"] is None
    assert not prediction.driving["yellow_light"]
    assert prediction.driving["yellow_light_confidence"] == 0.


def test_yellow_retains_confidence_and_unknown_range():
    prediction = make_provider((2,)).predict(bundle())
    assert prediction.driving["yellow_light"]
    assert prediction.driving["yellow_light_confidence"] == pytest.approx(.8)
    assert prediction.driving["yellow_light_distance_m"] is None


@pytest.mark.parametrize("failure", ["missing_rgb", "missing_calibration", "low_confidence", "no_lane"])
def test_missing_or_unreliable_lane_never_falls_back_to_truth(failure):
    p, b = make_provider(low_confidence=failure == "low_confidence"), bundle()
    if failure == "missing_rgb":
        b.rgb = None
    elif failure == "missing_calibration":
        b.calibration = {}
    elif failure == "no_lane":
        p.road = lambda bgr: dict(lane_center=None, crosswalks=[])
    assert not p.predict(b).driving["observations_valid"]


def _lane_road(provider, center_line, extrapolated):
    def road(bgr):
        return dict(lane_center=dict(center_line=center_line, extrapolated_fraction=extrapolated,
                                     low_confidence=extrapolated > .5), crosswalks=[])
    provider.road = road


def test_extrapolation_threshold_replaces_fixed_low_confidence_flag():
    p = make_provider()
    line = [[639.5-640.*.7/(x-1.5), 359.5+640.*2.4/(x-1.5)] for x in (6., 10., 20.)]
    p.max_extrapolation, p.lane_hold_s, p._last_lane = .7, 0., None
    _lane_road(p, line, .6)
    assert p.predict(bundle()).driving["observations_valid"]
    _lane_road(p, line, .8)
    assert not p.predict(bundle()).driving["observations_valid"]


def test_lane_hold_bridges_short_gaps_only():
    p = make_provider()
    line = [[639.5-640.*.7/(x-1.5), 359.5+640.*2.4/(x-1.5)] for x in (6., 10., 20.)]
    p.max_extrapolation, p.lane_hold_s, p._last_lane = .7, 2., None
    _lane_road(p, line, .1)
    first = p.predict(bundle()).driving
    assert first["observations_valid"]
    p.road = lambda bgr: dict(lane_center=None, crosswalks=[])
    held = bundle()
    held.timestamp = 3.5
    out = p.predict(held).driving
    assert out["observations_valid"] and out["lane_error_m"] == pytest.approx(first["lane_error_m"])
    late = bundle()
    late.timestamp = 4.2
    assert not p.predict(late).driving["observations_valid"]


def depth_bundle(metres=12.):
    b = bundle()
    b.depth = np.full((720, 1280), metres, np.float32)
    b.valid["depth"] = True
    return b


def test_depth_camera_places_objects_and_fills_depth_channels():
    p = make_provider((0, 4))
    prediction = p.predict(depth_bundle(12.))
    # Bottom-centre pixel (640, 600) at 12 m along the camera axis, camera at x=1.5, z=2.4.
    np.testing.assert_allclose(prediction.objects[:, 0], 13.5/100., rtol=1e-5)
    np.testing.assert_allclose(prediction.depth_m, 12., rtol=1e-5)
    assert prediction.depth_valid.all()
    assert prediction.uncertainty is not None and np.isfinite(prediction.uncertainty).all()
    assert not prediction.objects[:, 12:15].any()  # the environment writes these from depth_m
    assert prediction.driving["observations_valid"]


def test_truncated_box_with_depth_no_longer_invalidates_frame():
    p = make_provider((4,), boxes=[[600., 500., 680., 720.]])
    prediction = p.predict(depth_bundle(5.))
    assert prediction.driving["observations_valid"] and len(prediction.objects) == 1
    no_depth = make_provider((4,), boxes=[[600., 500., 680., 720.]]).predict(bundle())
    assert not no_depth.driving["observations_valid"]


def light_box(lateral_m, depth_m):
    u = 639.5 + 640. * lateral_m / depth_m
    return [u-10., 200., u+10., 240.]


def test_red_light_distance_uses_depth_and_town02_stop_offset():
    p = make_provider((1,), boxes=[light_box(2.5, 15.)])
    fields = p.predict(depth_bundle(15.)).driving
    assert fields["red_light"]
    # Lamp 15 m along the camera axis plus 1.5 m mount, minus 7.1 m to the stop line and 2.5 m bumper.
    assert fields["red_light_distance_m"] == pytest.approx(16.5-7.1-2.5, abs=1e-4)


def test_red_light_for_another_lane_is_ignored_but_unknown_range_still_stops():
    far_left = make_provider((1,), boxes=[light_box(-6., 15.)]).predict(depth_bundle(15.)).driving
    assert not far_left["red_light"] and far_left["red_light_distance_m"] is None
    unknown = make_provider((1,), boxes=[light_box(-6., 15.)]).predict(bundle()).driving
    assert unknown["red_light"] and unknown["red_light_distance_m"] is None


def test_yellow_light_distance_and_red_priority():
    yellow = make_provider((2,), boxes=[light_box(2.5, 20.)]).predict(depth_bundle(20.)).driving
    assert yellow["yellow_light"] and yellow["yellow_light_confidence"] == pytest.approx(.8)
    assert yellow["yellow_light_distance_m"] == pytest.approx(21.5-7.1-2.5, abs=1e-4)
    both = make_provider((1, 2), boxes=[light_box(2.5, 20.)]*2).predict(depth_bundle(20.)).driving
    assert both["red_light"] and not both["yellow_light"] and both["yellow_light_distance_m"] is None


def test_speed_limit_applies_after_confirmed_sign_is_passed(monkeypatch):
    from dreamerv3_attention import vision
    from dreamerv3_attention.speed_sign import SpeedLimitTracker
    monkeypatch.setattr(vision, "read_speed_limit", lambda reader, crop: (40, .9))
    p = make_provider((6,), boxes=[light_box(3., 10.)])
    p.sign_reader, p.speed_limits = object(), SpeedLimitTracker()
    for t in (0., .1, .2):
        b = depth_bundle(10.)
        b.timestamp = t
        assert p.predict(b).driving["speed_limit_mps"] is None  # confirmed, not yet passed
    p.detector = NS(predict=lambda *a, **kw: [NS(boxes=NS(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0),
                                                        cls=torch.zeros(0, dtype=torch.long)))])
    b = depth_bundle(10.)
    b.timestamp = .6
    fields = p.predict(b).driving
    assert fields["speed_limit_mps"] == pytest.approx(40/3.6)
    assert fields["speed_limit_confidence"] == pytest.approx(1.)
    p.reset()
    assert p.predict(b).driving["speed_limit_mps"] is None


def test_speed_limit_tracker_needs_votes_and_a_close_pass():
    from dreamerv3_attention.speed_sign import SpeedLimitTracker
    tracker = SpeedLimitTracker()
    tracker.update(0., 30., 30, True)
    tracker.update(.1, 25., 30, True)
    assert tracker.update(.6, None) == (None, 0.)  # two votes only
    tracker = SpeedLimitTracker()
    for t, d in ((0., 40.), (.1, 35.), (.2, 30.)):
        tracker.update(t, d, 30, True)
    assert tracker.update(.6, None) == (None, 0.)  # left the view while still far away
    tracker = SpeedLimitTracker()
    for t, value in ((0., 30), (.1, 30), (.2, 30), (.3, 60), (.4, 60)):
        tracker.update(t, 12., value, True)
    assert tracker.update(.8, None) == (None, 0.)  # 3 against 2 is not a clear majority

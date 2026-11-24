import json
"""Camera geometry, frozen-model adapter contract, and missing-data behavior."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor
from dreamerv3_attention.observations import FrameBundle
from dreamerv3_attention.sensors import SensorFrameBuffer
from dreamerv3_attention.object_tracker import ObjectTracker
from dreamerv3_attention.vision import VisionPerceptionProvider, merge_nested_lights, path_lane


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


def route_line(x):
    return np.array([[x, -4., 0.], [x, 4., 0.]])


def measured_signal_scene(classes=(1,), lamp_depth=20., route_x=15.):
    # The rendered road has a painted stripe at x=12 m; the map stop line is elsewhere.
    from dreamerv3_attention.tests.test_stop_lines import scene
    image, depth, c, _, _ = scene(road_height=-.3)
    cx, fx = c['intrinsics_px']['cx'], c['intrinsics_px']['fx']
    u = cx+fx*2./lamp_depth
    boxes = [[u-6.,175.,u+6.,195.]]*len(classes)
    p = make_provider(classes, boxes=boxes)
    center = [[cx, c['intrinsics_px']['cy']+fx*2.4/(x-1.5)] for x in (6.,10.,20.)]
    p.road = lambda bgr: dict(lane_center=dict(center_line=center, low_confidence=False), crosswalks=[])
    for x1,y1,x2,y2 in boxes:
        depth[int(y1):int(y2),int(x1):int(x2)] = lamp_depth
    b = FrameBundle(42, 2.1, rgb=image[:,:,::-1].copy(), depth=depth, valid=dict(rgb=True,depth=True),
                    calibration={'sensors':{'rgb':c}}, ego_pose=(0.,0.,.3,0.),
                    ego_path=np.array([[0.,0.],[40.,0.]]), front_bumper_m=2.2,
                    route_stop_lines=(route_line(route_x),))
    return p, b


@pytest.mark.parametrize('lamp_depth', [15.,30.])
def test_red_light_distance_uses_route_stop_line_and_actual_vehicle_bumper(lamp_depth):
    p, b = measured_signal_scene(lamp_depth=lamp_depth)
    out = p.predict(b)
    fields = out.driving
    assert fields["red_light"]
    # Map stop line at x=15 m, regardless of lamp mounting or the painted stripe.
    assert fields["red_light_distance_m"] == pytest.approx(15.-2.2, abs=1e-6)
    assert out.signals.shape == (2,16)


def test_red_light_without_route_stop_on_planned_path_is_not_for_this_route():
    # Regression: a red lamp seen before its stop entered the planned path used to
    # give an unknown range and a full stop tens of metres early.
    p, b = measured_signal_scene(lamp_depth=30.)
    b.route_stop_lines = ()
    fields = p.predict(b).driving
    assert not fields['red_light'] and fields['red_light_distance_m'] is None
    p, b = measured_signal_scene((2,))
    b.route_stop_lines = (route_line(60.),)  # beyond the 40 m planned path
    fields = p.predict(b).driving
    assert not fields['yellow_light'] and fields['yellow_light_confidence'] == 0.


def test_lamp_depth_alone_does_not_invent_a_stop_line():
    fields = make_provider((1,), boxes=[light_box(2.5,15.)]).predict(depth_bundle(15.)).driving
    assert fields['red_light'] and fields['red_light_distance_m'] is None


def test_red_light_for_another_lane_is_ignored_but_unknown_range_still_stops():
    far_left = make_provider((1,), boxes=[light_box(-6., 15.)]).predict(depth_bundle(15.)).driving
    assert not far_left["red_light"] and far_left["red_light_distance_m"] is None
    unknown = make_provider((1,), boxes=[light_box(-6., 15.)]).predict(bundle()).driving
    assert unknown["red_light"] and unknown["red_light_distance_m"] is None


def test_yellow_light_distance_and_red_priority():
    p, b = measured_signal_scene((2,))
    yellow = p.predict(b).driving
    assert yellow["yellow_light"] and yellow["yellow_light_confidence"] == pytest.approx(.8)
    assert yellow["yellow_light_distance_m"] == pytest.approx(15.-2.2, abs=1e-6)
    p, b = measured_signal_scene((1,2))
    both = p.predict(b).driving
    assert both["red_light"] and not both["yellow_light"] and both["yellow_light_distance_m"] is None


def test_nested_light_boxes_on_one_head_merge_into_the_larger_box():
    housing, lamp = [100., 100., 120., 160.], [102., 102., 118., 125.]
    boxes, conf, cls = merge_nested_lights([lamp, housing], [.6, .3], [3, 3])
    np.testing.assert_allclose(boxes, [housing])
    assert conf.tolist() == pytest.approx([.6]) and cls.tolist() == [3]


@pytest.mark.parametrize("pair, confidences, expected", [
    ((3, 1), (.9, .3), 1),  # any red stays red
    ((1, 2), (.3, .9), 1),
    ((2, 3), (.25, .5), 3),  # otherwise the more confident colour
    ((3, 2), (.5, .25), 3),
])
def test_conflicting_nested_light_colours_resolve_like_the_signal_tracker(pair, confidences, expected):
    boxes, _, cls = merge_nested_lights([[100., 100., 120., 160.], [102., 102., 118., 125.]],
                                        confidences, pair)
    assert len(boxes) == 1 and cls.tolist() == [expected]


def test_light_merge_keeps_separate_heads_and_never_touches_other_classes():
    left, right = [100., 100., 120., 160.], [115., 100., 135., 160.]  # 25% overlap: two heads
    adult, child = [300., 300., 360., 450.], [310., 340., 350., 440.]  # child inside adult
    sign = [302., 302., 318., 318.]
    boxes, conf, cls = merge_nested_lights([left, right, adult, child, sign], [.5]*5, [1, 3, 4, 5, 6])
    assert cls.tolist() == [1, 3, 4, 5, 6] and len(boxes) == 5
    empty = merge_nested_lights(np.zeros((0, 4)), [], [])
    assert all(len(x) == 0 for x in empty)
    with pytest.raises(ValueError, match="same length"):
        merge_nested_lights([left], [.5, .5], [1])


def test_provider_sends_one_light_token_for_a_double_boxed_head():
    p, b = measured_signal_scene((1, 1))
    cx, fx = 639.5, 640.
    u = cx+fx*2./20.
    p.detector = make_provider((1, 1), boxes=[[u-6., 175., u+6., 195.], [u-5., 176., u+5., 186.]]).detector
    out = p.predict(b)
    assert out.driving["red_light"]
    assert out.driving["red_light_distance_m"] == pytest.approx(15.-2.2, abs=1e-6)
    assert out.signals.shape == (2, 16)  # stop line plus a single light, not two


def test_provider_holds_red_at_route_stop_line_and_retires_it_after_passing():
    p, b = measured_signal_scene()
    first = p.predict(b).driving['red_light_distance_m']
    p.detector = NS(predict=lambda *a, **kw: [NS(boxes=NS(xyxy=torch.zeros((0,4)),
                     conf=torch.zeros(0), cls=torch.zeros(0,dtype=torch.long)))])
    b.rgb[:] = 45
    # The environment re-expresses the map line in the moved ego frame.
    b.timestamp, b.ego_pose, b.route_stop_lines = 2.4, (4.,0.,.3,0.), (route_line(11.),)
    held = p.predict(b).driving
    assert held['red_light'] and held['red_light_distance_m'] == pytest.approx(first-4.)
    # Range stays known while the lamp is out of view; colour is not cleared by absence.
    b.timestamp = 4.5
    later = p.predict(b).driving
    assert later['red_light'] and later['red_light_distance_m'] == pytest.approx(first-4.)
    b.timestamp, b.ego_pose, b.route_stop_lines = 4.6, (14.,0.,.3,0.), ()
    assert not p.predict(b).driving['red_light']


@pytest.mark.parametrize('missing', ['rgb','calibration'])
def test_missing_camera_keeps_tracked_red_but_marks_observations_invalid(missing):
    p, b = measured_signal_scene()
    p.predict(b)
    b.timestamp = 2.3
    if missing == 'rgb':
        b.rgb = None
    else:
        b.calibration = {}
    fields = p.predict(b).driving
    assert fields['red_light'] and not fields['observations_valid']
    p.reset()
    assert not p.predict(b).driving['red_light']


def test_speed_limit_applies_after_confirmed_sign_is_passed(monkeypatch):
    from dreamerv3_attention import vision
    from dreamerv3_attention.speed_sign import SpeedLimitTracker
    monkeypatch.setattr(vision, "read_speed_limit", lambda reader, crop: (40, .9))
    p = make_provider((6,), boxes=[light_box(3., 10.)])
    p.sign_reader, p.speed_limits = object(), SpeedLimitTracker()
    for t in (0., .15, .3, .45, .6, .75):  # a first limit needs the stricter raise vote count
        b = depth_bundle(10.)
        b.timestamp = t
        assert p.predict(b).driving["speed_limit_mps"] is None  # confirmed, not yet passed
    p.detector = NS(predict=lambda *a, **kw: [NS(boxes=NS(xyxy=torch.zeros((0, 4)), conf=torch.zeros(0),
                                                        cls=torch.zeros(0, dtype=torch.long)))])
    b = depth_bundle(10.)
    b.timestamp = 1.2
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


def test_speed_limit_raise_needs_more_votes_than_a_reduction():
    # Regression (experiment_28): a 60 confirmed from three reads in a 30 zone.
    from dreamerv3_attention.speed_sign import SIGN_MIN_VOTES, SIGN_RAISE_MIN_VOTES, SpeedLimitTracker

    def pass_sign(tracker, start, value, reads):
        for i in range(reads):
            tracker.update(start + .1*i, 12., value, True)
        return tracker.update(start + .1*reads + .4, None)

    tracker = SpeedLimitTracker()
    assert pass_sign(tracker, 0., 30, SIGN_RAISE_MIN_VOTES - 1) == (None, 0.)  # first limit: raise rule
    assert pass_sign(tracker, 2., 30, SIGN_RAISE_MIN_VOTES)[0] == 30
    assert pass_sign(tracker, 4., 60, SIGN_RAISE_MIN_VOTES - 1)[0] == 30  # raise ignored
    assert pass_sign(tracker, 6., 60, SIGN_RAISE_MIN_VOTES)[0] == 60
    assert pass_sign(tracker, 8., 30, SIGN_MIN_VOTES)[0] == 30  # reductions stay quick


def test_speed_limit_parser_ignores_non_ascii_digit_glyphs():
    from dreamerv3_attention.speed_sign import parse_limit
    # Regression: OCR returned "③0" during stage1 training and int() raised.
    assert parse_limit("③0") is None
    assert parse_limit("3O") == 30


def test_path_lane_matches_vision_sign_conventions():
    from dreamerv3_attention.vision import path_lane
    # Centre line at y=-0.7 gives +0.7, as the vision fit does in the tests above.
    assert path_lane(np.array([[0., -.7], [20., -.7]])) == pytest.approx((.7, 0.))
    # Path bending right (+y): negative heading error, like a vision slope of +1.
    assert path_lane(np.array([[-5., 0.], [5., 0.], [15., 10.]]))[1] == pytest.approx(0.)
    assert path_lane(np.array([[0., 0.], [10., 10.]])) == pytest.approx((0., -math.pi/4))
    assert path_lane(None) is None and path_lane(np.array([[0., 0.]])) is None


@pytest.mark.parametrize("low_confidence", [False, True])
def test_near_junction_lane_comes_from_planned_path_not_vision(low_confidence):
    # Regression: vision lost the ego lane ~9 m before Town01 junctions, the hold
    # expired, and the car halted inside the junction with invalid observations.
    b = bundle()
    b.ego_path, b.near_junction = np.array([[0., .4], [10., .4]]), True
    fields = make_provider(low_confidence=low_confidence).predict(b).driving
    assert fields["observations_valid"]
    assert fields["lane_error_m"] == pytest.approx(-.4)  # vision alone would report +0.7
    b.near_junction = False
    fields = make_provider().predict(b).driving
    # Outside junctions vision is used; only a disagreement over 2 m rejects it.
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.7)
    b.ego_path = np.array([[0., -.5], [10., -.5]])  # route agrees with vision within 0.5 m
    fields = make_provider().predict(b).driving
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.7)


def test_route_lane_seeds_hold_after_junction_exit():
    p, b = make_provider(low_confidence=True), bundle()
    p.lane_hold_s = 2.
    b.ego_path, b.near_junction = np.array([[0., .4], [10., .4]]), True
    p.predict(b)
    b.near_junction, b.timestamp = False, b.timestamp + 1.
    held = p.predict(b).driving
    assert held["observations_valid"] and held["lane_error_m"] == pytest.approx(-.4)
    b.ego_path = None
    b.timestamp += 1.5
    assert not p.predict(b).driving["observations_valid"]


def test_expired_lane_recovers_at_queue_speed_after_red_clears():
    provider, camera = measured_signal_scene(route_x=20.6)
    provider.lane_hold_s = 2.
    vision_road = provider.road
    assert provider.predict(camera).driving["lane_source"] == "vision"
    provider.road = lambda _: dict(lane_center=None, crosswalks=[])
    camera.timestamp += 1.
    assert provider.predict(camera).driving["lane_source"] == "held"

    camera.timestamp += 1.05
    camera.ego_path = np.array([[0., -.8], [40., -.8]])
    fields = provider.predict(camera).driving
    assert fields["observations_valid"] and fields["lane_source"] == "route_fallback"
    assert fields["lane_error_m"] == pytest.approx(.8)
    supervisor = SafetySupervisor(DrivingConfig(restart_assist_enabled=True))
    queued = DrivingState(**fields, vehicle_ahead=True, front_hazard_m=6.5)
    assert queued.red_light and queued.red_light_distance_m == pytest.approx(18.4)
    stopped = supervisor.step(queued, (0., 1.), .05)
    assert stopped.action == (0., -1.) and "red_light_stop" in stopped.intervention_reasons

    camera.timestamp += .05
    provider.detector = measured_signal_scene((3,), route_x=20.6)[0].detector
    fields = provider.predict(camera).driving
    assert not fields["red_light"] and fields["lane_source"] == "route_fallback"
    following = DrivingState(**fields, vehicle_ahead=True, front_hazard_m=6.5)
    assert 0. < supervisor.step(following, (0., 1.), .05).target_speed_mps < 1.5
    released = supervisor.step(DrivingState(**fields), (0., 1.), .05)
    assert released.action[1] > 0. and released.target_speed_mps == pytest.approx(1.5)
    overspeed = supervisor.step(DrivingState(**fields, speed_mps=2.), (0., 1.), .05)
    assert overspeed.action[1] < 0.

    provider.road = vision_road
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    # Vision is 0.8 m from the aligned route here and is still used; neither is capped.
    assert fields["lane_source"] == "vision"
    assert supervisor.step(DrivingState(**fields), (0., 1.), .05).target_speed_mps > 1.5
    camera.ego_path = np.array([[0., 0.], [40., 0.]])
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "vision"
    assert supervisor.step(DrivingState(**fields), (0., 1.), .05).target_speed_mps > 1.5


@pytest.mark.parametrize("path", [None, [[10., 0.], [20., 0.]],
                                 [[0., 2.5], [10., 2.5]], [[0., 0.], [10., 12.]]])
def test_expired_lane_rejects_unavailable_or_misaligned_route(path):
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    assert provider.predict(camera).driving["observations_valid"]
    provider.road = make_provider(low_confidence=True).road
    camera.timestamp += 2.05
    camera.ego_path = np.array(path) if path is not None else None
    fields = provider.predict(camera).driving
    assert not fields["observations_valid"] and fields["lane_source"] == "unavailable"
    assert fields["observation_invalid_reason"] == "lane_low_confidence"


def test_spawn_without_lane_paint_uses_aligned_route():
    # Regression: lane paint unseen from the first frame left no estimate to age,
    # so the route fallback never ran and the supervisor braked until inactivity timeout.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.road = lambda _: dict(lane_center=None, crosswalks=[])
    camera.ego_path = np.array([[0., -.3], [20., -.3]])
    fields = provider.predict(camera).driving
    assert fields["observations_valid"] and fields["lane_source"] == "route_fallback"
    assert fields["lane_error_m"] == pytest.approx(.3)
    provider.reset()
    camera.ego_path = np.array([[0., 2.5], [20., 2.5]])  # route too far off: still invalid
    fields = provider.predict(camera).driving
    assert not fields["observations_valid"]
    assert fields["observation_invalid_reason"] == "lane_geometry_unavailable"


def test_lane_hold_follows_ego_motion_relative_to_route():
    # Regression (experiment_28 eval2): a frozen held estimate kept the policy
    # correcting for 2 s and drove the car 2.3 m across the lane and off the road.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    camera.ego_path = np.array([[0., -.5], [20., -.5]])
    assert provider.predict(camera).driving["lane_error_m"] == pytest.approx(.7)
    provider.road = lambda _: dict(lane_center=None, crosswalks=[])
    camera.timestamp += 1.
    camera.ego_path = np.array([[0., -1.5], [20., -1.5]])  # ego moved 1 m further off
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "held"
    assert fields["lane_error_m"] == pytest.approx(1.7)
    assert fields["heading_error_rad"] == pytest.approx(0.)
    camera.ego_path = None  # no route: hold the last value unchanged
    assert provider.predict(camera).driving["lane_error_m"] == pytest.approx(.7)


def test_expired_lane_recovers_a_drifted_car_instead_of_braking_forever():
    # Regression (experiment_28 eval2): the car drifted 1.3-1.8 m and 31 degrees
    # while vision was lost, the 1 m / 15 degree gate rejected the route and the
    # supervisor braked until the inactivity timeout.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.predict(camera)
    provider.road = lambda _: dict(lane_center=None, crosswalks=[])
    camera.timestamp += 2.05
    angle = math.radians(31.)
    camera.ego_path = np.array([[0., -1.8], [20.*math.cos(angle), -1.8-20.*math.sin(angle)]])
    fields = provider.predict(camera).driving
    assert fields["observations_valid"] and fields["lane_source"] == "route_fallback"
    assert fields["lane_error_m"] == pytest.approx(1.8*math.cos(angle))
    assert fields["heading_error_rad"] == pytest.approx(angle)


def lane_road(lane_error):
    """Road model whose straight centre line puts the ego `lane_error` metres off it."""
    center_line = [[639.5-640.*lane_error/(x-1.5), 359.5+640.*2.4/(x-1.5)] for x in (6., 10., 20.)]
    return lambda _: dict(lane_center=dict(center_line=center_line, low_confidence=False), crosswalks=[])


def test_vision_lane_far_from_route_is_rejected_until_it_agrees_again():
    # Regression (experiment_28 eval10 ep2): vision locked onto the neighbouring lane
    # (+3.7 m vs true -0.4 m), the policy steered toward it and left the road.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    camera.ego_path = np.array([[0., -.5], [20., -.5]])  # route: ego 0.5 m off centre
    assert provider.predict(camera).driving["lane_source"] == "vision"
    provider.road = lane_road(4.2)  # one lane width off
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "held" and fields["lane_error_m"] == pytest.approx(.7)
    camera.timestamp += 2.05  # still wrong after the hold: navigation stands in
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "route_fallback" and fields["lane_error_m"] == pytest.approx(.5)
    provider.road = lane_road(.9)
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.9)


def test_vision_lane_jump_without_route_falls_back_to_hold():
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.predict(camera)
    provider.road = lane_road(-3.)
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "held" and fields["lane_error_m"] == pytest.approx(.7)
    provider.road = lane_road(1.5)  # an ordinary correction is accepted
    camera.timestamp += .05
    assert provider.predict(camera).driving["lane_error_m"] == pytest.approx(1.5)


def test_route_not_a_wrong_hold_decides_whether_vision_is_plausible():
    # A hold seeded by a wrong reading must not reject the correct one that follows.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.road = lane_road(4.2)
    provider.predict(camera)  # no route yet: nothing to check against
    provider.road = lane_road(.7)
    camera.timestamp += .05
    camera.ego_path = np.array([[0., -.6], [20., -.6]])
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.7)


def test_route_fallback_preserves_other_invalid_observation_checks():
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.predict(camera)
    camera.timestamp += 2.05
    camera.ego_path = np.array([[0., 0.], [20., 0.]])
    provider.road = lambda _: dict(lane_center=None, crosswalks=[])
    provider.detector = make_provider((4,), boxes=[[600., 500., 680., 720.]]).detector
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "route_fallback" and not fields["observations_valid"]
    assert fields["observation_invalid_reason"] == "object_position_unavailable"
    assert SafetySupervisor().step(DrivingState(**fields), (0., 1.), .05).action == (0., -1.)

    camera.rgb = None
    fields = provider.predict(camera).driving
    assert not fields["observations_valid"] and fields["lane_source"] == "unavailable"
    assert fields["observation_invalid_reason"] == "rgb_unavailable"


def test_lane_debug_records_fit_points_and_hold():
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    camera.ego_path = np.array([[0., -.5], [20., -.5]])
    provider.predict(camera)
    debug = provider.lane_debug
    assert debug["source"] == "vision" and debug["fit"][0] == pytest.approx(.7)
    assert len(debug["points_ego_m"]) == 3 and debug["route"][0] == pytest.approx(.5)
    provider.road = lane_road(4.2)  # rejected neighbouring lane: raw fit kept, hold used
    camera.timestamp += .05
    provider.predict(camera)
    debug = provider.lane_debug
    assert debug["source"] == "held" and debug["fit"][0] == pytest.approx(4.2)
    assert debug["lane_age_s"] == pytest.approx(.05) and debug["held"][0] == pytest.approx(.7)
    json.dumps(debug, allow_nan=False)


def road_from_points(points):
    """Road model whose projected centre line is exactly these ego-frame (x, y) points."""
    center_line = [[639.5+640.*y/(x-1.5), 359.5+640.*2.4/(x-1.5)] for x, y in points]
    return lambda _: dict(lane_center=dict(center_line=center_line, low_confidence=False), crosswalks=[])


def test_vision_line_is_used_when_route_disagrees_at_bend_entry():
    # Captured at Town02 88->61 frame 378053 (town_transfer E01, left bend entry): the
    # centre line leans right (heading -0.27) while the planned path turns left (+0.68).
    # The route no longer overrides vision here; this case once ended in a collision.
    captured = [[9.161, 2.587], [8.195, 2.32], [7.446, 2.112], [6.848, 1.947], [6.359, 1.811], [5.773, 1.649]]
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.road = road_from_points(captured)
    heading = .68
    camera.ego_path = np.array([[0., -.37], [20.*math.cos(heading), -.37-20.*math.sin(heading)]])
    fields = provider.predict(camera).driving
    assert provider.lane_debug["fit"][1] == pytest.approx(-.27, abs=.01)
    assert fields["lane_source"] == "vision" and fields["observations_valid"]
    assert fields["heading_error_rad"] == pytest.approx(-.27, abs=.01)
    # With vision lost, the hold carries the last estimate instead of the route.
    provider.road = lambda _: dict(lane_center=dict(center_line=[], low_confidence=True), crosswalks=[])
    camera.timestamp += .05
    assert provider.predict(camera).driving["lane_source"] == "held"


def test_vision_agreeing_with_route_is_kept():
    # Straight before the captured jump (frame 377997): small offsets stay with vision.
    provider, camera = make_provider(), bundle()
    provider.lane_hold_s = 2.
    provider.road = lane_road(.3)
    camera.ego_path = np.array([[0., 0.], [20., 0.]])
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.3)
    provider.road = lane_road(.98)  # captured 0.98 m jump against a centred route
    camera.timestamp += .05
    fields = provider.predict(camera).driving
    assert fields["lane_source"] == "vision" and fields["lane_error_m"] == pytest.approx(.98)
    provider.road = lane_road(2.3)  # over 2 m from the route: still rejected as a neighbouring lane
    camera.timestamp += .05
    assert provider.predict(camera).driving["lane_source"] == "held"


@pytest.mark.parametrize("speed, spacing", [(22., 15.5), (22., 22.), (6., 15.5), (6., 22.)])
def test_back_to_back_signs_are_each_applied(speed, spacing):
    """Signs closer together than their view range never leave the view (eval12 mixed_3)."""
    from dreamerv3_attention.speed_sign import SpeedLimitTracker
    tracker = SpeedLimitTracker()
    tracker.limit_kmh, tracker.confidence = 90, 1.
    signs, x, t, seen = [(0., 60), (spacing, 30)], -30., 0., []
    while x < spacing + 25.:
        visible = [(pos - x, value) for pos, value in signs if 3. <= pos - x <= 25.]
        near = min(visible) if visible else None
        read = near is not None and tracker.should_read(t)
        limit = tracker.update(t, near[0] if near else None, near[1] if read else None, read)[0]
        if not seen or seen[-1] != limit:
            seen.append(limit)
        x, t = x + speed * .05, t + .05
    tracker.update(t + .5, None)
    assert seen[:2] == [90, 60] and tracker.limit_kmh == 30


def test_sign_debug_reports_why_a_sign_was_not_read(monkeypatch):
    from dreamerv3_attention import vision
    from dreamerv3_attention.speed_sign import SpeedLimitTracker
    monkeypatch.setattr(vision, "read_speed_limit", lambda reader, crop: (None, 0.))
    p = make_provider((6,), boxes=[light_box(3., 10.)])
    p.sign_reader, p.speed_limits = object(), SpeedLimitTracker()
    b = depth_bundle(10.)
    b.timestamp = 0.
    p.predict(b)
    debug = p.sign_debug
    assert debug["detected"] == 1 and debug["reason"] == "ocr_failed" and debug["ocr_value"] is None
    assert 8. < debug["distance_m"] < 15. and debug["limit_kmh"] is None
    json.dumps(debug, allow_nan=False)


def road_depth(scale):
    """Depth image of a flat road seen by the test camera, scaled (scale != 1 tilts the road)."""
    rows = np.arange(720, dtype=np.float32)[:, None]
    flat = np.where(rows > 360., 640.*2.4/np.maximum(rows-359.5, 1e-3), 1000.)
    return np.repeat(np.where(rows > 360., flat*scale, 1000.), 1280, axis=1).astype(np.float32)


def test_lane_points_use_measured_depth_and_fall_back_off_the_road():
    provider, camera = make_provider(), bundle()
    camera.depth, camera.valid["depth"] = road_depth(.9), True
    # Every lane pixel measures 0.9x the flat-road depth: points are 10% nearer and up to 0.24 m high.
    assert provider.predict(camera).driving["lane_error_m"] == pytest.approx(.63, abs=.01)
    # A car over the lane (depth 3 m) is not road, so those pixels use the flat-road projection.
    camera.depth = np.full((720, 1280), 3., np.float32)
    assert provider.predict(camera).driving["lane_error_m"] == pytest.approx(.7)


def test_object_velocity_comes_from_tracked_depth():
    provider, camera = make_provider((0,)), bundle()
    provider.object_tracker = ObjectTracker()
    camera.valid["depth"], camera.ego_pose = True, (0., 0., 0., 0.)
    for i in range(6):  # a vehicle 20 m ahead closing at 10 m/s on a stationary ego
        camera.timestamp, camera.depth = 2.1 + .05*i, np.full((720, 1280), 20. - .5*i, np.float32)
        objects = provider.predict(camera).objects
    np.testing.assert_allclose(objects[0, 3:6]*30., [-10., 0., 0.], atol=.01)

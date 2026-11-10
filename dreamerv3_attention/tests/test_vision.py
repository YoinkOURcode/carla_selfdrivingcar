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


def make_provider(classes=(), *, low_confidence=False):
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
    boxes = NS(xyxy=torch.tensor([[600., 400., 680., 600.]]*len(classes)).reshape(-1, 4),
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


def test_lane_metrics_and_detector_rows_use_rgb_only():
    p, b = make_provider((0, 4, 5, 6)), bundle()
    b.depth = np.full((720, 1280), .01)
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

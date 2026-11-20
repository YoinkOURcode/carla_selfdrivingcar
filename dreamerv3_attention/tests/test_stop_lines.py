"""Offline measured-stripe geometry with calibrated synthetic RGB/depth."""

import numpy as np
import pytest

from dreamerv3_attention.stop_lines import extract_stop_lines, path_intersection, project_to_path


def scene(x=12., lateral=0., length=3.5, longitudinal=False, road_height=0., slope=(0., 0.)):
    calibration = dict(intrinsics_px=dict(fx=320., fy=320., cx=319.5, cy=199.5),
                       extrinsics_ego=dict(location_m=(1.5, 0., 2.4),
                                           rotation_pitch_yaw_roll_deg=(0., 0., 0.)))
    v, u = np.indices((480, 640))
    denominator = (v-199.5)/320. + slope[0] + slope[1]*(u-319.5)/320.
    axial = np.divide(2.4-road_height-1.5*slope[0], denominator,
                      out=np.full(v.shape, np.nan), where=(v > 199.5) & (denominator > 0.))
    forward, side = axial + 1.5, axial * (u-319.5)/320.
    if longitudinal:
        paint = (forward >= 5.) & (forward <= 25.) & (np.abs(side-lateral) < .10)
    else:
        paint = (np.abs(forward-x) <= .22) & (np.abs(side-lateral) <= length/2.)
    bgr = np.full((*v.shape, 3), 45, np.uint8)
    bgr[paint] = 245
    return bgr, axial, calibration, np.array([[3., 0.], [35., 0.]]), paint


@pytest.mark.parametrize("distance", [8., 12., 18., 25.])
def test_measured_stop_stripe_matches_known_ground_geometry(distance):
    bgr, depth, calibration, path, _ = scene(x=distance)
    candidates = extract_stop_lines(bgr, depth, calibration, path)
    assert len(candidates) == 1
    line = candidates[0]
    assert line["points_ego"].shape == (2, 3)
    assert line["pixels"].shape == (2, 2)
    station, xy = path_intersection(line["points_ego"], path)
    assert station == pytest.approx(distance-3., abs=.3)
    assert xy == pytest.approx([distance, 0.], abs=.3)
    assert line["points_ego"][:, 2] == pytest.approx([0., 0.], abs=1e-10)
    assert np.min(line["points_ego"][:, 1]) < -1.5
    assert np.max(line["points_ego"][:, 1]) > 1.5
    assert 0. < line["confidence"] <= 1.


def test_curved_path_intersection_uses_segment_arc_length():
    path = [[0., 0.], [10., 1.], [20., 3.], [25., 8.]]
    station, xy = path_intersection([[16., -1., 0.], [16., 5., 0.]], path)
    assert station == pytest.approx(np.hypot(10., 1.) + .6*np.hypot(10., 2.))
    assert xy == pytest.approx([16., 2.2])
    assert path_intersection([[16., 4.], [16., 6.]], path) is None


def test_path_projection_starts_at_path_origin_and_retains_signed_lateral():
    assert project_to_path([12., -2.], [[4., 1.], [20., 1.]]) == pytest.approx((8., -3.))
    assert project_to_path([-2., 2.], [[4., 1.], [20., 1.]]) == pytest.approx((0., 1.))
    station, xy = path_intersection([[-2., -4.], [-2., 2.]], [[-5., -1.], [5., -1.]])
    assert station == pytest.approx(3.)
    assert xy == pytest.approx([-2., -1.])
    assert project_to_path([12., 2.], [[4., 1.], [4., 1.], [20., 1.]]) == pytest.approx((8., 1.))


@pytest.mark.parametrize("path", [[], [[0., 0.]], [[1., 1.], [1., 1.]], [[0., 0.], [np.nan, 0.]]])
def test_invalid_or_degenerate_path_is_unknown(path):
    assert project_to_path([1., 0.], path) is None
    assert path_intersection([[1., -1.], [1., 1.]], path) is None


def test_adjacent_lane_stripe_does_not_cross_planned_path():
    bgr, depth, calibration, path, _ = scene(lateral=3.5)
    assert extract_stop_lines(bgr, depth, calibration, path) == []


def test_longitudinal_lane_boundary_is_rejected_even_if_path_crosses_it():
    bgr, depth, calibration, _, _ = scene(lateral=1.75, longitudinal=True)
    path = [[3., 0.], [30., 3.]]
    assert extract_stop_lines(bgr, depth, calibration, path) == []


@pytest.mark.parametrize("failure", ["empty", "no_depth", "nan", "saturated", "zero", "sparse", "elevated"])
def test_missing_invalid_or_above_ground_evidence_remains_unknown(failure):
    bgr, depth, calibration, path, paint = scene()
    if failure == "empty":
        bgr[:] = 45
    elif failure == "no_depth":
        depth = None
    elif failure in ("nan", "saturated", "zero"):
        depth[paint] = dict(nan=np.nan, saturated=1000., zero=0.)[failure]
    elif failure == "sparse":
        v, u = np.nonzero(paint)
        depth[v[::2], u[::2]] = np.nan
    elif failure == "elevated":
        depth[paint] *= .6
    assert extract_stop_lines(bgr, depth, calibration, path) == []


def test_crosswalk_box_excludes_complete_stripe_without_clipping_new_edge():
    bgr, depth, calibration, path, paint = scene()
    v, u = np.nonzero(paint)
    box = [u.min()+10., v.min(), u.max()-10., v.max()]
    assert extract_stop_lines(bgr, depth, calibration, path, crosswalk_boxes=[box]) == []


def test_depth_must_be_pixel_aligned_and_camera_level():
    bgr, depth, calibration, path, _ = scene()
    assert extract_stop_lines(bgr, depth[:-1], calibration, path) == []
    calibration["extrinsics_ego"]["rotation_pitch_yaw_roll_deg"] = (5., 0., 0.)
    with pytest.raises(ValueError, match="level front camera"):
        extract_stop_lines(bgr, depth, calibration, path)


@pytest.mark.parametrize("road_height,slope", [(-.3, (0., 0.)), (-.3, (.035, .025)), (-.6, (-.025, .02))])
def test_measured_road_plane_handles_ego_height_and_modest_slopes(road_height, slope):
    bgr, depth, calibration, path, _ = scene(road_height=road_height, slope=slope)
    candidates = extract_stop_lines(bgr, depth, calibration, path)
    assert len(candidates) == 1
    endpoints = candidates[0]["points_ego"]
    assert endpoints[:, 0] == pytest.approx([12., 12.], abs=.3)
    expected_height = road_height + endpoints[:, :2] @ np.asarray(slope)
    np.testing.assert_allclose(endpoints[:, 2], expected_height, atol=.015)
    assert path_intersection(endpoints, path)[0] == pytest.approx(9., abs=.3)


def test_raised_white_object_is_rejected_against_translated_sloped_road():
    bgr, depth, calibration, path, paint = scene(road_height=-.3, slope=(.03, .02))
    depth[paint] *= .6
    assert extract_stop_lines(bgr, depth, calibration, path) == []


def test_raised_white_object_at_ego_zero_is_not_mistaken_for_lower_road():
    bgr, depth, calibration, path, paint = scene(road_height=-.3)
    # Measured paint-shaped object is z=0; the actual road is z=-0.3.
    depth[paint] *= 2.4/2.7
    assert extract_stop_lines(bgr, depth, calibration, path) == []


def test_stripe_depth_alone_cannot_establish_a_road_plane():
    bgr, depth, calibration, path, paint = scene(road_height=-.3)
    depth[~paint] = np.nan
    assert extract_stop_lines(bgr, depth, calibration, path) == []


def test_local_road_plane_is_robust_to_nonwhite_obstacle_depth():
    bgr, depth, calibration, path, _ = scene(road_height=-.3, slope=(.03, .02))
    # A dark obstacle obscures part of the road used for plane estimation.
    depth[330:410, 285:325] *= .55
    candidates = extract_stop_lines(bgr, depth, calibration, path)
    assert len(candidates) == 1
    assert path_intersection(candidates[0]["points_ego"], path)[0] == pytest.approx(9., abs=.3)

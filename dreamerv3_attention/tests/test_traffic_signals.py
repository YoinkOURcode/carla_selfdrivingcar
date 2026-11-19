import numpy as np
import pytest

from dreamerv3_attention.observations import STOP_LINE_TOKEN, TRAFFIC_LIGHT_TOKEN
from dreamerv3_attention.traffic_signals import SignalTracker, stop_clearance, transform_points


PATH = np.array([[0., 0.], [40., 0.]])
POSE = (0., 0., .3, 0.)


def stripe(x=12., y=0.):
    return dict(points_ego=np.array([[x, y-1.75, -.3], [x, y+1.75, -.3]]), confidence=.95)


def lamp(colour=1, x=19., y=2.):
    return dict(point_ego=np.array([x, y, 2.]), class_id=colour, confidence=.9)


def update(tracker, *, t=0., lines=None, lights=None, pose=POSE, path=PATH, bumper=2.):
    return tracker.update(t, lines if lines is not None else [stripe()],
                          lights if lights is not None else [lamp()], path, bumper, pose)


def test_clearance_measures_line_from_actual_bumper_not_lamp_offset():
    for head_x in (14., 19., 28.):
        fields, rows = update(SignalTracker(), lights=[lamp(x=head_x)])
        assert fields['red_light'] and fields['red_light_distance_m'] == pytest.approx(10.)
        assert len(rows) == 2 and rows[0, 15] == STOP_LINE_TOKEN
        assert rows[1, 15] == TRAFFIC_LIGHT_TOKEN and rows[1, 11] == pytest.approx(1./3.)
    assert update(SignalTracker(), bumper=3.)[0]['red_light_distance_m'] == pytest.approx(9.)


def test_projection_uses_arc_length_and_bumper_on_curved_path():
    path = np.array([[0., 0.], [10., 0.], [10., 10.]])
    line = np.array([[8., 5., 0.], [12., 5., 0.]])
    assert stop_clearance(line, path, 2.) == pytest.approx(13.)
    assert stop_clearance(stripe()['points_ego'], [[6., 0.], [40., 0.]], 2.5) == pytest.approx(9.5)


def test_other_lane_red_does_not_override_our_green_and_candidates_survive():
    fields, rows = update(SignalTracker(), lights=[lamp(1, y=7.), lamp(3, y=1.)])
    assert not fields['red_light'] and not fields['yellow_light']
    heads = rows[rows[:, 15] == TRAFFIC_LIGHT_TOKEN]
    assert len(heads) == 2
    assert set(np.round(heads[:, 11]*3).astype(int)) == {1, 3}
    assert heads[heads[:, 11] > .9, 5].item() > 0.
    assert heads[heads[:, 11] < .4, 5].item() == 0.


def test_adjacent_lane_stripe_never_supplies_our_stop_distance():
    fields, _ = update(SignalTracker(), lines=[stripe(y=5.)])
    assert fields['red_light'] and fields['red_light_distance_m'] is None


def test_odometry_updates_distance_when_boundary_and_head_are_occluded():
    tracker = SignalTracker()
    update(tracker)
    fields, _ = update(tracker, t=.5, lines=[], lights=[], pose=(4., 0., .3, 0.))
    assert fields['red_light'] and fields['red_light_distance_m'] == pytest.approx(6.)


def test_old_geometry_becomes_unknown_without_releasing_red():
    tracker = SignalTracker(hold_s=1.)
    update(tracker)
    fields, rows = update(tracker, t=1.1, lines=[], lights=[], pose=(4., 0., .3, 0.))
    assert fields['red_light'] and fields['red_light_distance_m'] is None
    assert not len(rows)


def test_missing_head_without_any_stripe_keeps_red_with_unknown_range():
    tracker = SignalTracker()
    assert update(tracker, lines=[])[0]['red_light']
    for timestamp in (.1, 3.):
        fields, _ = update(tracker, t=timestamp, lines=[], lights=[])
        assert fields['red_light'] and fields['red_light_distance_m'] is None


def test_passed_boundary_and_still_visible_far_side_lamp_are_retired():
    tracker = SignalTracker()
    update(tracker)
    fields, _ = update(tracker, t=.5, lines=[], lights=[lamp(x=8.)], pose=(11., 0., .3, 0.))
    assert not fields['red_light'] and fields['red_light_distance_m'] is None


def test_passed_lamp_without_depth_is_matched_by_predicted_camera_bearing():
    tracker = SignalTracker()
    update(tracker)
    relative_ray = np.array([19.-11.-1.5, 2., 2.-2.4])
    unknown = dict(point_ego=None, class_id=1, confidence=.9, camera_location=np.array([1.5,0.,2.4]),
                   bearing=np.array([relative_ray[1]/relative_ray[0], -relative_ray[2]/relative_ray[0]]),
                   bearing_radius=np.array([.01,.01]))
    fields, _ = update(tracker, t=.5, lines=[], lights=[unknown], pose=(11.,0.,.3,0.))
    assert not fields['red_light']


def test_unknown_depth_red_is_not_cleared_by_known_green():
    tracker = SignalTracker()
    unknown = dict(point_ego=None, class_id=1, confidence=.8)
    fields, _ = update(tracker, lights=[lamp(3), unknown])
    assert fields['red_light']
    assert update(tracker, t=.2, lights=[], lines=[])[0]['red_light']


def test_other_lamp_green_cannot_release_tracked_red():
    tracker = SignalTracker()
    update(tracker)
    fields, _ = update(tracker, t=.5, lines=[], lights=[lamp(3, x=30., y=-2.)])
    assert fields['red_light'] and fields['red_light_distance_m'] == pytest.approx(10.)
    fields, _ = update(tracker, t=.6, lines=[], lights=[lamp(3)])
    assert not fields['red_light'] and not fields['yellow_light']


def test_stale_lock_off_the_turning_path_lets_own_green_release_red():
    # Experiment 28 episode 6228: ~90 m out the tracker locked onto another
    # approach's red head. Waiting 2.45 m from our stop, where the route turns,
    # that head lies off the path; our own head (6.5 m ahead, 2.6 m aside) is green.
    tracker = SignalTracker()
    update(tracker, lines=[], lights=[lamp(1, x=90., y=3.)], path=np.array([[0., 0.], [100., 0.]]))
    turn = np.array([[0., 0.], [6., 0.], [14., -8.], [14., -40.]])
    for t in (20., 20.1):
        fields, _ = update(tracker, t=t, lines=[stripe(x=4.45)], pose=(66., 0., .3, 0.), path=turn,
                           lights=[lamp(3, x=6.5, y=2.6)])
        assert not fields['red_light'] and not fields['yellow_light']


def test_far_side_lock_past_the_stop_yields_to_own_head():
    # Experiment 28 episode 6196: the lock sat on a head across the junction,
    # on the path but ~20 m past our stop; our own head stands just past the line.
    tracker = SignalTracker()
    update(tracker, lines=[], lights=[lamp(1, x=91., y=2.2)], path=np.array([[0., 0.], [100., 0.]]))
    for t in (20., 20.1):
        fields, _ = update(tracker, t=t, lines=[stripe(x=4.9)], pose=(66., 0., .3, 0.),
                           lights=[lamp(3, x=6.5, y=2.2)])
        assert not fields['red_light'] and not fields['yellow_light']


def test_slightly_better_fitting_green_cannot_release_tracked_red():
    tracker = SignalTracker()
    update(tracker, lights=[lamp(1, x=19., y=2.)])
    fields, _ = update(tracker, t=.5, lights=[lamp(3, x=14., y=-1.)])
    assert fields['red_light']


def test_reversed_timestamp_and_explicit_reset_discard_episode_state():
    tracker = SignalTracker()
    update(tracker, t=5.)
    assert not update(tracker, t=0., lines=[], lights=[])[0]['red_light']
    update(tracker)
    tracker.reset()
    assert not update(tracker, lines=[], lights=[])[0]['red_light']


@pytest.mark.parametrize('bumper', [None, float('nan'), -1.])
def test_unknown_vehicle_geometry_does_not_invent_distance(bumper):
    fields, _ = update(SignalTracker(), bumper=bumper)
    assert fields['red_light'] and fields['red_light_distance_m'] is None


def test_missing_odometry_prevents_distance_hold():
    tracker = SignalTracker()
    update(tracker)
    fields, _ = update(tracker, t=.5, pose=None, lines=[], lights=[lamp()])
    assert fields['red_light'] and fields['red_light_distance_m'] is None


def test_red_yellow_conflict_is_conservative_and_unknown_depth_is_not_a_range():
    fields, _ = update(SignalTracker(), lights=[lamp(2), lamp(1)])
    assert fields['red_light'] and not fields['yellow_light']
    fields, _ = update(SignalTracker(), lines=[], lights=[dict(point_ego=None,class_id=2,confidence=.8)])
    assert fields['yellow_light'] and fields['yellow_light_distance_m'] is None
    assert fields['yellow_light_confidence'] == pytest.approx(.8)


def test_pose_transform_round_trip_includes_yaw_and_height():
    points = np.array([[12., -1., -.3], [12., 1., -.3]])
    pose = (100., -20., 2., np.pi/2.)
    np.testing.assert_allclose(transform_points(transform_points(points, pose), pose, True), points, atol=1e-10)


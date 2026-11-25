import math

import numpy as np
import pytest

from dreamerv3_attention.object_tracker import ObjectTracker


def run(tracker, frames):
    """frames: [(ego_pose, points, pedestrian)] at 0.05 s; returns the last output."""
    for i, (pose, points, walkers) in enumerate(frames):
        out = tracker.update(.05*i, pose, points, walkers)
    return out


def test_stationary_object_seen_from_moving_ego_closes_at_ego_speed():
    frames = [((.5*i, 0., 0., 0.), [np.array([20. - .5*i, 0., 0.])], [False]) for i in range(6)]
    np.testing.assert_allclose(run(ObjectTracker(), frames)[0], [-10., 0., 0.], atol=1e-6)


def test_lead_at_same_speed_has_zero_relative_velocity_in_any_heading():
    yaw = math.pi/2  # driving along world +y
    frames = [((0., .5*i, 0., yaw), [np.array([15., 0., 0.])], [False]) for i in range(6)]
    np.testing.assert_allclose(run(ObjectTracker(), frames)[0], 0., atol=1e-6)


def test_crossing_pedestrian_velocity_is_in_ego_frame():
    yaw = math.pi/2
    # Ego stopped, facing world +y; a pedestrian walks along world -x = ego +y (right).
    frames = [((0., 0., 0., yaw), [np.array([10., .06*i, 0.])], [True]) for i in range(6)]
    np.testing.assert_allclose(run(ObjectTracker(), frames)[0], [0., 1.2, 0.], atol=1e-6)


def test_zero_until_enough_history_and_kinds_never_match():
    tracker = ObjectTracker()
    frames = [((0., 0., 0., 0.), [np.array([10. - .5*i, 0., 0.])], [False]) for i in range(4)]
    assert not run(tracker, frames)[0].any()
    # A pedestrian at the same spot starts its own track: no velocity yet.
    assert not tracker.update(.2, (0., 0., 0., 0.), [np.array([8., 0., 0.])], [True])[0].any()


def test_stale_tracks_are_forgotten_and_reset_clears():
    tracker = ObjectTracker()
    run(tracker, [((0., 0., 0., 0.), [np.array([10., 0., 0.])], [False]) for _ in range(3)])
    tracker.update(1.0, (0., 0., 0., 0.), [], [])
    assert tracker.tracks == []
    tracker.update(1.05, (0., 0., 0., 0.), [np.array([10., 0., 0.])], [False])
    tracker.reset()
    assert tracker.tracks == [] and tracker.ego == []

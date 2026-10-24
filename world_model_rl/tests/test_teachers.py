"""Counterfactual training labels use current geometry, controls and observed objects."""

from types import SimpleNamespace

import numpy as np
import pytest

from world_model_rl.schema import CONTINUOUS, OBJECT_OFFSET, SLOTS
from world_model_rl.teachers import geometry_actions
from world_model_rl.sac import Demonstrations
from world_model_rl.model import RSSM
from world_model_rl.schema import Normalizer
from world_model_rl.tests.test_pipeline import cfg, make_episode


def observation_sequence():
    c = np.zeros((2, CONTINUOUS), np.float32)
    c[:, 0] = 8
    c[:, 10:14] = [1, 14, 12, 100]
    c[:, 20:OBJECT_OFFSET] = np.array([[5, 0], [10, 0], [20, 0], [30, 0], [40, 0]]).ravel()
    return {'continuous': c, 'present': np.zeros((2, SLOTS), bool),
            'classes': np.zeros((2, SLOTS), np.int64), 'controls': np.zeros((2, 1), bool),
            'light': np.zeros(2, np.int64)}


def test_geometric_teacher_turns_in_the_correct_direction_and_slows_for_curves():
    obs = observation_sequence()
    straight = geometry_actions(obs)[0]
    assert straight[0] == 0 and straight[1] > 0 and straight[2] == 0
    points = np.array([[4, 3], [7, 8], [8, 18], [8, 28], [8, 38]], np.float32)
    obs['continuous'][:, 20:OBJECT_OFFSET] = points.ravel()
    left = geometry_actions(obs)[0]
    assert left[0] < 0 and left[1] == 0 and left[2] > 0
    points[:, 1] *= -1
    obs['continuous'][:, 20:OBJECT_OFFSET] = points.ravel()
    assert geometry_actions(obs)[0, 0] > 0


def test_geometric_teacher_brakes_for_relevant_lights_and_present_lead_vehicles():
    obs = observation_sequence()
    obs['controls'][:, 0] = True
    obs['continuous'][:, 14] = 5
    obs['light'][:] = 2
    assert geometry_actions(obs)[0, 2] > 0
    obs = observation_sequence()
    obs['continuous'][:, OBJECT_OFFSET:OBJECT_OFFSET + 7] = [8, 0, -8, 0, 2, 1, 1]
    assert geometry_actions(obs)[0, 2] == 0  # Invalid/absent slots carry no hazard.
    obs['present'][:, 0] = True
    assert geometry_actions(obs)[0, 2] > 0


def test_geometric_teacher_does_not_read_future_observations_or_outcomes():
    obs = observation_sequence()
    expected = geometry_actions(obs)
    obs['continuous'][-1] = 10000
    obs['present'][-1] = True
    obs['light'][-1] = 2
    obs['events'] = np.ones((1, 9))
    obs['rewards'] = np.full(1, 10000)
    np.testing.assert_equal(geometry_actions(obs), expected)


def test_calibrated_teacher_matches_tesla_steering_geometry():
    obs = observation_sequence()
    points = np.array([[4, 3], [7, 8], [8, 18], [8, 28], [8, 38]], np.float32)
    obs['continuous'][:, 20:OBJECT_OFFSET] = points.ravel()
    steer = geometry_actions(obs, calibrated=True)[0, 0]
    expected_lookahead_point = .6 * points[0] + .4 * points[1]  # 7 m at 8 m/s.
    desired_curvature = 2 * expected_lookahead_point[1] / np.square(expected_lookahead_point).sum()
    effective_max_angle = np.deg2rad(70) * .878  # Curve interpolated at 28.8 km/h.
    achieved_curvature = -np.tan(steer * effective_max_angle) / 3.
    assert achieved_curvature == pytest.approx(desired_curvature, rel=1e-5)
    assert abs(steer) < abs(geometry_actions(obs)[0, 0])


@pytest.mark.parametrize('hazard', ['red_light', 'lead_vehicle'])
def test_stop_hold_never_requests_throttle_at_zero_target_speed(hazard):
    obs = observation_sequence()
    obs['continuous'][:, 0] = 0
    if hazard == 'red_light':
        obs['controls'][:, 0] = True
        obs['light'][:] = 2
        obs['continuous'][:, 14] = 1
    else:
        obs['continuous'][:, OBJECT_OFFSET:OBJECT_OFFSET + 7] = [6, 0, 0, 0, 2, 1, 1]
        obs['present'][:, 0] = True
    action = geometry_actions(obs, calibrated=True, hold_stops=True)[0]
    assert action[1] == 0 and action[2] >= .35 - 1e-6  # Float32 action storage.
    # Clearing the hazard should release the hold rather than immobilize the car.
    obs['controls'][:] = False
    obs['present'][:] = False
    action = geometry_actions(obs, calibrated=True, hold_stops=True)[0]
    assert action[1] > 0 and action[2] == 0


@pytest.mark.parametrize('split', ['validation', 'test'])
def test_geometric_label_training_still_rejects_heldout_states(split):
    ep = SimpleNamespace(metadata={'split': split, 'collector': 'sac'})
    with pytest.raises(ValueError, match='training split'):
        Demonstrations([ep], None, None, None, 1, teacher='geometry_v1')


def test_traffic_sampling_practises_moving_stops_without_losing_waits_or_cruising(cfg):
    ep = make_episode(cfg, steps=100)
    ep.arrays['present'][:] = False
    ep.arrays['continuous'][:, 0] = 4
    ep.arrays['continuous'][:, 20:OBJECT_OFFSET] = np.array(
        [[5, 0], [10, 0], [20, 0], [30, 0], [40, 0]]).ravel()
    ep.arrays['controls'][:2, 0] = True
    ep.arrays['light'][:2] = 2
    ep.arrays['continuous'][:2, 14] = [1, 4]
    ep.arrays['continuous'][:2, 0] = [0, 8]
    # A final observation has no associated action and must not enter sampling.
    ep.arrays['controls'][-1, 0] = True
    ep.arrays['light'][-1] = 2
    norm = Normalizer.fit([ep])
    demos = Demonstrations([ep], RSSM(cfg.model), norm, 'cpu', 1,
                           teacher='geometry_v3', traffic_fraction=.8)
    actions = demos.sample(1000, np.random.default_rng(2))['actions']
    assert np.count_nonzero(actions[:, 2] >= .79) >= 600  # Moving approach.
    assert np.count_nonzero(np.isclose(actions[:, 2], .35)) >= 200  # Hold/release coverage.
    assert np.count_nonzero(actions[:, 1] > 0) >= 90  # Uniform remainder retains cruising.
    assert demos.episode_ids == [ep.metadata['episode_id']]


def test_traffic_sampling_handles_training_data_without_lights(cfg):
    ep = make_episode(cfg)
    demos = Demonstrations([ep], RSSM(cfg.model), Normalizer.fit([ep]), 'cpu', 1,
                           teacher='geometry_v3', traffic_fraction=1)
    assert not len(demos.traffic_moving) and not len(demos.traffic_waiting)
    assert np.isfinite(demos.sample(32, np.random.default_rng(3))['actions']).all()

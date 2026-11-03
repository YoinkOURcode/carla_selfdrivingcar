"""Combined hazards select the lowest independent speed cap."""

from dataclasses import replace
from itertools import combinations

import pytest

from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor, compute_target_speed
from dreamerv3_attention.tests.test_controller import observation


SOFT_HAZARDS = [
    {"rain": 1.}, {"wetness": .9}, {"night": 1.},
    {"visibility_m": 20.}, {"pedestrian_risk": 1.},
    {"child_pedestrian_risk": 1.}, {"vehicle_ahead": True},
]


@pytest.mark.parametrize("hazards", [*combinations(SOFT_HAZARDS, 2), SOFT_HAZARDS])
@pytest.mark.parametrize("state,config", [
    (DrivingState(speed_limit_mps=30. / 3.6), DrivingConfig()),
    (DrivingState(speed_limit_mps=30. / 3.6, speed_limit_confidence=.1), DrivingConfig()),
    (DrivingState(speed_limit_mps=30. / 3.6), DrivingConfig(max_speed_mps=3.)),
])
def test_soft_hazards_use_lowest_independent_cap(hazards, state, config):
    combined = {key: value for hazard in hazards for key, value in hazard.items()}
    expected = min(compute_target_speed(replace(state, **hazard), config) for hazard in hazards)
    assert compute_target_speed(replace(state, **combined), config) == pytest.approx(expected)


@pytest.mark.parametrize("gap", [1., 4., 6., 12., 40.])
def test_following_envelope_is_not_scaled_again_by_awareness(gap):
    state = DrivingState(speed_limit_mps=10., rain=1., wetness=.9, night=1.,
                         pedestrian_risk=1., child_pedestrian_risk=.5, vehicle_ahead=True)
    expected = min(compute_target_speed(state), max(0., (gap - 4.) / 2.))
    assert compute_target_speed(replace(state, front_hazard_m=gap)) == pytest.approx(expected)


def test_rain_night_lead_vehicle_target_agrees_between_controller_and_supervisor():
    state = DrivingState(speed_mps=5., speed_limit_mps=30. / 3.6, rain=1., wetness=.9,
                         night=1., vehicle_ahead=True, front_hazard_m=30.)
    obs = observation(speed=5., limit=30. / 3.6, rain=1., wetness=.9, night=1.)
    obs["tokens"][4, [0, 8]] = [.3, 1.]
    action = RouteController().act(obs, None)[0]
    result = SafetySupervisor().step(state, action, .05)
    assert result.target_speed_mps == pytest.approx(30. / 3.6 * .65)
    assert action[1] == pytest.approx(.5 * (result.target_speed_mps - 5.), abs=1e-6)
    assert result.action[1] == pytest.approx(action[1], abs=1e-6)
    assert result.action[1] > 0.


def test_yellow_approach_does_not_multiply_a_visibility_cap():
    state = DrivingState(speed_mps=4., speed_limit_mps=10., visibility_m=20.,
                         yellow_light=True, yellow_light_distance_m=100.)
    assert compute_target_speed(state) == pytest.approx(2.5)


def test_stage6_logged_rain_night_mixed_state_uses_visibility_cap():
    # Episode 2577, first safety_before state in stage6_run/metrics.jsonl.
    state = DrivingState(speed_mps=.4268595874325346, speed_limit_mps=30. / 3.6,
                         rain=.3, wetness=.8, night=1., visibility_m=54.54545454545454,
                         pedestrian_risk=1., vehicle_ahead=True, front_hazard_m=17.26832151412964)
    result = SafetySupervisor().step(state, (0., 1.), .05)
    assert result.target_speed_mps == pytest.approx(5.681818181818182)
    assert result.action[1] == 1.


@pytest.mark.parametrize("hazard,reason", [
    ({"front_hazard_m": 1.}, "front_hazard_brake"),
    ({"red_light": True, "red_light_distance_m": 1.}, "red_light_stop"),
    ({"yellow_light": True}, "yellow_light_stop"),
    ({"crossing_id": "crossing", "crossing_distance_m": 1.}, "crosswalk_stop"),
    ({"pedestrian_conflict": True}, "pedestrian_conflict_stop"),
    ({"crossing_occupied": True}, "pedestrian_conflict_stop"),
    ({"observations_valid": False}, "invalid_observation"),
])
def test_combined_hazards_preserve_full_brake_stops(hazard, reason):
    state = DrivingState(speed_mps=.2, speed_limit_mps=10., rain=1., wetness=.9,
                         night=1., vehicle_ahead=True, pedestrian_risk=1., child_pedestrian_risk=1.,
                         **hazard)
    result = SafetySupervisor().step(state, (0., 1.), .05)
    assert result.target_speed_mps == 0.
    assert result.action[1] == -1.
    assert reason in result.intervention_reasons

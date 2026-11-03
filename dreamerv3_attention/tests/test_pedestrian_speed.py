"""Runtime pedestrian awareness slowdown preserves independent hard stops."""

from dataclasses import replace
from pathlib import Path

import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor, compute_target_speed
from dreamerv3_attention.evaluation import collection_scenarios
from dreamerv3_attention.tests.test_controller import observation
from dreamerv3_attention.tests.test_environment import fake_environment


@pytest.mark.parametrize("factor", [0., -0.1, 1.1, True, None, "0.5", float("nan"), float("inf")])
@pytest.mark.parametrize("name", ["pedestrian_speed_factor", "child_pedestrian_speed_factor",
                                 "vehicle_speed_factor"])
def test_pedestrian_speed_factor_rejects_invalid_values(factor, name):
    with pytest.raises(ValueError, match=name):
        Config(driving={name: factor}).validate()


def test_pedestrian_speed_factors_default_by_age_and_round_trip(tmp_path):
    assert Config().resolved().driving["pedestrian_speed_factor"] == .8
    assert Config().resolved().driving["vehicle_speed_factor"] == .8
    assert Config().resolved().driving["child_pedestrian_speed_factor"] == .15
    state = DrivingState(speed_limit_mps=8., front_hazard_m=10., pedestrian_risk=1.)
    assert compute_target_speed(state) == pytest.approx(3.)
    assert compute_target_speed(replace(state, child_pedestrian_risk=1.)) == pytest.approx(1.2)
    cfg = Config(driving={"pedestrian_speed_factor": .6, "child_pedestrian_speed_factor": .2,
                          "vehicle_speed_factor": .7}).validate()
    path = tmp_path / "config.json"
    cfg.save(path)
    assert Config.load(path).driving["pedestrian_speed_factor"] == .6
    assert Config.load(path).driving["child_pedestrian_speed_factor"] == .2
    assert Config.load(path).driving["vehicle_speed_factor"] == .7


@pytest.mark.parametrize("risk,expected", [(0., 3.), (.5, 2.25), (1., 1.5)])
@pytest.mark.parametrize("child", [False, True])
def test_soft_slowdown_scales_base_speed_then_respects_following_limit(risk, expected, child):
    config = DrivingConfig(pedestrian_speed_factor=.5, child_pedestrian_speed_factor=.5, max_speed_mps=3.)
    state = DrivingState(speed_limit_mps=8., front_hazard_m=12.,
                         **{"child_pedestrian_risk" if child else "pedestrian_risk": risk})
    assert compute_target_speed(state, config) == pytest.approx(expected)
    assert compute_target_speed(replace(state, front_hazard_m=6.), config) == 1.


def test_runtime_factor_changes_controller_and_supervisor_without_changing_observations():
    obs = observation(speed=1., limit=3., risk=1.)
    original_tokens = obs["tokens"].copy()
    legacy_config = DrivingConfig(pedestrian_speed_factor=.15)
    legacy_action = RouteController(config=legacy_config).act(obs, None)[0]
    config = DrivingConfig(pedestrian_speed_factor=.5)
    action = RouteController(config=config).act(obs, None)[0]
    state = DrivingState(speed_mps=1., speed_limit_mps=3., pedestrian_risk=1.)
    legacy = SafetySupervisor(legacy_config).step(state, legacy_action, .05)
    relaxed = SafetySupervisor(config).step(state, action, .05)
    assert legacy_action[1] < 0. and legacy.action[1] < 0.
    assert action[1] > 0. and relaxed.action[1] > 0.
    assert "speed_target_brake" not in relaxed.intervention_reasons
    assert (obs["tokens"] == original_tokens).all()


@pytest.mark.parametrize("hazard,reason", [
    ({"pedestrian_conflict": True}, "pedestrian_conflict_stop"),
    ({"crossing_occupied": True}, "pedestrian_conflict_stop"),
    ({"front_hazard_m": 1.}, "front_hazard_brake"),
    ({"red_light": True, "red_light_distance_m": 1.}, "red_light_stop"),
    ({"observations_valid": False}, "invalid_observation"),
])
@pytest.mark.parametrize("child", [False, True])
def test_runtime_factor_never_releases_hard_stops(hazard, reason, child):
    state = DrivingState(speed_mps=1., speed_limit_mps=3., pedestrian_risk=1.,
                         child_pedestrian_risk=float(child), **hazard)
    result = SafetySupervisor(DrivingConfig(pedestrian_speed_factor=1.)).step(state, (0., 1.), .05)
    assert result.action[1] == -1.
    assert reason in result.intervention_reasons


@pytest.mark.parametrize("child", [False, True])
def test_runtime_factor_preserves_actor_requested_brake(child):
    state = DrivingState(speed_mps=1., speed_limit_mps=3., pedestrian_risk=1.,
                         child_pedestrian_risk=float(child))
    result = SafetySupervisor(DrivingConfig(pedestrian_speed_factor=.5)).step(state, (0., -.7), .05)
    assert result.action[1] == -.7


def test_runtime_evaluation_changes_only_the_soft_pedestrian_factor():
    root = Path("runs/experiment_26/curriculum")
    original = Config.load(root / "evaluate_pedestrians_vehicles.json").resolved()
    runtime = Config.load(root / "pedestrian_runtime_eval.json").resolved()
    assert runtime.driving == {**original.driving, "pedestrian_speed_factor": .5}
    assert replace(runtime, driving=original.driving) == original
    assert collection_scenarios(runtime) == collection_scenarios(original)


@pytest.mark.parametrize("child,expected", [(False, 2.4), (True, .45)])
def test_default_age_targets_agree_between_controller_and_supervisor(child, expected):
    state = DrivingState(speed_mps=1., speed_limit_mps=3., pedestrian_risk=1.,
                         child_pedestrian_risk=float(child))
    obs = observation(speed=1., limit=3., risk=1., child_risk=float(child))
    action = RouteController().act(obs, None)[0]
    result = SafetySupervisor().step(state, action, .05)
    assert compute_target_speed(state) == pytest.approx(expected)
    assert result.target_speed_mps == pytest.approx(expected)
    assert action[1] == pytest.approx(.5 * (expected - 1.))
    assert (result.action[1] < 0.) == child


@pytest.mark.parametrize("child,expected", [(False, 8.), (True, 1.5)])
def test_nearby_sidewalk_pedestrian_has_age_specific_target(child, expected):
    import carla

    env, _ = fake_environment()
    actor = env.world.get_actors()[0]
    actor.attributes = {"age": "child" if child else "adult"}
    actor.get_location = lambda: carla.Location(x=10., y=4.)
    state, _ = env._observe(42)
    result = env.supervisor.step(state, (0., 1.), .05)
    assert state.pedestrian_risk == 1.
    assert state.child_pedestrian_risk == float(child)
    assert not state.pedestrian_conflict
    assert state.front_hazard_m is None
    assert result.target_speed_mps == pytest.approx(expected)


def test_child_risk_alone_and_mixed_risks_use_strongest_factor_without_stacking():
    state = DrivingState(speed_limit_mps=10., child_pedestrian_risk=1.)
    assert compute_target_speed(state) == pytest.approx(1.5)
    assert compute_target_speed(replace(state, pedestrian_risk=1.)) == pytest.approx(1.5)
    assert compute_target_speed(replace(state, child_pedestrian_risk=.5,
                                       pedestrian_risk=.5)) == pytest.approx(5.75)


def test_stage5_runtime_config_changes_only_pedestrian_factors():
    root = Path("runs/experiment_26/curriculum")
    original = Config.load(root / "stage5_run_2/config.json").resolved()
    runtime = Config.load(root / "stage5_pedestrian_runtime.json").resolved()
    assert runtime.driving == {**original.driving, "pedestrian_speed_factor": .5,
                               "child_pedestrian_speed_factor": .15}
    assert replace(runtime, driving=original.driving) == original


def test_vehicle_factor_preserves_following_limits_and_uses_strongest_slowdown():
    state = DrivingState(speed_limit_mps=10., vehicle_ahead=True)
    assert compute_target_speed(state) == pytest.approx(8.)
    assert compute_target_speed(replace(state, pedestrian_risk=1.)) == pytest.approx(8.)
    assert compute_target_speed(replace(state, child_pedestrian_risk=1.)) == pytest.approx(1.5)
    assert compute_target_speed(replace(state, front_hazard_m=6.)) == pytest.approx(1.)
    assert compute_target_speed(replace(state, front_hazard_m=4.)) == 0.
    assert compute_target_speed(replace(state, front_hazard_m=1.)) == 0.
    result = SafetySupervisor().step(replace(state, front_hazard_m=1.), (0., 1.), .05)
    assert result.action[1] == -1. and "front_hazard_brake" in result.intervention_reasons
    assert SafetySupervisor().step(state, (0., -.7), .05).action[1] == -.7


@pytest.mark.parametrize("x,y,pedestrian,valid,present,expected", [
    (30., 0., False, True, True, 8.),
    (-30., 0., False, True, True, 10.),
    (30., 4., False, True, True, 10.),
    (81., 0., False, True, True, 10.),
    (30., 0., True, True, True, 10.),
    (30., 0., False, False, True, 10.),
    (30., 0., False, True, False, 10.),  # Radar points have no object presence flag.
])
def test_controller_vehicle_factor_uses_only_valid_vehicle_tokens_ahead(
        x, y, pedestrian, valid, present, expected):
    obs = observation(speed=7., limit=10.)
    obs["tokens"][4, [0, 1, 6, 8]] = [x / 100., y / 100., float(pedestrian), float(present)]
    obs["mask"][4] = valid
    action = RouteController().act(obs, None)[0]
    assert action[1] == pytest.approx(min(1., .5 * (expected - 7.)))


def test_environment_vehicle_detection_agrees_with_controller_and_supervisor():
    import carla

    env, _ = fake_environment()
    actor = env.world.get_actors()[0]
    actor.type_id = "vehicle.tesla.model3"
    actor.get_location = lambda: carla.Location(x=30.)
    env.ego.get_velocity = lambda: carla.Vector3D(x=7.)
    state, obs = env._observe(42)
    assert state.vehicle_ahead and not state.pedestrian_risk
    action = RouteController().act(obs, None)[0]
    result = env.supervisor.step(state, action, .05)
    assert compute_target_speed(state) == pytest.approx(8.)
    assert result.target_speed_mps == pytest.approx(8.)
    assert action[1] == pytest.approx(.5)


def test_external_vehicle_detection_uses_provider_objects_without_oracle_leak():
    import carla
    import numpy as np
    from dreamerv3_attention.sensors import PerceptionPrediction
    from dreamerv3_attention.tests.test_environment import safe_external_fields

    rows = np.zeros((1, 16), np.float32)
    rows[0, [0, 8]] = [.3, 1.]
    prediction = PerceptionPrediction(objects=rows, driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction=prediction)
    actor = env.world.get_actors()[0]
    actor.type_id = "vehicle.tesla.model3"
    actor.get_location = lambda: carla.Location(x=30.)
    state, _ = env._observe(42)
    assert state.vehicle_ahead
    prediction.objects = np.zeros((0, 16), np.float32)
    prediction.object_valid = np.zeros(0, dtype=bool)
    state, _ = env._observe(42)
    assert env.oracle_state.vehicle_ahead
    assert not state.vehicle_ahead

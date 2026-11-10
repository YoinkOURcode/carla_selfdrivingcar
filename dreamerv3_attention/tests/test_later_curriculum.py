"""Offline checks for signal, pedestrian and wet-weather curriculum stages."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import carla
import numpy as np
import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.driving import DrivingState, compute_target_speed
from dreamerv3_attention.environment import CarlaEnv, _UnsupportedRoute
from dreamerv3_attention.evaluation import collection_scenarios, episode_metrics
from dreamerv3_attention.navigation import Route
from dreamerv3_attention.tests.test_environment import (
    npc_control_environment, reset_test_environment, spawn_test_environment, traffic_waypoint,
)
from dreamerv3_attention.tests.test_motion_pilot import route_env
from dreamerv3_attention.weather import scenario_weather


ROOT = Path("runs/experiment_26/curriculum")


@pytest.mark.parametrize("position,valid", [(5., False), (29., False), (30., True),
                                          (60., True), (90., True), (95., False)])
def test_light_task_requires_an_approachable_stop_before_destination(position, valid):
    env = object.__new__(CarlaEnv)
    env.scenario = {"requires_traffic_light": True}
    env.map = NS(get_crosswalks=lambda: [])
    light = NS(type_id="traffic.traffic_light", get_stop_waypoints=Mock(
        return_value=[traffic_waypoint(position)]))
    env.world = NS(get_actors=Mock(return_value=[light]))
    env.route = Route([(900., 0.), (1000., 0.)])
    original = env.route
    waypoints = [(traffic_waypoint(0), None), (traffic_waypoint(100), None)]
    if valid:
        for _ in range(2):
            route, _ = env._route_geometry(waypoints)
            assert route.length == 100.
        light.get_stop_waypoints.assert_called_once()
        assert env.route is original  # Candidate checks do not replace the live route.
    else:
        with pytest.raises(_UnsupportedRoute, match="no usable traffic-light stop"):
            env._route_geometry(waypoints)


def test_wrong_lane_light_does_not_satisfy_signal_task():
    env = object.__new__(CarlaEnv)
    env.scenario = {"requires_traffic_light": True}
    env.map = NS(get_crosswalks=lambda: [])
    env.world = NS(get_actors=lambda: [NS(type_id="traffic.traffic_light",
                   get_stop_waypoints=lambda: [traffic_waypoint(50., lane=2)])])
    with pytest.raises(_UnsupportedRoute, match="no usable traffic-light stop"):
        env._route_geometry([(traffic_waypoint(0), None), (traffic_waypoint(100), None)])


def test_signal_evaluation_uses_bounded_route_search_even_without_randomization(monkeypatch):
    env, spawns = route_env(monkeypatch)
    env.scenario = {"name": "lights", "seed": 2604, "requires_traffic_light": True}
    env.cfg = replace(env.cfg, randomize_routes=False)
    light = NS(type_id="traffic.traffic_light", get_stop_waypoints=lambda: [traffic_waypoint(50.)])
    env.world = NS(get_actors=lambda: [light])
    env.route_planner.trace_route = lambda *_: [(traffic_waypoint(0), None), (traffic_waypoint(100), None)]
    selected = env._select_route(spawns, 0, 1)
    assert len(env.route_pools["lights"]) == 1
    assert selected[3].length == 100.
    env.route_light_waypoints = []
    env.route_pools.clear()
    with pytest.raises(RuntimeError, match="no valid varied route in 2048 candidates"):
        env._select_route(spawns, 0, 1)


def test_selected_light_route_is_rechecked_before_episode_starts(monkeypatch):
    env = reset_test_environment(monkeypatch)
    env.scenario = {"requires_traffic_light": True}
    env._select_route = lambda *_: (0, 1, [(traffic_waypoint(0), None), (traffic_waypoint(100), None)],
                                   Route([(0., 0.), (100., 0.)]), [])
    with pytest.raises(_UnsupportedRoute, match="selected route has no usable traffic-light stop"):
        env._reset()


def test_signal_route_search_finds_seeded_candidate_after_first_128(monkeypatch):
    from dreamerv3_attention import evaluation

    monkeypatch.setattr(evaluation, "scenario_records", lambda _: [])
    env, _ = route_env(monkeypatch)
    env.cfg = replace(env.cfg, seed=7, split="train", randomize_routes=True)
    env.scenario = {"name": "train_stage7_skill3_variant2", "seed": 7301,
                    "requires_traffic_light": True, "max_heading_change_rad": .5}
    env.scenarios = [env.scenario]
    spawns = [carla.Transform(carla.Location(x=float(i))) for i in range(20)]
    candidates = np.random.default_rng(np.random.SeedSequence([7, 7301])).permutation(20 * 19)
    a, b = divmod(int(candidates[341]), 19)
    b += b >= a
    calls = []

    def trace(start, end):
        calls.append((int(start.x), int(end.x)))
        return ([(traffic_waypoint(0), None), (traffic_waypoint(100), None)]
                if calls[-1] == (a, b) else [])

    env.route_planner.trace_route = trace
    env.world = NS(get_actors=lambda: [NS(type_id="traffic.traffic_light",
                   get_stop_waypoints=lambda: [traffic_waypoint(50.)])])
    selected = env._select_route(spawns, 0, 1)
    assert selected[:2] == (a, b)
    assert calls.index((a, b)) == 341
    assert len(calls) == 20 * 19
    assert selected[3].length == 100.
    assert len(env.route_pools[env.scenario["name"]]) == 1
    env._select_route(spawns, 0, 1)
    assert len(calls) == 20 * 19  # Reuse the validated pool.


@pytest.mark.parametrize("signals,limit", [(False, 128), (True, 2048)])
def test_route_search_is_bounded_when_no_candidate_satisfies_constraints(monkeypatch, signals, limit):
    from dreamerv3_attention import evaluation

    monkeypatch.setattr(evaluation, "scenario_records", lambda _: [])
    env, _ = route_env(monkeypatch)
    env.cfg = replace(env.cfg, split="train", randomize_routes=True)
    env.scenario = {"name": "unsupported", "requires_traffic_light": signals}
    env.scenarios = [env.scenario]
    spawns = [carla.Transform(carla.Location(x=float(i))) for i in range(50)]
    calls = []
    env.route_planner.trace_route = lambda start, end: calls.append((start, end)) or []
    with pytest.raises(RuntimeError, match=f"no valid varied route in {limit} candidates"):
        env._select_route(spawns, 0, 1)
    assert len(calls) == limit


@pytest.mark.parametrize("failure", ["no_nav", "distant_nav", "blocked"])
def test_required_pedestrian_cannot_be_silently_skipped(failure):
    env = spawn_test_environment(7, lambda bp, _pose: NS(type_id=bp.id))
    env.scenario = {"requires_pedestrians": True, "background_traffic": False,
                    "actors": [{"kind": "pedestrian", "x": 10., "y": 6., "speed_mps": 1.}]}
    if failure == "no_nav":
        env.world.get_random_location_from_navigation = Mock(return_value=None)
    elif failure == "distant_nav":
        env.world.get_random_location_from_navigation = Mock(return_value=carla.Location(x=50., y=6.))
    else:
        env.world.try_spawn_actor = lambda *_: None
    with pytest.raises(RuntimeError, match="Unsupported scenario: required pedestrian"):
        env._spawn_hazards()
    if failure != "blocked":
        assert 128 < env.world.get_random_location_from_navigation.call_count <= 16384


def test_required_pedestrian_is_spawned_in_initial_awareness_corridor():
    placements = []
    env = spawn_test_environment(7, lambda bp, pose:
                                 placements.append(pose.location) or NS(type_id=bp.id))
    env.scenario = {"requires_pedestrians": True, "background_traffic": False,
                    "actors": [{"kind": "pedestrian", "x": 10., "y": 6., "speed_mps": 1.}]}
    env._spawn_hazards()
    assert len(env.required_pedestrians) == len(env.actors) == 1
    assert 3. <= placements[0].x <= 14. and 2. <= abs(placements[0].y) < 7.


@pytest.mark.parametrize("yaw", [0., 90., 180.])
def test_required_pedestrian_searches_beyond_unrelated_nearby_navigation(yaw):
    placements = []
    env = spawn_test_environment(7, lambda bp, pose:
                                 placements.append(pose.location) or NS(type_id=bp.id))
    transform = carla.Transform(carla.Location(x=100., y=200.), carla.Rotation(yaw=yaw))
    env.ego.get_transform = lambda: transform
    env.scenario = {"requires_pedestrians": True, "background_traffic": False,
                    "actors": [{"kind": "pedestrian", "x": 10., "y": 6., "speed_mps": 1.}]}
    missed = transform.transform(carla.Location(x=40., y=6.))
    visible = transform.transform(carla.Location(x=10., y=6.))
    draws = iter([missed] * 256 + [visible] * 8)
    env.world.get_random_location_from_navigation = Mock(side_effect=lambda: next(draws))
    env._spawn_hazards()
    assert len(env.required_pedestrians) == len(placements) == 1
    assert placements[0].distance(visible + carla.Location(z=.5)) < .01
    assert env.world.get_random_location_from_navigation.call_count > 256


def test_required_pedestrian_tries_another_visible_navigation_point_if_blocked():
    placements = []

    def spawn(bp, pose):
        placements.append(pose.location)
        return None if len(placements) == 1 else NS(type_id=bp.id)

    env = spawn_test_environment(7, spawn)
    env.scenario = {"requires_pedestrians": True, "background_traffic": False,
                    "actors": [{"kind": "pedestrian", "x": 10., "y": 6., "speed_mps": 1.}]}
    env._spawn_hazards()
    assert len(env.required_pedestrians) == len(env.actors) == 1
    assert len(placements) == 2 and placements[0].distance(placements[1]) > .1


@pytest.mark.parametrize("failure", ["target", "controller", None])
def test_required_moving_pedestrian_needs_working_native_controller(failure):
    env, _ = npc_control_environment()
    actor = Mock(type_id="walker.pedestrian.0001", is_alive=True)
    env.moving = [(actor, {"speed_mps": 1.})]
    env.required_pedestrians = [actor]
    controller = Mock()
    env.world.get_blueprint_library = lambda: NS(find=lambda _: object())
    env.world.try_spawn_actor = lambda *_args, **_kwargs: None if failure == "controller" else controller
    if failure == "target":
        env.world.get_random_location_from_navigation.return_value = None
    if failure:
        with pytest.raises(RuntimeError, match="Unsupported scenario: required pedestrian"):
            env._start_npcs()
    else:
        env._start_npcs()
        controller.start.assert_called_once()
        controller.go_to_location.assert_called_once()
        controller.set_max_speed.assert_called_once_with(1.)
        actor.set_collisions.assert_called_once_with(True)


@pytest.mark.parametrize("field,value", [("requires_traffic_light", 1), ("requires_pedestrians", "yes"),
                                        ("requires_vehicles", 1),
                                        ("wetness", True), ("wetness", -1), ("wetness", 101),
                                        ("wetness", float("nan"))])
def test_new_catalog_settings_are_validated_before_carla(tmp_path, field, value):
    record = {"name": "case", "split": "train", "seed": 1, "route_start": 0,
              "route_end": 1, "category": "straight", field: value}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match=field):
        collection_scenarios(Config(scenario_file=str(path), scenario="case"))


def test_new_stage_sequence_retains_rewards_prior_tasks_and_expert_coverage():
    previous = Config.load(ROOT / "stage2_turns.json").resolved()
    labels = ("stage3_lights", "stage4_pedestrians", "stage5_wet_weather")
    counts = (4, 6, 9)
    for label, count in zip(labels, counts):
        cfg = Config.load(ROOT / (label + ".json")).resolved()
        assert cfg.scenarios[:len(previous.scenarios)] == previous.scenarios
        assert len(cfg.scenarios) == count
        assert cfg.reward == previous.reward and cfg.driving == previous.driving
        assert cfg.steps > previous.steps
        assert math.gcd(cfg.expert_episode_interval, len(cfg.scenarios)) == 1
        assert not any(r["background_traffic"] for r in collection_scenarios(cfg))
        previous = cfg
    records = collection_scenarios(previous)
    assert records[3]["requires_traffic_light"]
    assert records[4]["requires_pedestrians"] and len(records[4]["actors"]) == 1
    assert records[5]["requires_vehicles"] and records[5]["requires_pedestrians"]
    assert [actor["kind"] for actor in records[5]["actors"]] == ["vehicle", "pedestrian"]
    for record in records[6:]:
        weather = scenario_weather(carla, record, "ClearNoon")
        assert weather.wetness > 0
        state = DrivingState(speed_limit_mps=12., rain=weather.precipitation / 100., wetness=weather.wetness / 100.)
        assert compute_target_speed(state) < 12.
    for label, record in zip(("lights", "pedestrians", "pedestrians_vehicles", "wet_straight",
                             "rain_turns", "heavy_rain_lights"),
                             records[3:]):
        cfg = Config.load(ROOT / ("evaluate_" + label + ".json")).resolved()
        assert cfg.split == "train" and not cfg.scenarios and cfg.scenario == record["name"]
        assert cfg.reward == previous.reward and cfg.driving == previous.driving
        assert len(collection_scenarios(cfg)) == 1


def test_required_vehicle_cannot_be_silently_skipped():
    env = spawn_test_environment(7, lambda bp, _pose: None if bp.id.startswith("vehicle.") else NS(type_id=bp.id))
    env.scenario = {"background_traffic": False, "requires_vehicles": True,
                    "actors": [{"kind": "vehicle", "x": 22., "speed_mps": 3.}]}
    with pytest.raises(RuntimeError, match="required vehicle could not spawn"):
        env._spawn_hazards()


def test_required_vehicle_needs_an_authored_vehicle_in_catalog(tmp_path):
    record = {"name": "case", "split": "train", "seed": 1, "route_start": 0,
              "route_end": 1, "category": "straight", "requires_vehicles": True, "actors": []}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match="requires_vehicles needs authored vehicle"):
        collection_scenarios(Config(scenario_file=str(path), scenario="case"))


def test_reset_applies_wetness_override_without_mutating_preset():
    env = object.__new__(CarlaEnv)
    env.carla, env.cfg = carla, Config()
    env.scenarios = [{"name": "wet", "seed": 1, "weather": "WetNoon", "wetness": 60.}]
    env.scenario_episode = 0
    env.world = NS(set_weather=Mock())
    env._reset = lambda: {}
    env._destroy = Mock()
    before = carla.WeatherParameters.WetNoon.wetness
    env.reset()
    assert env.world.set_weather.call_args.args[0].wetness == 60.
    assert carla.WeatherParameters.WetNoon.wetness == before


def test_evaluation_counts_signal_and_pedestrian_exposure_separately_from_violations():
    metrics = episode_metrics([{"red_light": True, "pedestrian_risk": .5},
                               {"red_light": True}, {"yellow_light": True}, {}], .05)
    assert metrics["red_light_observation_steps"] == 2
    assert metrics["yellow_light_observation_steps"] == 1
    assert metrics["pedestrian_encounter_steps"] == 1
    assert metrics["red_light_violations"] == metrics["pedestrian_incidents"] == 0

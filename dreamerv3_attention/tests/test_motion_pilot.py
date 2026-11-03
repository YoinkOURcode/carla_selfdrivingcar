from types import SimpleNamespace as NS
from dataclasses import replace
import math
import sys

import numpy as np
import pytest

from dreamerv3_attention.environment import CarlaEnv, _UnsupportedRoute
from dreamerv3_attention.config import Config
from dreamerv3_attention.tests.test_environment import spawn_test_environment


def waypoints(length=60., step=2., junction_at=None):
    result = []
    for x in np.arange(0., length + step, step):
        wp = NS(transform=NS(location=NS(x=float(x), y=0.)),
                is_junction=junction_at is not None and x >= junction_at,
                lane_width=3.5)
        result.append((wp, None))
    return result


def test_motion_pilot_crops_route_before_geometry():
    env = object.__new__(CarlaEnv)
    env.scenario = {"route_length_m": 40.}
    cropped = env._crop_scenario_route(waypoints())
    assert cropped[-1][0].transform.location.x == 40.
    assert len(cropped) == 21


def test_motion_pilot_rejects_junction_in_cropped_segment():
    env = object.__new__(CarlaEnv)
    env.scenario = {"route_length_m": 40.}
    with pytest.raises(_UnsupportedRoute, match="contains a junction"):
        env._crop_scenario_route(waypoints(junction_at=30.))


def route_env(monkeypatch, *, route_length=50.):
    # _select_route imports the planner class even when a test planner is injected.
    agents = NS(__path__=[])
    navigation = NS(__path__=[])
    planner_module = NS(GlobalRoutePlanner=object)
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.navigation", navigation)
    monkeypatch.setitem(sys.modules, "agents.navigation.global_route_planner", planner_module)
    env = object.__new__(CarlaEnv)
    env.cfg = Config(split="validation", route_start=0, route_end=1)
    env.scenarios = None
    env.scenario = {"name": "motion_pilot", "seed": 2501, "route_length_m": 40.,
                    "category": "straight"}
    env.route_visit = 99
    env.route_pools = {}
    env.map = NS(get_crosswalks=lambda: [])
    spawns = [NS(location=NS(x=float(i * 20), y=float(i * 3))) for i in range(5)]
    env.route_planner = NS(trace_route=lambda start, end: waypoints(route_length))
    return env, spawns


def test_short_route_uses_one_seeded_candidate_in_evaluation(monkeypatch):
    env, spawns = route_env(monkeypatch)
    selected = env._select_route(spawns, 0, 1)
    again = env._select_route(spawns, 3, 4)
    assert selected[0:2] == again[0:2]
    assert (selected[0], selected[1]) != (0, 1)
    assert selected[3].length == pytest.approx(40.)
    assert not selected[3].junctions.any()
    assert len(env.route_pools["motion_pilot"]) == 1


def test_short_route_pool_fails_bounded_when_candidates_are_too_short(monkeypatch):
    env, spawns = route_env(monkeypatch, route_length=30.)
    calls = []
    env.route_planner.trace_route = lambda start, end: calls.append((start, end)) or waypoints(30.)
    with pytest.raises(RuntimeError, match="no valid route in 128 candidates"):
        env._select_route(spawns, 0, 1)
    assert 0 < len(calls) <= 128


def test_varied_short_routes_cycle_distinct_starts_and_reproduce(monkeypatch):
    def make_env():
        env, _ = route_env(monkeypatch)
        env.cfg = replace(env.cfg, split="train", randomize_routes=True)
        env.scenarios = [env.scenario]
        spawns = [NS(location=NS(x=float(i * 100), y=0.)) for i in range(24)]

        def trace(start, end):
            points = waypoints()
            for wp, _ in points:
                wp.transform.location.x += start.x
            return points

        env.route_planner.trace_route = trace
        return env, spawns

    env, spawns = make_env()
    routes = []
    for visit in range(16):
        env.route_visit = visit
        selected = env._select_route(spawns, 0, 1)
        assert selected[3].length == pytest.approx(40.)
        assert not selected[3].junctions.any()
        assert selected[3].index == 0
        routes.append(selected[:2])
        selected[3].index = 10
    assert len({start for start, end in routes}) == 16
    env.route_visit = 16
    assert env._select_route(spawns, 0, 1)[:2] == routes[0]
    assert env.route_pools["motion_pilot"][0][3].index == 0
    reproduced, spawns = make_env()
    for visit, expected in enumerate(routes):
        reproduced.route_visit = visit
        assert reproduced._select_route(spawns, 0, 1)[:2] == expected


def test_fixed_pilot_overrides_global_route_randomization(monkeypatch):
    original, spawns = route_env(monkeypatch)
    original.cfg = replace(original.cfg, split="train", randomize_routes=False)
    original.scenarios = [original.scenario]
    expected = original._select_route(spawns, 0, 1)
    env, spawns = route_env(monkeypatch)
    env.cfg = replace(env.cfg, split="train", randomize_routes=True)
    env.scenarios = [env.scenario]
    env.scenario["randomize_routes"] = False
    for visit in (0, 1, 20):
        env.route_visit = visit
        assert env._select_route(spawns, 0, 1)[:2] == expected[:2]
    assert len(env.route_pools["motion_pilot"]) == 1


def test_turn_stage_heading_bound_accepts_one_corner_and_rejects_u_turn(monkeypatch):
    env, _ = route_env(monkeypatch)
    env.scenario = {"category": "curved", "max_heading_change_rad": 1.8}

    def points(coordinates):
        return [(NS(transform=NS(location=NS(x=x, y=y)), lane_width=3.5, is_junction=False), None)
                for x, y in coordinates]

    route, _ = env._route_geometry(points([(0., 0.), (40., 0.), (40., 40.)]))
    assert route.length == 80.
    with pytest.raises(_UnsupportedRoute, match="maximum heading change"):
        env._route_geometry(points([(0., 0.), (40., 0.), (40., 40.), (0., 40.)]))
    with pytest.raises(_UnsupportedRoute, match="no curve"):
        env._route_geometry(waypoints())


@pytest.mark.parametrize("length", [0, -1, float("inf"), float("nan"), "bad"])
def test_motion_pilot_rejects_invalid_route_length(length):
    env = object.__new__(CarlaEnv)
    env.scenario = {"route_length_m": length}
    with pytest.raises(ValueError, match="finite and positive"):
        env._crop_scenario_route(waypoints())


@pytest.mark.parametrize("background", [0, 1, "false", None])
def test_motion_pilot_rejects_non_boolean_background_traffic(background):
    env = object.__new__(CarlaEnv)
    env.scenario = {"background_traffic": background}
    env.world = NS(get_blueprint_library=lambda: pytest.fail("validation must happen first"))
    with pytest.raises(ValueError, match="background_traffic must be a boolean"):
        env._spawn_hazards()


def test_authored_pedestrian_keeps_sidewalk_sampling_without_background_traffic():
    placements = []
    env = spawn_test_environment(7, lambda bp, transform:
                                 placements.append((bp.id, transform.location)) or NS(type_id=bp.id))
    env.scenario = {"background_traffic": False,
                    "actors": [{"kind": "pedestrian", "x": 20., "speed_mps": 1.}]}
    env._spawn_hazards()
    assert len(placements) == 1 and placements[0][0].startswith("walker.")
    assert abs(placements[0][1].y) == 6.


def test_no_background_and_no_authored_pedestrian_skips_sidewalk_sampling(caplog):
    env = spawn_test_environment(7, lambda bp, _transform: NS(type_id=bp.id))
    env.scenario = {"background_traffic": False, "actors": []}
    env.world.get_random_location_from_navigation = lambda: pytest.fail("no pedestrians requested")
    env._spawn_hazards()
    assert not env.actors and not caplog.records



@pytest.mark.parametrize("field,value", [("route_length_m", True), ("route_length_m", "40"),
                                        ("route_length_m", -1), ("route_length_m", float("inf")),
                                        ("background_traffic", 0), ("randomize_routes", 1),
                                        ("max_heading_change_rad", True), ("max_heading_change_rad", 0),
                                        ("max_heading_change_rad", "bad"),
                                        ("max_heading_change_rad", float("nan"))])
def test_invalid_pilot_settings_rejected_during_catalog_loading(tmp_path, field, value):
    import json
    from dreamerv3_attention.evaluation import collection_scenarios

    record = {"name": "pilot", "split": "train", "seed": 25, "route_start": 0,
              "route_end": 1, "category": "straight", field: value}
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match=field):
        collection_scenarios(Config(scenario_file=str(path), scenario="pilot"))


def test_curriculum_configs_preserve_rewards_and_rehearse_earlier_tasks():
    from dreamerv3_attention.evaluation import collection_scenarios

    prefix = "runs/experiment_26/curriculum/"
    pilot = Config.load("runs/experiment_26/motion_pilot_config.json").resolved()
    first = Config.load(prefix + "stage1_straights.json").resolved()
    second = Config.load(prefix + "stage2_turns.json").resolved()
    assert first.reward == second.reward == pilot.reward
    assert first.driving == second.driving == pilot.driving
    assert first.steps > pilot.steps and second.steps > first.steps
    assert first.scenarios == ["train_motion_pilot", "train_varied_straight"]
    assert second.scenarios == [*first.scenarios, "train_turns"]
    for cfg in (first, second):
        assert math.gcd(cfg.expert_episode_interval, len(cfg.scenarios)) == 1
        records = collection_scenarios(cfg)
        assert records[0]["randomize_routes"] is False
        assert records[0]["route_length_m"] == 40.
        assert all(not r["background_traffic"] and not r["actors"] for r in records)
        assert all(r["weather"] == "ClearNoon" for r in records)
        assert all(r["route_length_m"] == 80. and r["randomize_routes"] for r in records[1:])
    for label, name in zip(("pilot", "straight", "turns"), second.scenarios):
        cfg = Config.load(prefix + "evaluate_" + label + ".json").resolved()
        assert cfg.split == "train" and cfg.scenarios == [] and cfg.scenario == name
        assert len(collection_scenarios(cfg)) == 1
        assert cfg.reward == pilot.reward and cfg.driving == pilot.driving

from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock
import sys

import carla
import numpy as np
import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingConfig, DrivingState, RewardConfig, SafetyResult, SafetySupervisor
from dreamerv3_attention.environment import CarlaEnv, ego_coordinates
from dreamerv3_attention.inactivity import InactivityWatchdog
from dreamerv3_attention.navigation import Route
from dreamerv3_attention.observations import FrameBundle
from dreamerv3_attention.sensors import PerceptionPrediction


def fake_environment(mode="ground_truth", prediction=None, offroad=False):
    env = object.__new__(CarlaEnv)
    env.cfg = Config(image_size=16, max_objects=2, perception=mode,
                     provider="mock:provider" if mode != "ground_truth" else "")
    env.carla = carla
    env.scenario = {}
    transform = carla.Transform()
    env.ego = NS(id=1, get_transform=lambda: transform, get_velocity=lambda: carla.Vector3D(x=2),
                 get_speed_limit=lambda: 36., bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1)),
                 is_at_traffic_light=lambda: False)
    actor = NS(id=2, type_id="walker.pedestrian.0001", is_alive=True, attributes={"age": "child"},
               get_location=lambda: carla.Location(x=12, y=2), get_transform=lambda: carla.Transform(carla.Location(x=12, y=2)),
               get_velocity=lambda: carla.Vector3D(), bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(.3, .3, 1)))
    env.world = NS(get_snapshot=lambda: NS(frame=42, timestamp=NS(elapsed_seconds=2.1)),
                   get_weather=lambda: NS(precipitation=0, wetness=0, sun_altitude_angle=60, fog_density=0),
                   get_actors=lambda: [actor])
    projected = NS(lane_width=3.5)
    env.map = NS(get_waypoint=lambda loc, **kwargs: (
        None if offroad and kwargs.get("project_to_road") is False else projected))
    bundle = FrameBundle(42, 2.1, rgb=np.zeros((32, 32, 3), np.uint8), depth=None,
                         radars={s: np.zeros((0, 4), np.float32) for s in ("front", "left", "right", "rear")},
                         valid={s: True for s in ("rgb", "front", "left", "right", "rear")})
    env.rig = NS(get=lambda *args, **kwargs: bundle, sensors=[None, NS(get_transform=lambda: transform)],
                 consume_collision=lambda frame: [], camera=NS(width=1280, height=720, fov=90))
    env.route, env.crossings = Route([(0, 0), (100, 0)]), []
    env.traffic_stops, env.supervisor = [], SafetySupervisor()
    env.provider = NS(predict=lambda b: prediction) if prediction else None
    env.rng = np.random.default_rng(1)
    return env, bundle


def safe_external_fields():
    return dict(speed_limit_mps=5., speed_limit_confidence=1., lane_error_m=0., heading_error_rad=0.,
                crossing_id=None, crossing_distance_m=None, crossing_occupied=False, red_light=False,
                red_light_distance_m=None, yellow_light=False, yellow_light_distance_m=None,
                yellow_light_confidence=0., visibility_m=100., observations_valid=True)


def fake_step_environment(*, before=None, after=None, result=None, timeout_s=.1,
                          blocked_timeout_s=.3, max_steps=100, route_length=100.):
    env = object.__new__(CarlaEnv)
    before = before or DrivingState(speed_limit_mps=10.)
    after = after or replace(before)
    result = result or SafetyResult((0., 0.), target_speed_mps=3.)
    env.cfg = Config(dt=.1, max_episode_steps=max_steps, inactivity_timeout_s=timeout_s,
                     inactivity_min_progress_m=.5, blocked_timeout_s=blocked_timeout_s,
                     inactivity_grace_s=0., weather="ClearNoon", perception="ground_truth")
    env.carla = carla
    env.scenario = {}
    env.state = before
    env.oracle_before = replace(before)
    env.oracle_state = replace(after)
    env.radar_hazards = []
    env.supervisor = NS(step=lambda *_: result, config=DrivingConfig())
    env.ego = NS(apply_control=lambda control: None)
    env.moving = []
    env.steps = 0
    env.previous_action, env.previous_acceleration = (0., 0.), 0.
    env.reward_config = RewardConfig()
    env.route = NS(length=route_length)
    env.reward_progress_highwater = min(route_length, max(0., before.route_progress_m))
    env.next_state = replace(after)
    env.world = NS(tick=lambda: 42)
    env.inactivity = InactivityWatchdog(timeout_s, .5, blocked_timeout_s, 0.)

    def observe(_frame):
        env.oracle_state = replace(env.next_state)
        env.frame = 42
        env.last_bundle = NS(timestamp=.1, valid={"rgb": True})
        return replace(env.next_state), {"observation": 1}

    env._observe = observe
    return env


def step_until(env, count):
    transition = None
    for _ in range(count):
        transition = env.step((0., 0.))
    return transition


@pytest.mark.parametrize("reason,fields", [
    ("collision", {"collision": True}),
    ("pedestrian_collision", {"collision": True, "pedestrian_collision": True}),
    ("offroad", {"offroad": True}),
    ("lane_deviation", {"lane_error_m": -6.1}),
    ("red_light_violation", {"route_progress_m": 10.}),
    ("crossing_violation", {"crossing_id": "crossing", "crossing_distance_m": -1.}),
    ("success", {"route_progress_m": 98.}),
])
def test_terminal_safety_predicates_report_the_actual_reason(reason, fields):
    before = DrivingState(red_light=reason == "red_light_violation", red_light_distance_m=5.,
                          crossing_id="crossing", crossing_distance_m=1.)
    env = fake_step_environment(before=before, after=DrivingState(**fields))
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert terminated and not truncated
    assert info["termination_reason"] == reason
    assert info["truncation_reason"] is None


def test_ground_truth_units_objects_and_authored_children():
    env, _ = fake_environment()
    state, obs = env._observe(42)
    assert state.speed_limit_mps == 10.
    assert state.child_pedestrian_risk == state.pedestrian_risk == 1.
    assert obs["pixels"].shape == (3, 16, 16)
    assert obs["tokens"][4, 7] == 1.
    assert obs["tokens"][4, 13] == 0.  # no sensor depth, known GT position still valid
    assert (obs["tokens"][1:4, 15] == -1).all()


def test_physical_offroad_uses_unprojected_driving_lane_query():
    env, _ = fake_environment(offroad=True)
    state, _ = env._observe(42)
    assert state.offroad
    assert state.lane_width_m == 3.5
    env, _ = fake_environment(offroad=False)
    env.route.project = lambda *_args, **_kwargs: (10., 4., 0.)
    adjacent_lane, _ = env._observe(42)
    assert adjacent_lane.lane_departure
    assert not adjacent_lane.offroad


def test_external_prediction_cannot_hide_oracle_offroad_terminal():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32),
        driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction, offroad=True)
    policy_state, _ = env._observe(42)
    assert not policy_state.offroad
    assert env.oracle_state.offroad


def test_external_mode_does_not_leak_oracle_hazards():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    state, obs = env._observe(42)
    assert state.pedestrian_risk == state.child_pedestrian_risk == 0
    assert state.speed_limit_mps == 5
    assert env.oracle_state.child_pedestrian_risk == 1
    assert not obs["mask"][4:6].any()
    assert state.observations_valid


def test_external_objects_trigger_supervisor_hazard_state():
    objects = np.zeros((1, 16), np.float32)
    objects[0, :2] = [.04, .01]
    objects[0, 6] = 1.
    prediction = PerceptionPrediction(objects, driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    state, _ = env._observe(42)
    assert state.pedestrian_conflict
    assert state.front_hazard_m <= 2.01


def test_external_child_flags_require_authored_provenance():
    objects = np.zeros((1, 16), np.float32)
    objects[0, :2] = [.10, .02]
    objects[0, 6:8] = 1.
    prediction = PerceptionPrediction(objects, driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    assert env._observe(42)[0].child_pedestrian_risk == 0.
    env.provider.predict = lambda b: replace(prediction, child_labels_authored=True)
    assert env._observe(42)[0].child_pedestrian_risk == 1.


def test_missing_external_contract_and_missing_sensors_force_invalid():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), driving={"observations_valid": True}, frame=42, timestamp=2.1)
    env, bundle = fake_environment("external", prediction)
    assert not env._observe(42)[0].observations_valid
    env.provider.predict = lambda b: replace(prediction, driving=safe_external_fields())
    bundle.valid["rear"] = False
    assert not env._observe(42)[0].observations_valid


def test_external_future_predictions_rejected_and_stale_inputs_masked():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), frame=43, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    with pytest.raises(ValueError, match="frame"):
        env._observe(42)
    env, _ = fake_environment()
    env.scenario["stale_sensors"] = ["rgb"]
    state, obs = env._observe(42)
    assert not state.observations_valid and not obs["pixels_valid"]


def test_reset_failure_cleans_spawned_actors_and_close_preserves_unowned_settings():
    env = object.__new__(CarlaEnv)
    env.scenarios = None
    destroyed = []
    env.rig = None
    env.actors = [NS(is_alive=True, destroy=lambda: destroyed.append(1))]
    env._reset = lambda: (_ for _ in ()).throw(RuntimeError("failed sensor spawn"))
    with pytest.raises(RuntimeError, match="failed sensor"):
        env.reset()
    assert destroyed == [1] and env.actors == []
    env.closed, env.provider, env.owns_settings = False, None, False
    env.close()  # no world/settings access when ownership was never acquired


def test_clear_road_stationary_step_truncates_without_failure_reward():
    env = fake_step_environment()
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert not terminated and truncated
    assert info["truncation_reason"] == "inactivity_timeout"
    assert "termination" not in info["reward_components"]
    assert info["reward_components"]["collision"] == 0.


@pytest.mark.parametrize("gate", ("red_light", "crosswalk", "pedestrian", "invalid", "zero_target"))
def test_blocked_conditions_use_long_timeout_instead_of_short_timeout(gate):
    before = DrivingState(speed_limit_mps=10.)
    after = replace(before)
    result = SafetyResult((0., 0.), target_speed_mps=3.)
    if gate == "red_light":
        before.red_light = True
        after.red_light = True
    elif gate == "crosswalk":
        result = replace(result, crossing_phase="stopped")
    elif gate == "pedestrian":
        before.pedestrian_conflict = True
        after.pedestrian_conflict = True
    elif gate == "invalid":
        before.observations_valid = False
        after.observations_valid = False
        result = replace(result, intervention_reasons=("invalid_observation",))
    else:
        result = replace(result, target_speed_mps=0.)
    env = fake_step_environment(before=before, after=after, result=result)

    _, _, terminated, truncated, info = step_until(env, 2)
    assert not terminated and not truncated
    assert info["truncation_reason"] is None
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert not terminated and truncated
    assert info["truncation_reason"] == "blocked_timeout"


@pytest.mark.parametrize("terminal", ("collision", "offroad", "success"))
def test_genuine_termination_wins_over_simultaneous_watchdog_timeout(terminal):
    before = DrivingState(speed_limit_mps=10., route_progress_m=0.)
    after = replace(before)
    route_length = 100.
    if terminal == "collision":
        after.collision = True
    elif terminal == "offroad":
        after.offroad = True
        after.lane_error_m = 0.1  # Physical lane check, not route-error threshold.
    else:
        before.route_progress_m = after.route_progress_m = 97.
    env = fake_step_environment(before=before, after=after, timeout_s=.1,
                                blocked_timeout_s=1., route_length=route_length)
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert terminated and not truncated
    assert info["truncation_reason"] is None
    assert (info["success"] is (terminal == "success"))
    if terminal == "offroad":
        assert info["reward_components"]["offroad"] == -100.
        assert info["reward_components"]["success"] == 0.


def test_route_checkpoint_highwater_prevents_reverse_and_recross_farming():
    initial = fake_step_environment(before=DrivingState(route_progress_m=31., speed_limit_mps=10.),
                                    after=DrivingState(route_progress_m=32., speed_limit_mps=10.))
    *_, info = initial.step((0., 0.))
    assert info["waypoints_reached"] == 0
    assert info["new_progress_m"] == 1.

    env = fake_step_environment()
    env.next_state.route_progress_m = 31.
    *_, info = env.step((0., 0.))
    assert info["waypoints_reached"] == 2
    assert info["new_progress_m"] == 31.
    env.next_state.route_progress_m = 20.
    *_, info = env.step((0., 0.))
    assert info["waypoints_reached"] == 0
    assert info["new_progress_m"] == 0.
    env.next_state.route_progress_m = 46.
    *_, info = env.step((0., 0.))
    assert info["waypoints_reached"] == 1
    assert info["new_progress_m"] == 15.


def test_offroad_or_collision_at_route_goal_is_not_success():
    for failure in ("collision", "pedestrian_collision", "offroad"):
        before = DrivingState(speed_limit_mps=10., route_progress_m=97.)
        after = replace(before)
        setattr(after, failure, True)
        env = fake_step_environment(before=before, after=after, route_length=100.)
        _, _, terminated, truncated, info = env.step((0., 0.))
        assert terminated and not truncated
        assert not info["success"]


def test_max_episode_steps_sets_time_limit_reason_over_watchdog():
    env = fake_step_environment(max_steps=1)
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert not terminated and truncated
    assert info["truncation_reason"] == "time_limit"


def reset_test_environment(monkeypatch):
    import dreamerv3_attention.environment as environment_module

    env = object.__new__(CarlaEnv)
    env.carla = carla
    env.cfg = Config(dt=.1, route_start=0, route_end=1, weather="ClearNoon", camera={}, radar={})
    env.scenario, env.provider, env.actors, env.rig = {}, None, [], None
    env.supervisor = NS(reset=lambda: None)
    env.inactivity = InactivityWatchdog(1., .5, 2., 0.)
    env.inactivity.update(10., 1., can_move=True)
    env.rng = np.random.default_rng(1)
    points = [NS(location=NS(x=0., y=0.)), NS(location=NS(x=100., y=0.))]
    env.map = NS(name="Town01", get_spawn_points=lambda: points, get_crosswalks=lambda: [])
    blueprint = NS(set_attribute=lambda *args: None)
    library = NS(find=lambda _name: blueprint)
    env.ego = NS()
    env.world = NS(get_blueprint_library=lambda: library,
                   spawn_actor=lambda *_: env.ego, tick=lambda: 42, get_actors=lambda: [])
    env._destroy = lambda: None
    env._spawn_hazards = lambda: setattr(env, "moving", [])
    env._start_npcs = lambda: None

    def observe(_frame):
        env.oracle_state = DrivingState(route_progress_m=7.)
        return DrivingState(route_progress_m=7.), {"pixels": np.zeros((3, 16, 16), np.uint8)}

    env._observe = observe

    class Planner:
        def __init__(self, _map, _resolution):
            pass

        def trace_route(self, *_args):
            return [(NS(transform=NS(location=NS(x=0., y=0.)), is_junction=False), None),
                    (NS(transform=NS(location=NS(x=100., y=0.)), is_junction=False), None)]

    agents = ModuleType("agents")
    navigation = ModuleType("agents.navigation")
    planner_module = ModuleType("agents.navigation.global_route_planner")
    planner_module.GlobalRoutePlanner = Planner
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.navigation", navigation)
    monkeypatch.setitem(sys.modules, "agents.navigation.global_route_planner", planner_module)
    monkeypatch.setattr(environment_module, "crosswalk_polygons", lambda _map: [])
    monkeypatch.setattr(environment_module, "SensorRig", lambda *_args, **_kwargs: NS())
    return env


def test_reset_hook_restarts_watchdog_after_initial_oracle_observation(monkeypatch):
    env = reset_test_environment(monkeypatch)
    env.episode_return = -700.
    observation = env._reset()
    assert env.episode_return == 0.
    assert "pixels" in observation
    assert env.reward_progress_highwater == 7.
    assert env.inactivity._elapsed_s == 0.
    assert list(env.inactivity._long_history) == [(0., 7.)]
    assert list(env.inactivity._short_history) == [(0., 7.)]


@pytest.mark.parametrize("x", [-7., 22.])
def test_reset_updates_ego_snapshot_before_spawning_relative_hazards(monkeypatch, x):
    env = reset_test_environment(monkeypatch)
    start = carla.Transform(carla.Location(x=200., y=100., z=.3), carla.Rotation(yaw=90.))
    env.map.get_spawn_points = lambda: [start, carla.Transform(carla.Location(x=200., y=200.))]
    cached_transform = carla.Transform()
    env.ego = NS(get_transform=lambda: cached_transform)
    ticks = 0

    def tick():
        nonlocal cached_transform, ticks
        ticks += 1
        cached_transform = start
        return 42 + ticks

    placements = []

    def spawn(bp, transform):
        placements.append(ego_coordinates(start, transform.location))
        return NS(type_id=bp.id)

    library = spawn_test_environment(7, spawn).world.get_blueprint_library()
    library.find = lambda _name: NS(set_attribute=lambda *_args: None)
    env.world.get_blueprint_library = lambda: library
    env.world.try_spawn_actor = spawn
    fixture = spawn_test_environment(7, spawn)
    env.map.get_waypoint = lambda loc, **_kwargs: NS(
        transform=carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z-.5), start.rotation),
        is_junction=False)
    env.world.set_pedestrians_seed = fixture.world.set_pedestrians_seed
    env.world.set_pedestrians_cross_factor = fixture.world.set_pedestrians_cross_factor
    env.world.get_random_location_from_navigation = lambda: start.transform(
        fixture.world.get_random_location_from_navigation())
    env.world.tick = tick
    env.scenario = {"actors": [{"kind": "vehicle", "x": x, "y": 0.}]}
    env._spawn_hazards = lambda: CarlaEnv._spawn_hazards(env)
    env._reset()
    np.testing.assert_allclose(placements[0], [x, 0., 1.], atol=1e-5)
    assert len(env.actors) > 3


def traffic_waypoint(x, y=0., *, road=1, lane=1, section=0, yaw=0.):
    return NS(transform=carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw)),
              road_id=road, section_id=section, lane_id=lane, lane_width=3.5)


def install_traffic_stops(env, route_waypoints, stops):
    phase = [carla.TrafficLightState.Red]
    calls = {"geometry": 0, "phase": 0}

    def stop_waypoints():
        calls["geometry"] += 1
        return stops

    def get_state():
        calls["phase"] += 1
        return phase[0]

    light = NS(type_id="traffic.traffic_light", id=20,
               get_stop_waypoints=stop_waypoints, get_state=get_state)
    env.world.get_actors = lambda: [light]
    env.route = Route([(wp.transform.location.x, wp.transform.location.y) for wp in route_waypoints])
    env._cache_traffic_stops(route_waypoints)
    return phase, calls


def test_light_detected_before_trigger_with_bumper_offset_and_live_phase_release():
    env, _ = fake_environment()
    env.ego.bounding_box.location.x = .5
    env.ego.get_traffic_light = lambda: pytest.fail("Trigger lookup must not gate early detection")
    phase, calls = install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(100)],
                                      [traffic_waypoint(20)])
    assert not env.ego.is_at_traffic_light()
    state, _ = env._observe(42)
    assert state.red_light and state.red_light_distance_m == 17.5
    phase[0] = carla.TrafficLightState.Yellow
    yellow = env._observe(42)[0]
    assert yellow.yellow_light and not yellow.red_light
    assert yellow.yellow_light_distance_m == 17.5 and yellow.yellow_light_confidence == 1.
    phase[0] = carla.TrafficLightState.Green
    released, _ = env._observe(42)
    assert not released.red_light and released.red_light_distance_m is None
    assert calls == {"geometry": 1, "phase": 3}


@pytest.mark.parametrize("stop", [
    traffic_waypoint(20, road=2), traffic_waypoint(20, lane=2),
    traffic_waypoint(20, section=1), traffic_waypoint(20, yaw=180),
    traffic_waypoint(20, y=3.5), traffic_waypoint(-2), traffic_waypoint(102),
])
def test_unrelated_or_outside_route_stops_do_not_fabricate_hazards(stop):
    env, _ = fake_environment()
    install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(100)], [stop])
    state, _ = env._observe(42)
    assert env.traffic_stops == []
    assert not state.red_light and state.red_light_distance_m is None


def test_curved_route_uses_arc_distance_and_preserves_route_index_when_caching():
    env, _ = fake_environment()
    route = [traffic_waypoint(0), traffic_waypoint(10), traffic_waypoint(10, 20, yaw=90)]
    install_traffic_stops(env, route, [traffic_waypoint(10, 15, yaw=90)])
    env.route.index = 1
    env._cache_traffic_stops(route)
    assert env.route.index == 1
    state, _ = env._observe(42)
    assert state.red_light_distance_m == 23.  # 10 + 15 along route, minus 2 m bumper.


def test_stop_cache_preserves_separate_route_visits_and_deduplicates_segment_endpoints():
    env, _ = fake_environment()
    route = [traffic_waypoint(0), traffic_waypoint(10), traffic_waypoint(20),
             traffic_waypoint(20, 10, yaw=90), traffic_waypoint(0, 10, yaw=180),
             traffic_waypoint(0), traffic_waypoint(20)]
    install_traffic_stops(env, route, [traffic_waypoint(10)])
    np.testing.assert_allclose([position for _, position in env.traffic_stops], [10., 70.])
    state = DrivingState(route_progress_m=30.)
    env._observe_traffic_light(state, 2.)
    assert not state.red_light  # Second visit is still outside early lookahead.
    state.route_progress_m = 40.
    env._observe_traffic_light(state, 2.)
    assert state.red_light_distance_m == 28.


def test_light_lookahead_extends_with_speed_and_does_not_mark_entire_route_red():
    env, _ = fake_environment()
    install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(200)],
                         [traffic_waypoint(80)])
    slow = DrivingState(speed_mps=3.)
    env._observe_traffic_light(slow, 2.)
    assert not slow.red_light
    fast = DrivingState(speed_mps=14.)
    env._observe_traffic_light(fast, 2.)
    assert fast.red_light and fast.red_light_distance_m == 78.


def test_nearest_upcoming_red_is_selected_and_green_nearer_light_does_not_hide_it():
    env, _ = fake_environment()
    red = NS(type_id="traffic.traffic_light", get_stop_waypoints=lambda: [traffic_waypoint(25)],
             get_state=lambda: carla.TrafficLightState.Red)
    green = NS(type_id="traffic.traffic_light", get_stop_waypoints=lambda: [traffic_waypoint(15)],
               get_state=lambda: carla.TrafficLightState.Green)
    env.world.get_actors = lambda: [red, green]
    env._cache_traffic_stops([traffic_waypoint(0), traffic_waypoint(100)])
    state = DrivingState(speed_mps=3.)
    env._observe_traffic_light(state, 2.)
    assert state.red_light_distance_m == 23.


def test_positive_to_past_red_crossing_terminates_but_spawn_already_past_does_not():
    env, _ = fake_environment()
    install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(100)], [traffic_waypoint(20)])
    before = DrivingState(route_progress_m=17.5, speed_limit_mps=10.)
    env._observe_traffic_light(before, 2.)
    after = DrivingState(route_progress_m=18.5, speed_limit_mps=10.)
    env._observe_traffic_light(after, 2.)
    assert before.red_light_distance_m == .5
    assert not after.red_light and after.red_light_distance_m is None
    step_env = fake_step_environment(before=before, after=after)
    _, _, terminated, _, info = step_env.step((0., 0.))
    assert terminated and info["red_light_violation"]
    assert info["reward_components"]["red_light_violation"] == -8.
    spawned_past = fake_step_environment(before=after, after=replace(after, route_progress_m=19.))
    _, _, terminated, _, info = spawned_past.step((0., 0.))
    assert not terminated and not info["red_light_violation"]


def test_external_prediction_cannot_hide_or_leak_early_red_oracle():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32),
        driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(100)], [traffic_waypoint(20)])
    policy, observation = env._observe(42)
    assert not policy.red_light and policy.red_light_distance_m is None
    assert observation["tokens"][0, 12] == 0.
    assert env.oracle_state.red_light and env.oracle_state.red_light_distance_m == 18.
    step_env = fake_step_environment(before=policy, after=replace(policy, route_progress_m=19.))
    step_env.cfg.perception = "external"
    step_env.oracle_before = replace(env.oracle_state)
    _, _, terminated, _, info = step_env.step((0., 0.))
    assert terminated and info["red_light_violation"]


def test_observed_light_controller_and_supervisor_stop_hold_and_release_in_fake_plant():
    env, _ = fake_environment()
    env.cfg.dt = .1
    phase, _ = install_traffic_stops(env, [traffic_waypoint(0), traffic_waypoint(100)],
                                   [traffic_waypoint(42)])
    controller = RouteController()
    position, speed = 0., 8.
    transform = env.ego.get_transform()
    env.ego.get_speed_limit = lambda: 28.8
    env.ego.get_velocity = lambda: carla.Vector3D(x=speed)
    stopped_position = None
    for _ in range(400):
        transform.location = carla.Location(x=position)
        state, observation = env._observe(42)
        requested, _ = controller.act(observation, None)
        result = env.supervisor.step(state, requested, env.cfg.dt)
        next_speed = max(0., speed + result.action[1]*1.5*env.cfg.dt)
        position += .5*(speed+next_speed)*env.cfg.dt
        speed = next_speed
        assert position < 40.  # Front bumper remains before the stop waypoint.
        if speed == 0.:
            stopped_position = position if stopped_position is None else stopped_position
            assert position == stopped_position and result.action[1] == -1.
    assert stopped_position is not None and 1.5 < 40.-position < 7.
    phase[0] = carla.TrafficLightState.Green
    state, observation = env._observe(42)
    assert not state.red_light
    request, _ = controller.act(observation, None)
    assert env.supervisor.step(state, request, env.cfg.dt).action[1] > 0.


@pytest.mark.parametrize("observations_valid", [True, False])
@pytest.mark.parametrize("distance,queue,crossing", [(2., False, None), (25., True, None), (12., False, 3.)])
def test_red_light_wait_pauses_reward_and_episode_watchdogs(distance, queue, crossing, observations_valid):
    state = DrivingState(speed_limit_mps=10., red_light=True, red_light_distance_m=distance,
                         red_light_queue=queue, crossing_distance_m=crossing, observations_valid=observations_valid)
    env = fake_step_environment(before=state, after=state, max_steps=1)
    for _ in range(10):
        _, _, terminated, truncated, info = env.step((0., 0.))
        assert not terminated and not truncated
        assert info["reward_components"]["time"] == 0.
        assert info["reward_components"]["idle"] == 0.
        assert info["reward_components"]["target_tracking"] == 0.
    env.next_state = replace(state, red_light=False, red_light_queue=False)
    _, _, _, truncated, info = env.step((0., 0.))
    assert truncated and info["truncation_reason"] == "time_limit"
    assert info["reward_components"]["time"] < 0.


def test_distant_red_does_not_pause_time():
    state = DrivingState(speed_limit_mps=10., red_light=True, red_light_distance_m=20.)
    env = fake_step_environment(before=state, after=state, max_steps=1)
    _, _, _, truncated, info = env.step((0., 0.))
    assert truncated and info["reward_components"]["time"] < 0.


@pytest.mark.parametrize("signal", ["red", "yellow"])
def test_arrival_in_signal_stop_area_waives_stale_approach_tracking_target(signal):
    fields = {signal + "_light": True, signal + "_light_distance_m": 2.51}
    if signal == "yellow":
        fields["yellow_light_confidence"] = 1.
    before = DrivingState(speed_mps=.3, speed_limit_mps=10., **fields)
    after = replace(before, speed_mps=.1, **{signal + "_light_distance_m": 2.49})
    # The current transition still carries the moving, pre-step speed target.
    result = SafetyResult((0., -1.), target_speed_mps=1.)
    env = fake_step_environment(before=before, after=after, result=result, max_steps=1)
    _, _, terminated, truncated, info = env.step((0., -1.))
    assert not terminated and not truncated
    for component in ("time", "idle", "target_tracking"):
        assert info["reward_components"][component] == 0.
    assert info["reward_components"]["jerk"] < 0.  # Motion costs remain auditable.


@pytest.mark.parametrize("distance,accepted", [
    (2.49, True), (2.5, True), (2.501, False), (4.83539, False), (6.75, False),
])
def test_stationary_red_wait_exemptions_follow_two_point_five_metre_area(distance, accepted):
    state = DrivingState(speed_limit_mps=10., red_light=True, red_light_distance_m=distance)
    env = fake_step_environment(before=state, after=state, max_steps=1)
    _, _, _, truncated, info = env.step((0., -1.))
    assert truncated is not accepted
    assert info["truncation_reason"] == (None if accepted else "time_limit")
    for component in ("time", "idle", "target_tracking"):
        if accepted:
            assert info["reward_components"][component] == 0.
        else:
            assert info["reward_components"][component] < 0.


def test_queue_must_connect_stopped_vehicles_to_red_light():
    env, _ = fake_environment()
    light = NS(get_state=lambda: carla.TrafficLightState.Red)
    env.traffic_stops = [(light, 25.)]
    def vehicle(actor_id, x, speed=0.):
        return NS(id=actor_id, type_id="vehicle.test", is_alive=True, attributes={},
                  get_location=lambda: carla.Location(x=x),
                  get_transform=lambda: carla.Transform(carla.Location(x=x)),
                  get_velocity=lambda: carla.Vector3D(x=speed),
                  bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1)))
    env.world.get_actors = lambda: [vehicle(3, 21.), vehicle(4, 12.), vehicle(5, 5.)]
    env._observe(42)
    assert env.oracle_state.red_light_queue
    env.world.get_actors = lambda: [vehicle(3, 21., 2.), vehicle(5, 5.)]
    env._observe(42)
    assert not env.oracle_state.red_light_queue


def test_scenario_spawns_vehicle_adult_and_child():
    spawned = []
    def spawn(bp, transform):
        actor = NS(type_id=bp.id)
        spawned.append(actor)
        return actor
    env = spawn_test_environment(1, spawn)
    env.scenario = {"actors": [{"kind": "vehicle", "x": 10}]}
    env._spawn_hazards()
    assert {actor.type_id for actor in spawned} == {
        "vehicle.test", "walker.pedestrian.0001", "walker.pedestrian.0011"}


def test_external_yellow_state_reaches_observation_without_ground_truth():
    fields = safe_external_fields()
    fields.update(yellow_light=True, yellow_light_distance_m=12., yellow_light_confidence=.7)
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), driving=fields, frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    state, observation = env._observe(42)
    assert state.observations_valid and state.yellow_light and not env.oracle_state.yellow_light
    np.testing.assert_allclose(observation["tokens"][2, 5:8], [1., .12, .7])


@pytest.mark.parametrize("invalid", [-1., 0., 1.1, float("nan")])
def test_external_invalid_yellow_confidence_fails_safe(invalid):
    fields = safe_external_fields()
    fields.update(yellow_light=True, yellow_light_distance_m=12., yellow_light_confidence=invalid)
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), driving=fields, frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    state, _ = env._observe(42)
    assert not state.observations_valid
    assert SafetySupervisor().step(state, (0., 1.), .1).action == (0., -1.)


def test_external_missing_yellow_contract_fails_safe():
    fields = safe_external_fields()
    del fields["yellow_light"]
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32), driving=fields, frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    assert not env._observe(42)[0].observations_valid


def test_episode_return_cutoff_is_strict_and_terminates_before_time_limit(monkeypatch):
    from dreamerv3_attention import environment as module
    rewards = iter([-500., -.25])
    monkeypatch.setattr(module, "compute_reward", lambda *args, **kwargs:
                        NS(reward=next(rewards), components={"time": 0.}))
    env = fake_step_environment(timeout_s=10., blocked_timeout_s=20., max_steps=2)
    _, reward, terminated, truncated, info = env.step((0., 0.))
    assert reward == -500. and not terminated and not truncated
    assert info["episode_return"] == -500. and info["termination_reason"] is None
    _, reward, terminated, truncated, info = env.step((0., 0.))
    assert reward == -.25 and terminated and not truncated
    assert info["episode_return"] == -500.25
    assert info["termination_reason"] == "reward_limit" and info["truncation_reason"] is None


@pytest.mark.parametrize("direction", [-1., 1.])
def test_junction_departure_uses_planned_turn_lane_and_still_flags_adjacent_lane(direction):
    env, _ = fake_environment()
    env.route = Route([(0., 0.), (10., 0.), (10., direction * 10.), (10., direction * 20.)],
                      lane_widths=[4., 4., 4., 4.], junctions=[False, True, True, False])
    # The map's nearest lane can be narrower and belong to another road.
    env.map.get_waypoint = lambda *_args, **_kwargs: NS(lane_width=2.)
    transform = carla.Transform(carla.Location(x=11.9, y=direction * 5.))
    env.ego.get_transform = lambda: transform
    state, _ = env._observe(42)
    assert state.lane_width_m == 4. and not state.lane_departure and not state.offroad
    transform.location.x = 12.6
    state, _ = env._observe(42)
    assert state.lane_departure and not state.offroad
    env.map.get_waypoint = lambda *_args, **kwargs: None if kwargs.get("project_to_road") is False else NS(lane_width=2.)
    state, _ = env._observe(42)
    assert state.offroad


def test_nonjunction_lane_departure_has_no_extra_margin():
    env, _ = fake_environment()
    env.route = Route([(0., 0.), (100., 0.)], lane_widths=[4., 4.], junctions=[False, False])
    transform = carla.Transform(carla.Location(x=10., y=2.1))
    env.ego.get_transform = lambda: transform
    state, _ = env._observe(42)
    assert state.lane_departure


def spawn_test_environment(seed, spawn):
    env = object.__new__(CarlaEnv)
    env.carla = carla
    env.cfg = Config(seed=seed)
    env.scenario = {}
    env.ego = NS(get_transform=lambda: carla.Transform())
    env.actors = []
    env.rng = np.random.default_rng(seed)
    env.traffic_manager = None
    env.map = NS(get_waypoint=lambda loc, **_kwargs: NS(
        transform=carla.Transform(carla.Location(x=loc.x, y=round(loc.y/3.5)*3.5, z=loc.z-.5)),
        is_junction=False))
    def blueprint(name):
        return NS(id=name, has_attribute=lambda _name: False)
    vehicle = blueprint("vehicle.test")
    adult = blueprint("walker.pedestrian.0001")
    child = blueprint("walker.pedestrian.0011")
    library = NS(filter=lambda pattern: [vehicle] if pattern == "vehicle.*" else [adult, child])
    nav_index = 0
    def nav_location():
        nonlocal nav_index
        nav_index += 1
        return carla.Location(x=8.+(nav_index % 48), y=6. if nav_index % 2 else -6.)
    env.world = NS(get_blueprint_library=lambda: library, try_spawn_actor=spawn,
                   set_pedestrians_seed=Mock(), set_pedestrians_cross_factor=Mock(),
                   get_random_location_from_navigation=nav_location)
    return env


def test_random_spawn_retries_and_required_actors_spawn_first():
    attempts = []
    def spawn(bp, transform):
        attempts.append((bp.id, transform.location.x, transform.location.y))
        if len(attempts) in (1, 5):
            return None
        return NS(type_id=bp.id)
    env = spawn_test_environment(7, spawn)
    env._spawn_hazards()
    assert attempts[0][0] == attempts[1][0] == "vehicle.test"
    assert attempts[0][1:] != attempts[1][1:]
    assert [actor.type_id for actor in env.actors[:3]] == [
        "vehicle.test", "walker.pedestrian.0001", "walker.pedestrian.0011"]
    assert env.actor_spawn_report["requested_extra"] == env.actor_spawn_report["spawned_extra"]
    assert 6 <= env.actor_spawn_report["spawned_extra"] <= 12
    assert len(attempts) == len(env.actors) + 2


@pytest.mark.parametrize("seed", range(10))
def test_random_traffic_has_at_most_two_extra_vehicles(seed):
    env = spawn_test_environment(seed, lambda bp, _transform: NS(type_id=bp.id))
    env._spawn_hazards()
    extras = env.actors[3:]
    assert 1 <= sum(actor.type_id.startswith("vehicle.") for actor in extras) <= 2
    assert 5 <= sum(actor.type_id.startswith("walker.") for actor in extras) <= 10
    assert env.actors[0].type_id == "vehicle.test"


def test_exhausted_optional_spawns_do_not_abort_or_remove_required_actors(caplog):
    count = 0
    def spawn(bp, _transform):
        nonlocal count
        count += 1
        return NS(type_id=bp.id) if count <= 3 else None
    env = spawn_test_environment(7, spawn)
    env._spawn_hazards()
    assert len(env.actors) == 3
    assert env.actor_spawn_report["spawned_extra"] == 0
    assert count == 3 + 12 * env.actor_spawn_report["requested_extra"]
    assert "optional random actors after retries" in caplog.text


@pytest.mark.parametrize("spec", [
    {"kind": "vehicle", "x": 20.},
    {"kind": "pedestrian", "x": 20.},
    {"kind": "pedestrian", "child": True, "x": 20.},
])
def test_invalid_authored_spawn_is_skipped_and_remaining_actors_spawn(spec, caplog):
    attempts = []
    def spawn(bp, transform):
        attempts.append((bp.id, transform.location.x))
        return None if len(attempts) == 1 else NS(type_id=bp.id)
    env = spawn_test_environment(7, spawn)
    env.scenario = {"name": "blocked_spawn", "actors": [spec]}
    env._spawn_hazards()
    if spec["kind"] == "vehicle":
        assert attempts[0][1] == 20.
    assert len(attempts) == len(env.actors) + 1
    assert len(env.actors) == 2 + env.actor_spawn_report["spawned_extra"]
    assert not any(spawn_spec is spec for _actor, spawn_spec in env.moving)
    assert f"Skipping authored {spec['kind']}" in caplog.text
    assert "blocked_spawn after 1 failed spawn attempt(s)" in caplog.text


def test_random_spawn_positions_are_reproducible():
    def collect():
        positions = []
        def spawn(bp, transform):
            positions.append((bp.id, transform.location.x, transform.location.y))
            return NS(type_id=bp.id)
        env = spawn_test_environment(7, spawn)
        env._spawn_hazards()
        return positions
    assert collect() == collect()


def test_all_invalid_spawns_are_skipped_after_bounded_retries(caplog):
    attempts = []
    def spawn(bp, _transform):
        attempts.append(bp.id)
        return None
    env = spawn_test_environment(7, spawn)
    env._spawn_hazards()
    assert len(attempts) == 12 * (3 + env.actor_spawn_report["requested_extra"])
    assert not env.actors and not env.moving
    assert env.actor_spawn_report["spawned_extra"] == 0
    assert "Skipping default vehicle" in caplog.text
    assert "after 12 failed spawn attempt(s)" in caplog.text


def test_pedestrian_spawns_use_sidewalk_navmesh_and_disable_jaywalking():
    placements = []
    def spawn(bp, transform):
        placements.append((bp.id, transform.location))
        return NS(type_id=bp.id)
    env = spawn_test_environment(7, spawn)
    env.scenario = {"actors": [{"kind": "pedestrian", "x": 20., "y": 0., "yaw": 90., "speed_mps": 1.}]}
    env._spawn_hazards()
    env.world.set_pedestrians_seed.assert_called_once_with(7)
    env.world.set_pedestrians_cross_factor.assert_called_once_with(0.)
    pedestrians = [loc for name, loc in placements if name.startswith("walker.")]
    assert pedestrians and all(abs(loc.y) == 6. for loc in pedestrians)
    assert all(loc.z == .5 for loc in pedestrians)


def test_missing_navigation_skips_pedestrians_instead_of_spawning_in_road(caplog):
    env = spawn_test_environment(7, lambda bp, _transform: NS(type_id=bp.id))
    env.world.get_random_location_from_navigation = lambda: None
    env._spawn_hazards()
    assert env.actors and all(actor.type_id.startswith("vehicle.") for actor in env.actors)
    assert "No nearby sidewalk navigation" in caplog.text


def test_vehicles_align_to_driving_lane_and_map_heading():
    placements = []
    def spawn(bp, transform):
        placements.append((bp.id, transform))
        return NS(type_id=bp.id)
    env = spawn_test_environment(7, spawn)
    road = carla.Transform(carla.Location(x=20., y=3.5, z=.25), carla.Rotation(yaw=90.))
    env.map.get_waypoint = Mock(return_value=NS(transform=road, is_junction=False))
    env.scenario = {"actors": [{"kind": "vehicle", "x": 20., "y": 3., "yaw": 180.}]}
    env._spawn_hazards()
    placed = placements[0][1]
    assert (placed.location.x, placed.location.y, placed.location.z) == (20., 3.5, .75)
    assert placed.rotation.yaw == 90.
    assert env.map.get_waypoint.call_args_list[0].kwargs == {"lane_type": carla.LaneType.Driving}


@pytest.mark.parametrize("road", [None, NS(is_junction=True, transform=carla.Transform()),
                                 NS(is_junction=False, transform=carla.Transform(carla.Location(x=1000.)))])
def test_vehicles_skip_invalid_lane_locations(road):
    env = spawn_test_environment(7, lambda bp, _transform: NS(type_id=bp.id))
    env.map.get_waypoint = lambda *_args, **_kwargs: road
    env._spawn_hazards()
    assert env.actors and all(actor.type_id.startswith("walker.") for actor in env.actors)


def npc_control_environment():
    env = object.__new__(CarlaEnv)
    env.carla, env.cfg = carla, Config()
    env.actors, env.walker_controllers, env.traffic_manager = [], [], None
    manager = Mock()
    manager.get_port.return_value = 8000
    env.client = NS(get_trafficmanager=Mock(return_value=manager))
    env.world = NS(tick=Mock(return_value=42), get_random_location_from_navigation=Mock(
        return_value=carla.Location(x=10., y=6.)))
    return env, manager


def test_episode_seed_reaches_native_pedestrians_and_reused_traffic_manager():
    spawn_env = spawn_test_environment(7, lambda bp, _transform: NS(type_id=bp.id))
    spawn_env.episode_seed = 1234
    spawn_env._spawn_hazards()
    spawn_env.world.set_pedestrians_seed.assert_called_once_with(1234)
    env, manager = npc_control_environment()
    actor = Mock(type_id="vehicle.test")
    actor.get_speed_limit.return_value = 30.
    env.moving = [(actor, {"speed_mps": 2.})]
    for seed in (1234, 5678):
        env.episode_seed = seed
        env._start_npcs()
        manager.set_random_device_seed.assert_called_with(seed)
    assert manager.set_random_device_seed.call_count == 2


def test_step_logs_actual_route_variant_and_episode_seed():
    env = fake_step_environment()
    env.route_start, env.route_end, env.episode_seed = 4, 9, 1234
    *_, info = env.step((0., 0.))
    assert (info["route_start"], info["route_end"], info["episode_seed"]) == (4, 9, 1234)


def test_moving_vehicle_uses_rule_following_autopilot_and_current_speed_limit():
    env, manager = npc_control_environment()
    actor = Mock(type_id="vehicle.test")
    actor.get_speed_limit.return_value = 30.
    env.moving = [(actor, {"speed_mps": 20.})]
    env._start_npcs()
    manager.set_synchronous_mode.assert_called_once_with(True)
    manager.set_random_device_seed.assert_called_once_with(env.cfg.seed)
    actor.set_autopilot.assert_called_once_with(True, 8000)
    for rule in (manager.ignore_lights_percentage, manager.ignore_signs_percentage,
                 manager.ignore_walkers_percentage, manager.ignore_vehicles_percentage,
                 manager.random_left_lanechange_percentage, manager.random_right_lanechange_percentage):
        rule.assert_called_once_with(actor, 0.)
    manager.auto_lane_change.assert_called_once_with(actor, False)
    manager.distance_to_leading_vehicle.assert_called_once_with(actor, 5.)
    manager.update_vehicle_lights.assert_called_once_with(actor, True)
    manager.set_desired_speed.assert_called_once_with(actor, 24.)
    actor.get_speed_limit.return_value = 10.
    env._update_npcs()
    manager.set_desired_speed.assert_called_with(actor, 8.)
    env.moving[0][1]["speed_mps"] = 1.
    env._update_npcs()
    manager.set_desired_speed.assert_called_with(actor, 3.6)
    actor.get_speed_limit.return_value = float("nan")
    env._update_npcs()
    manager.set_desired_speed.assert_called_with(actor, 0.)
    actor.set_target_velocity.assert_not_called()


def test_authored_stationary_vehicle_remains_braked_without_autopilot():
    env, _ = npc_control_environment()
    actor = Mock(type_id="vehicle.test")
    env.moving = [(actor, {"speed_mps": 0.})]
    env._start_npcs()
    control = actor.apply_control.call_args.args[0]
    assert control.brake == 1. and control.hand_brake
    actor.set_autopilot.assert_not_called()
    env.client.get_trafficmanager.assert_not_called()


def test_walker_uses_native_navigation_after_snapshot_and_preserves_collisions():
    env, _ = npc_control_environment()
    walker = Mock(type_id="walker.pedestrian.0001")
    controller = Mock(type_id="controller.ai.walker")
    env.moving = [(walker, {"speed_mps": 3., "yaw": 180.})]
    env.world.get_blueprint_library = lambda: NS(find=lambda _name: "controller.ai.walker")
    env.world.try_spawn_actor = Mock(return_value=controller)
    controller.start.side_effect = lambda: env.world.tick.assert_called_once()
    env._start_npcs()
    controller.start.assert_called_once_with()
    controller.set_max_speed.assert_called_once_with(1.4)
    walker.set_collisions.assert_called_once_with(True)
    controller.go_to_location.assert_called_once_with(env.world.get_random_location_from_navigation.return_value)
    assert env.world.try_spawn_actor.call_args.kwargs == {"attach_to": walker}
    assert env.walker_controllers == [controller] and env.actors == [controller]
    walker.apply_control.assert_not_called()


def test_step_does_not_force_vehicle_velocity_or_pedestrian_heading():
    env = fake_step_environment()
    vehicle = Mock(type_id="vehicle.test")
    walker = Mock(type_id="walker.pedestrian.0001")
    vehicle.get_speed_limit.return_value = 30.
    env.traffic_manager = Mock()
    env.moving = [(vehicle, {"speed_mps": 3.}), (walker, {"speed_mps": 1.})]
    env.step((0., 0.))
    env.traffic_manager.set_desired_speed.assert_called_once_with(vehicle, 10.8)
    vehicle.set_target_velocity.assert_not_called()
    walker.apply_control.assert_not_called()


def test_npc_cleanup_stops_controllers_before_destroy_and_releases_synchrony():
    env, manager = npc_control_environment()
    calls = []
    vehicle = Mock(type_id="vehicle.test", is_alive=True)
    walker = Mock(type_id="walker.pedestrian.0001", is_alive=True)
    controller = Mock(type_id="controller.ai.walker", is_alive=True)
    controller.stop.side_effect = lambda: calls.append("stop")
    controller.destroy.side_effect = lambda: calls.append("controller")
    walker.destroy.side_effect = lambda: calls.append("walker")
    env.actors = [vehicle, walker, controller]
    env.moving = [(vehicle, {"speed_mps": 3.}), (walker, {"speed_mps": 1.})]
    env.walker_controllers = [controller]
    env.traffic_manager, env.rig, env.closed, env.provider, env.owns_settings = manager, None, False, None, True
    env.settings, env.weather = "settings", "weather"
    env.world.apply_settings, env.world.set_weather = Mock(), Mock()
    env.close()
    assert calls == ["stop", "controller", "walker"]
    vehicle.set_autopilot.assert_called_once_with(False, 8000)
    assert not env.actors and not env.moving and not env.walker_controllers
    manager.set_synchronous_mode.assert_called_once_with(False)
    env.world.apply_settings.assert_called_once_with("settings")
    env.world.set_weather.assert_called_once_with("weather")
    env.close()
    controller.stop.assert_called_once()


def test_occupied_crossing_spawn_stays_inside_marked_crossing():
    positions = []
    def spawn(bp, transform):
        positions.append((bp.id, transform.location.x, transform.location.y))
        return NS(type_id=bp.id)
    env = spawn_test_environment(7, spawn)
    env.route, env.crossings = Route([(0., 0.), (100., 0.)]), [("0", 19., 21.)]
    env.scenario = {"actors": [{"kind": "pedestrian", "at_crossing": True, "speed_mps": 0.}]}
    env._spawn_hazards()
    assert positions[0] == ("walker.pedestrian.0001", 20., 0.)


DESTROYED_ACTOR_ERROR = ("trying to operate on a destroyed actor; an actor's function was called, "
                         "but the actor is already destroyed.")


@pytest.mark.parametrize("failure", ["not_alive", "is_alive", "get_speed_limit", "set_desired_speed"])
def test_destroyed_npc_is_removed_from_updates_without_losing_live_npcs(failure):
    from unittest.mock import PropertyMock

    env, manager = npc_control_environment()
    env.traffic_manager = manager
    dead = Mock(type_id="vehicle.dead", is_alive=failure != "not_alive")
    live = Mock(type_id="vehicle.live", is_alive=True)
    live.get_speed_limit.return_value = dead.get_speed_limit.return_value = 30.
    if failure == "is_alive":
        type(dead).is_alive = PropertyMock(side_effect=RuntimeError(DESTROYED_ACTOR_ERROR))
    elif failure == "get_speed_limit":
        dead.get_speed_limit.side_effect = RuntimeError(DESTROYED_ACTOR_ERROR)
    elif failure == "set_desired_speed":
        def desired_speed(actor, _speed):
            if actor is dead:
                raise RuntimeError(DESTROYED_ACTOR_ERROR)
        manager.set_desired_speed.side_effect = desired_speed
    spec = {"speed_mps": 2.}
    env.moving = [(dead, spec), (live, spec)]
    env._update_npcs()
    assert env.moving == [(live, spec)]
    manager.set_desired_speed.assert_called_with(live, 7.2)
    if failure in ("not_alive", "is_alive"):
        dead.get_speed_limit.assert_not_called()
    env._update_npcs()
    assert live.get_speed_limit.call_count == 2


def test_npc_simulator_errors_still_propagate():
    env, manager = npc_control_environment()
    env.traffic_manager = manager
    actor = Mock(type_id="vehicle.test", is_alive=True)
    actor.get_speed_limit.side_effect = RuntimeError("RPC connection lost")
    env.moving = [(actor, {"speed_mps": 2.})]
    with pytest.raises(RuntimeError, match="RPC connection lost"):
        env._update_npcs()


@pytest.mark.parametrize("failure", ["not_alive", "get_location", "get_velocity", "get_transform"])
def test_observation_skips_destroyed_npc_without_partial_hazards(failure):
    env, _ = fake_environment()
    live = env.world.get_actors()[0]
    dead = Mock(id=3, type_id="vehicle.dead", is_alive=failure != "not_alive", attributes={})
    dead.get_location.return_value = carla.Location(x=5.)
    dead.get_velocity.return_value = carla.Vector3D()
    dead.get_transform.return_value = carla.Transform(carla.Location(x=5.))
    dead.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1))
    if failure != "not_alive":
        getattr(dead, failure).side_effect = RuntimeError(DESTROYED_ACTOR_ERROR)
    env.world.get_actors = lambda: [dead, live]
    state, observation = env._observe(42)
    assert state.front_hazard_m == pytest.approx(9.7)
    assert state.pedestrian_risk == 1.
    assert observation["mask"][4:].sum() == 1


def test_observation_actor_connection_errors_still_propagate():
    env, _ = fake_environment()
    actor = env.world.get_actors()[0]
    actor.get_velocity = Mock(side_effect=RuntimeError("RPC connection lost"))
    with pytest.raises(RuntimeError, match="RPC connection lost"):
        env._observe(42)


def test_red_light_queue_reuses_actor_state_instead_of_querying_actor_again():
    env, _ = fake_environment()
    env.traffic_stops = [(NS(get_state=lambda: carla.TrafficLightState.Red), 12.)]
    actor = Mock(id=3, type_id="vehicle.test", is_alive=True, attributes={})
    actor.get_location.side_effect = [carla.Location(x=6.), RuntimeError(DESTROYED_ACTOR_ERROR)]
    actor.get_velocity.return_value = carla.Vector3D()
    actor.get_transform.return_value = carla.Transform(carla.Location(x=6.))
    actor.bounding_box = carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1))
    env.world.get_actors = Mock(return_value=[actor])
    state, _ = env._observe(42)
    assert state.red_light and state.red_light_queue
    actor.get_location.assert_called_once()
    actor.get_velocity.assert_called_once()
    env.world.get_actors.assert_called_once()


@pytest.mark.parametrize("reverse,expected", [
    (True, dict(throttle=.4, brake=0., reverse=True)),
    (False, dict(throttle=0., brake=.4, reverse=False)),
])
def test_step_applies_reverse_as_throttle_and_normal_negative_longitudinal_as_brake(reverse, expected):
    controls = []
    env = fake_step_environment(result=SafetyResult((0., -.4), target_speed_mps=0., reverse=reverse))
    env.ego = NS(apply_control=controls.append)
    env.carla = NS(VehicleControl=lambda **kwargs: NS(**kwargs))
    env.step((0., 0.))
    assert {key: getattr(controls[0], key) for key in expected} == pytest.approx(expected)

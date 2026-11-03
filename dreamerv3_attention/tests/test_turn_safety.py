"""Sharp bends remain drivable beside barricades, with real obstacle stops."""

import json
import math
from pathlib import Path

import carla
import numpy as np
import pytest

from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor
from dreamerv3_attention.navigation import Route
from dreamerv3_attention.sensors import PerceptionPrediction
from dreamerv3_attention.tests.test_environment import fake_environment, safe_external_fields


def radar_environment(radars, *, route=None, transform=None):
    env, bundle = fake_environment()
    env.world.get_actors = lambda: []
    env.route = route or Route([(0., 0.), (100., 0.)])
    env.ego.get_transform = lambda: transform or carla.Transform()
    env.ego.bounding_box.extent = carla.Vector3D(2.3956, .98, 1.)
    bundle.radars = {name: np.asarray(rows, dtype=np.float32).reshape(-1, 4)
                     for name, rows in radars.items()}
    return env


def test_recorded_town01_barricade_bend_allows_route_following():
    path = Path(__file__).with_name("fixtures") / "town01_barricade_bend.json"
    data = json.loads(path.read_text())
    yaw = data["ego_yaw_degrees"]
    transform = carla.Transform(carla.Location(x=data["ego_xy"][0], y=data["ego_xy"][1]),
                                carla.Rotation(yaw=yaw))
    env = radar_environment({name: [row] for name, row in data["radars"].items()},
                            route=Route(data["points"]), transform=transform)
    speed = data["speed_mps"]
    env.ego.get_velocity = lambda: carla.Vector3D(x=speed*math.cos(math.radians(yaw)),
                                                 y=speed*math.sin(math.radians(yaw)))
    state, obs = env._observe(42)
    request, _ = RouteController().act(obs, None)
    result = env.supervisor.step(state, request, .05)
    assert state.front_hazard_m is None  # Barricade is >6 m off the route.
    assert state.side_hazard_m > 3.
    assert request[0] < -.25
    assert result.action[0] == pytest.approx(request[0])
    assert result.action[1] > 0.


def test_front_radar_keeps_farther_on_route_obstacle_after_nearest_off_route_hit():
    env = radar_environment({"front": [[8., 2., 1., 0.], [12., 0., 1., 0.]]})
    state, obs = env._observe(42)
    assert state.front_hazard_m == pytest.approx(12. - 2.3956)
    assert env.radar_hazards["front"]["point_ego_m"] == [12., 0., 1.]
    # Both raw detections remain available to the policy.
    offset = 1 + env.cfg.max_objects + 3
    assert obs["mask"][offset:offset+2].all()
    np.testing.assert_allclose(obs["tokens"][offset:offset+2, :3], [[.08, .02, .01], [.12, 0., .01]])


@pytest.mark.parametrize("speed,distance", [(2., 4.), (8., 12.)])
def test_body_obstacle_inside_stopping_distance_is_kept_when_route_turns_away(speed, distance):
    env = radar_environment({"front": [[distance, 0., 1., 0.]]},
                            route=Route([(0., 0.), (0., 30.)]))
    env.ego.get_velocity = lambda: carla.Vector3D(x=speed)
    state, _ = env._observe(42)
    result = env.supervisor.step(state, (0., 1.), .05)
    assert state.front_hazard_m == pytest.approx(distance - 2.3956)
    assert result.action[1] < 0.
    if distance == 4.:
        assert result.action[1] == -1.
        assert "front_hazard_brake" in result.intervention_reasons


@pytest.mark.parametrize("side,sign", [("left", -1.), ("right", 1.)])
def test_parallel_barricade_clearance_uses_vehicle_width(side, sign):
    env = radar_environment({side: [[.8, sign*4.2, 1., 0.]]})
    state, _ = env._observe(42)
    assert state.side_hazard_m == pytest.approx(4.2-.98)
    assert getattr(state, side + "_hazard_m") == pytest.approx(4.2-.98)
    assert env.supervisor.step(state, (sign*.7, 0.), .05).action[0] == pytest.approx(sign*.7)


@pytest.mark.parametrize("side,sign", [("left", -1.), ("right", 1.)])
def test_near_side_obstacle_limits_only_steering_toward_it(side, sign):
    env = radar_environment({side: [[0., sign*2., 1., 0.]]})
    state, _ = env._observe(42)
    toward = SafetySupervisor().step(state, (sign*.7, 0.), .05)
    away = SafetySupervisor().step(state, (-sign*.7, 0.), .05)
    assert toward.action[0] == pytest.approx(sign*.25)
    assert away.action[0] == pytest.approx(-sign*.7)


@pytest.mark.parametrize("steer", [-.7, .7])
def test_obstacles_on_both_sides_limit_both_turn_directions(steer):
    env = radar_environment({"left": [[0., -2., 1., 0.]], "right": [[0., 2., 1., 0.]]})
    state, _ = env._observe(42)
    assert env.supervisor.step(state, (steer, 0.), .05).action[0] == pytest.approx(math.copysign(.25, steer))


def test_side_direction_uses_decoded_ego_coordinates():
    env = radar_environment({"left": [[0., 2., 1., 0.]]})
    state, _ = env._observe(42)
    assert state.left_hazard_m is None
    assert state.right_hazard_m == pytest.approx(2.-.98)
    assert env.supervisor.step(state, (-.7, 0.), .05).action[0] == pytest.approx(-.7)


def test_undirected_provider_hazard_still_limits_both_directions():
    state = DrivingState(side_hazard_m=.5, right_hazard_m=2.)
    for steer in (-.7, .7):
        result = SafetySupervisor().step(state, (steer, 0.), .05)
        assert result.action[0] == pytest.approx(math.copysign(.25, steer))


@pytest.mark.parametrize("distance", [.1, float("nan")])
def test_very_close_or_malformed_side_clearance_retains_symmetric_guard(distance):
    state = DrivingState(side_hazard_m=.1, right_hazard_m=distance)
    for steer in (-.7, .7):
        result = SafetySupervisor().step(state, (steer, 0.), .05)
        assert result.action[0] == pytest.approx(math.copysign(.25, steer))


def test_external_mode_does_not_leak_ground_truth_hazard_directions():
    prediction = PerceptionPrediction(np.zeros((0, 16), np.float32),
                                     driving=safe_external_fields(), frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    actor = env.world.get_actors()[0]
    actor.get_location = lambda: carla.Location(x=2., y=2.)
    state, _ = env._observe(42)
    assert env.oracle_state.right_hazard_m is not None
    assert state.side_hazard_m is state.left_hazard_m is state.right_hazard_m is None


@pytest.mark.parametrize("cruise_cap", [3., 0.])
def test_controller_completes_recorded_bend_beside_barricades(cruise_cap):
    data = json.loads((Path(__file__).with_name("fixtures") / "town01_barricade_bend.json").read_text())
    yaw = math.radians(data["ego_yaw_degrees"])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    barriers = np.asarray([np.asarray(data["ego_xy"]) + rotation @ np.asarray(row[:2])
                           for row in data["radars"].values() if row[2] > .25])
    route = Route(data["points"])
    env = radar_environment({}, route=route)
    bundle = env.rig.get(42)
    env.ego.get_speed_limit = lambda: 30.
    config = DrivingConfig(max_speed_mps=cruise_cap)
    controller, env.supervisor = RouteController(config=config), SafetySupervisor(config)
    position = route.points[0].copy()
    yaw = math.atan2(route.delta[0, 1], route.delta[0, 0])
    speed = steer = max_error = 0.
    for _ in range(800):
        transform = carla.Transform(carla.Location(x=position[0], y=position[1]),
                                    carla.Rotation(yaw=math.degrees(yaw)))
        env.ego.get_transform = lambda: transform
        env.ego.get_velocity = lambda: carla.Vector3D(x=speed*math.cos(yaw), y=speed*math.sin(yaw))
        rotation = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        points = (rotation @ (barriers-position).T).T
        angles, distance = np.arctan2(points[:, 1], points[:, 0]), np.linalg.norm(points, axis=1)
        for name, bearing in (("front", 0.), ("left", -math.pi/2), ("right", math.pi/2), ("rear", math.pi)):
            visible = (abs(np.arctan2(np.sin(angles-bearing), np.cos(angles-bearing))) < math.radians(15))
            xy = points[visible & (distance < 40.)]
            bundle.radars[name] = np.c_[xy, np.ones(len(xy)), np.zeros(len(xy))].astype(np.float32)
        state, obs = env._observe(42)
        result = env.supervisor.step(state, controller.act(obs, None)[0], .05)
        max_error = max(max_error, abs(state.lane_error_m))
        if state.route_progress_m > route.length-3.:
            break
        # Offline bicycle plant with steering lag; no CARLA tyre simulation.
        steer += (.05/.15) * (result.action[0]-steer)
        acceleration = 1.5*max(0., result.action[1]) + 3.*min(0., result.action[1]) - .03*speed
        next_speed = max(0., speed + .05*acceleration)
        average = .5*(speed+next_speed)
        position += average*.05*np.array([math.cos(yaw), math.sin(yaw)])
        yaw += average/2.8*math.tan(steer*.7)*.05
        speed = next_speed
    assert state.route_progress_m > route.length-3.
    assert max_error < .5

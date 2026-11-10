"""Road radar returns must not mask real obstacles or force braking."""

from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from dreamerv3_attention.environment import ego_coordinates
from dreamerv3_attention.navigation import Route
from dreamerv3_attention.sensors import PerceptionPrediction
from dreamerv3_attention.tests.test_environment import fake_environment, safe_external_fields
from dreamerv3_attention.tests.test_turn_safety import radar_environment


def mapped_road(env, transform=None):
    road = NS(lane_width=3.5, transform=transform or carla.Transform())
    env.map.get_waypoint = lambda *_args, **_kwargs: road


@pytest.mark.parametrize("mode", ["ground_truth", "external", "mixed"])
def test_episode1768_road_return_does_not_mask_distant_vehicle(mode):
    # First supervisor-added brake: frames=1293940, speed=4.068 m/s.
    road = [13.22402668, -.24539898, -.01236951, -4.0683]
    vehicle = [35., 0., .7, 0.]
    env = radar_environment({"front": [road, vehicle]})
    mapped_road(env)
    env.ego.get_velocity = lambda: carla.Vector3D(x=4.06830447)
    env.ego.get_speed_limit = lambda: 30.
    env.world.get_weather = lambda: NS(precipitation=0., wetness=60.,
                                      sun_altitude_angle=60., fog_density=0.)
    env.cfg.perception = mode
    if mode != "ground_truth":
        fields = dict(safe_external_fields(), speed_limit_mps=8.33333333)
        env.provider = NS(predict=lambda _: PerceptionPrediction(
            np.zeros((0, 16), np.float32), driving=fields, frame=42, timestamp=2.1))
    state, obs = env._observe(42)
    result = env.supervisor.step(state, (.05503, .75657946), .05)
    assert state.front_hazard_m == pytest.approx(35. - 2.3956)
    assert env.radar_hazards["front"]["point_ego_m"] == pytest.approx(vehicle[:3])
    assert result.action[1] == pytest.approx(.75657946)
    assert "speed_target_brake" not in result.intervention_reasons
    offset = 1 + env.cfg.max_objects + 3
    assert obs["mask"][offset:offset+2].all()
    np.testing.assert_allclose(obs["tokens"][offset:offset+2, :3],
                               np.asarray([road, vehicle])[:, :3]/100., atol=1e-8)


@pytest.mark.parametrize("name,xy", [("front", (13., 0.)), ("rear", (-13., 0.)),
                                   ("left", (0., -13.)), ("right", (0., 13.))])
def test_road_only_scan_has_no_hazard(name, xy):
    env = radar_environment({name: [[*xy, .02, 0.]]})
    mapped_road(env)
    state, _ = env._observe(42)
    assert name not in env.radar_hazards
    assert state.front_hazard_m is state.side_hazard_m is state.rear_hazard_m is None


@pytest.mark.parametrize("grade", [-8., 8.])
def test_road_height_uses_point_local_plane_with_grade_bank_and_ego_pitch(grade):
    road = carla.Transform(carla.Location(z=4.), carla.Rotation(pitch=grade, yaw=20., roll=4.))
    ego = carla.Transform(carla.Location(z=4.), carla.Rotation(pitch=-3., yaw=20., roll=-2.))
    world_points = [road.transform(carla.Location(x=x, z=z)) for x, z in ((13., 0.), (20., .3))]
    points = [list(ego_coordinates(ego, point)) + [0.] for point in world_points]
    assert abs(points[0][2]) > 1.  # Road height differs from ego-origin height.
    end = road.transform(carla.Location(x=100.))
    env = radar_environment({"front": points}, transform=ego,
                            route=Route([(0., 0.), (end.x, end.y)]))
    mapped_road(env, road)
    env._observe(42)
    assert env.radar_hazards["front"]["point_ego_m"] == pytest.approx(points[1][:3])


@pytest.mark.parametrize("height", [.15, .7, 1.5])
def test_close_obstacles_above_road_keep_emergency_braking(height):
    env = radar_environment({"front": [[13., 0., 0., 0.], [3., 0., height, 0.]]})
    mapped_road(env)
    state, _ = env._observe(42)
    result = env.supervisor.step(state, (0., 1.), .05)
    assert state.front_hazard_m == pytest.approx(3. - 2.3956)
    assert result.action[1] == -1.
    assert "front_hazard_brake" in result.intervention_reasons


@pytest.mark.parametrize("missing", ["offroad", "transform"])
def test_unknown_road_surface_retains_low_return(missing):
    env = radar_environment({"front": [[3., 0., 0., 0.]]})
    env.map.get_waypoint = lambda _loc, **kwargs: (
        None if missing == "offroad" and kwargs.get("project_to_road") is False
        else NS(lane_width=3.5))
    state, _ = env._observe(42)
    assert state.front_hazard_m == pytest.approx(3. - 2.3956)


def test_ground_filter_does_not_remove_independent_actor_hazard():
    env = radar_environment({"front": [[13., 0., 0., 0.]]})
    mapped_road(env)
    actors = fake_environment()[0].world.get_actors()
    actors[0].get_location = lambda: carla.Location(x=3., y=0.)
    env.world.get_actors = lambda: actors
    state, _ = env._observe(42)
    assert "front" not in env.radar_hazards
    assert state.front_hazard_m == pytest.approx(3. - 2.3956 - .3)
    assert env.supervisor.step(state, (0., 1.), .05).action[1] == -1.

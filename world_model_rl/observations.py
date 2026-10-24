"""Ground-truth provider and the extension point for a future perception model."""

import importlib
import math
from typing import Protocol

import numpy as np

from .control import LIGHT_WAIT_DISTANCE_M, WAIT_SPEED_MPS, visibility_range, weather_limits
from .geometry import TrackedSlots, ego_relative
from .schema import (CONTEXT, OBJECT_OFFSET, PEDESTRIAN_SLOTS, VEHICLE_SLOTS,
                     Observation)


CHILD_MODELS = ("0009", "0010", "0011", "0012", "0013", "0014", "0048", "0049")


def is_child(actor):
    age = actor.attributes.get("age", "").lower()
    return age == "child" or (not age and actor.type_id.rsplit(".", 1)[-1] in CHILD_MODELS)


class ObservationProvider(Protocol):
    """Factory(env) -> provider. Custom providers own and synchronize any camera sensors.

    The oracle argument permits explicit hybrids, e.g. measured ego/map + predicted objects.
    Set `source` to describe that provenance; never label such observations ground truth.
    """
    source: str

    def reset(self): ...
    def observe(self, env, oracle: Observation) -> Observation: ...
    def close(self): ...


class GroundTruthProvider:
    source = "carla_ground_truth_with_geometric_visibility"

    def reset(self):
        pass

    def observe(self, env, oracle):
        return oracle

    def close(self):
        pass


def load_provider(spec, env):
    if spec == "ground_truth":
        return GroundTruthProvider()
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("observation_provider must be ground_truth or module:factory")
    provider = getattr(importlib.import_module(module), name)(env)
    if not getattr(provider, "source", "") or provider.source == GroundTruthProvider.source:
        raise ValueError("Custom providers must declare their actual observation source")
    return provider


def perception_caution(observation, cfg):
    """Recompute the policy target from predictions, preventing an oracle-child shortcut."""
    c = observation.continuous
    objects = c[OBJECT_OFFSET:].reshape(VEHICLE_SLOTS + PEDESTRIAN_SLOTS, -1)
    child_near = np.any(observation.present & (observation.classes == 2) &
        (objects[:, 0] > 0) & (objects[:, 0] <= cfg.child_caution_metres) &
        (np.abs(objects[:, 1]) <= cfg.child_lateral_metres))
    target = weather_limits(c[11], c[16], c[17], c[18], c[19], cfg)
    c[12] = min(target, cfg.child_speed_mps) if child_near else target
    return observation


class OracleExtractor:
    def __init__(self, env):
        self.env = env
        self.vehicles = TrackedSlots(VEHICLE_SLOTS)
        self.pedestrians = TrackedSlots(PEDESTRIAN_SLOTS)

    def visible(self, actor, ego_transform, distance_limit):
        env, carla = self.env, self.env.carla
        target = actor.get_location()
        yaw = math.radians(ego_transform.rotation.yaw)
        relative = ego_relative([target.x - ego_transform.location.x,
                                 target.y - ego_transform.location.y], yaw)
        distance = float(np.linalg.norm(relative))
        if relative[0] <= 0 or distance > distance_limit or abs(math.atan2(relative[1], relative[0])) > (
                math.radians(env.cfg.environment.field_of_view_degrees) / 2):
            return False
        # Start outside the ego body. Require an unobstructed ray to the target's body centre.
        origin = ego_transform.transform(carla.Location(x=2.6, z=1.5))
        centre = actor.get_transform().transform(actor.bounding_box.location)
        centre.z += 0.2
        target_distance = origin.distance(centre)
        if target_distance <= 1:
            return True
        # cast_ray has no actor ID; hits inside the target's bounding radius are accepted.
        allowance = math.hypot(actor.bounding_box.extent.x, actor.bounding_box.extent.y)
        for hit in env.world.cast_ray(origin, centre):
            if origin.distance(hit.location) < target_distance - allowance - 0.2:
                return False
        return True

    def measure(self):
        env, cfg, carla = self.env, self.env.cfg.environment, self.env.carla
        ego, world = env.ego, env.world
        transform = ego.get_transform()
        location = transform.location
        yaw = math.radians(transform.rotation.yaw)
        velocity, acceleration = ego.get_velocity(), ego.get_acceleration()
        local_velocity = ego_relative([velocity.x, velocity.y], yaw)
        local_acceleration = ego_relative([acceleration.x, acceleration.y], yaw)
        road = env.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        exact_road = env.map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Driving)
        if road is None:
            raise RuntimeError("Map has no driving waypoint near ego")
        road_yaw = math.radians(road.transform.rotation.yaw)
        lane_offset = ego_relative([location.x - road.transform.location.x,
                                    location.y - road.transform.location.y], road_yaw)[1]
        heading = yaw - road_yaw
        speed_limit = float(ego.get_speed_limit()) / 3.6
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            raise RuntimeError("CARLA did not provide a positive speed limit")
        weather = world.get_weather()
        weather_limit = weather_limits(speed_limit, weather.precipitation, weather.wetness,
                                       weather.fog_density, weather.sun_altitude_angle, cfg)
        distance_limit = visibility_range(weather.precipitation, weather.fog_density,
                                          weather.sun_altitude_angle, cfg)
        vehicle_candidates, pedestrian_candidates, child_near = [], [], False
        for actor in world.get_actors():
            vehicle = actor.type_id.startswith("vehicle.")
            pedestrian = actor.type_id.startswith("walker.pedestrian.")
            if actor.id == ego.id or not (vehicle or pedestrian):
                continue
            if location.distance(actor.get_location()) > distance_limit:
                continue
            if not self.visible(actor, transform, distance_limit):
                continue
            pos, vel = actor.get_location(), actor.get_velocity()
            relative = ego_relative([pos.x - location.x, pos.y - location.y], yaw)
            relative_velocity = ego_relative([vel.x - velocity.x, vel.y - velocity.y], yaw)
            child = pedestrian and is_child(actor)
            label = 0 if vehicle else (2 if child else 1)
            child_near |= child and 0 < relative[0] <= cfg.child_caution_metres and abs(
                relative[1]) <= cfg.child_lateral_metres
            features = np.array([*relative, *relative_velocity, actor.bounding_box.extent.x,
                                 actor.bounding_box.extent.y, 1.0], np.float32)
            priority = float(np.linalg.norm(relative)) - (cfg.visibility_metres if child else 0)
            item = (actor.id, priority, (features, label))
            (vehicle_candidates if vehicle else pedestrian_candidates).append(item)

        observation = Observation.empty()
        slots = self.vehicles.assign(vehicle_candidates) + self.pedestrians.assign(pedestrian_candidates)
        identities = self.vehicles.ids + self.pedestrians.ids
        objects = observation.continuous[OBJECT_OFFSET:].reshape(len(slots), -1)
        for i, value in enumerate(slots):
            if value is not None:
                objects[i], observation.classes[i] = value
                observation.present[i] = True
                observation.ids[i] = identities[i]
        progress_distance = env.route.update([location.x, location.y])
        controls = env.upcoming_controls()
        light = controls["light"]
        target = min(weather_limit, cfg.child_speed_mps if child_near else speed_limit)
        observation.controls[:] = [light is not None]
        light_distance = 0 if light is None else light["s"] - env.route.progress - ego.bounding_box.extent.x
        state = "none" if light is None else str(light["actor"].get_state()).split(".")[-1].lower()
        observation.light = np.array({"none": 0, "red": 2, "yellow": 3, "green": 4}.get(state, 1), np.int64)
        observation.continuous[:CONTEXT] = [*local_velocity, *local_acceleration,
            -math.radians(ego.get_angular_velocity().z), *env.previous_action, lane_offset,
            math.sin(heading), math.cos(heading), speed_limit, target,
            max(0, env.route.length - env.route.progress), light_distance, env.remaining_time_seconds,
            weather.precipitation, weather.wetness, weather.fog_density, weather.sun_altitude_angle]
        waypoints = env.route.at(env.route.progress + np.array([5, 10, 20, 30, 40]))
        observation.continuous[CONTEXT:OBJECT_OFFSET] = np.concatenate([
            ego_relative(point - [location.x, location.y], yaw) for point in waypoints])
        front = transform.transform(carla.Location(x=ego.bounding_box.extent.x))
        on_lane = exact_road is not None and abs(lane_offset) <= road.lane_width / 2 and math.cos(heading) > 0.8
        blocked = on_lane and legitimate_wait(light is not None, state, light_distance,
                                              float(np.linalg.norm(local_velocity)))
        return observation.validate(), {
            "speed": float(np.linalg.norm([velocity.x, velocity.y])), "speed_limit": speed_limit,
            "weather_limit": weather_limit, "child_near": bool(child_near),
            "acceleration": np.array([acceleration.x, acceleration.y]),
            "longitudinal_acceleration": float(local_acceleration[0]),
            "offroad": exact_road is None or abs(lane_offset) > road.lane_width / 2,
            "wrong_way": velocity.x * math.cos(road_yaw) + velocity.y * math.sin(road_yaw) < -0.5,
            "progress_distance": progress_distance, "route_s": env.route.progress,
            "success": env.route.length - env.route.progress < 3 and np.linalg.norm(
                np.array([location.x, location.y]) - env.route.points[-1]) < 4,
            "front": np.array([front.x, front.y]), "blocked": blocked,
            "position": np.array([location.x, location.y]),
            "lane_offset": float(lane_offset), "heading_error": math.atan2(math.sin(heading), math.cos(heading))}


def legitimate_wait(has_light, state, light_distance, speed):
    """Pause the deadline only while stopped just before a relevant red/yellow line."""
    return (has_light and state in ("red", "yellow") and
            0 <= light_distance <= LIGHT_WAIT_DISTANCE_M and speed < WAIT_SPEED_MPS)

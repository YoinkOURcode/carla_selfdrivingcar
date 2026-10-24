"""Opt-in collection exposure, sustained controls, and real CARLA hazard actors.

These helpers change actions and scenes only. Ending flags and rewards remain the
responsibility of CarlaDrivingEnv.step, including when a hazard is avoided.
"""

import math
from dataclasses import dataclass

import numpy as np

from .observations import CHILD_MODELS


@dataclass(frozen=True)
class CollectionOptions:
    """Collection-only options, deliberately outside simulator compatibility config."""

    perturbations: bool = False
    staged_hazards: bool = False
    exposure_fraction: float = 0.8

    def validate(self, policy_checkpoint=None):
        """Reject unsupported settings before creating a dataset or simulator session."""
        if not math.isfinite(self.exposure_fraction) or not 0 <= self.exposure_fraction <= 1:
            raise ValueError("exposure_fraction must be finite and in [0, 1]")
        if policy_checkpoint is not None and (self.perturbations or self.staged_hazards):
            raise ValueError("Collection exposure cannot be combined with --policy-checkpoint")

    def for_episode(self, seed):
        """Choose exposure from episode seed, independently of baseline control RNG."""
        exposed = bool(np.random.default_rng([seed, 731]).random() < self.exposure_fraction)
        return {"version": 1, "perturbations": self.perturbations,
                "staged_hazards": self.staged_hazards,
                "exposure_fraction": self.exposure_fraction, "exposed": exposed,
                "disturbances": [], "hazard": None}


class ControlPerturbations:
    """Repeat seeded disturbances with simulated-time warm-up and recovery gaps."""

    def __init__(self, seed, decision_dt, records):
        """Write realized disturbance steps into the episode's provenance list."""
        self.rng = np.random.default_rng([seed, 732])
        self.dt = decision_dt
        self.records = records
        self.next_step = self._steps(3, 8)
        self.end_step = 0
        self.current = None

    def _steps(self, low, high):
        """Round sampled durations up so a disturbance spans whole decisions."""
        return math.ceil(self.rng.uniform(low, high) / self.dt)

    def action(self, baseline, step, speed):
        """Override requested controls only; the environment still limits applied actions."""
        if step >= self.next_step:
            profiles = ("left", "right", "acceleration") if speed >= 2 else ("acceleration",)
            profile = str(self.rng.choice(profiles))
            magnitude = float(self.rng.uniform(0.35, 0.9))
            duration = self._steps(2, 5)
            self.end_step = step + duration
            self.next_step = self.end_step + self._steps(3, 8)
            self.current = {"profile": profile, "start_step": step,
                            "duration_steps": duration, "realized_steps": 0,
                            "steering": None if profile == "acceleration" else
                            (-magnitude if profile == "left" else magnitude),
                            "throttle": float(self.rng.uniform(0.5, 1.0))}
            self.records.append(self.current)
        requested = np.asarray(baseline, dtype=np.float32).copy()
        if step < self.end_step:
            if self.current["steering"] is not None:
                requested[0] = self.current["steering"]
            requested[1:] = [self.current["throttle"], 0]
            self.current["realized_steps"] += 1
        return np.clip(requested, [-1, 0, 0], [1, 1, 1]).astype(np.float32)


class StagedHazard:
    """Own one parked vehicle or directly controlled adult/child crossing.

    Actors join the environment's ownership list and ordinary oracle observations.
    Background actor budgets stay unchanged; hazard counts are recorded separately.
    """

    def __init__(self, env, seed):
        """Place a seeded hazard before background actors and the initial observation."""
        self.env = env
        self.rng = np.random.default_rng([seed, 733])
        self.actor = None
        self.elapsed = 0.0
        self.metadata = {"kind": str(self.rng.choice(("vehicle", "adult", "child"))),
                         "triggered": False, "trigger_frame": None}
        self._spawn()

    def _blueprints(self):
        """Return stable blueprint candidates without requiring background pedestrians."""
        library = self.env.world.get_blueprint_library()
        kind = self.metadata["kind"]
        if kind == "vehicle":
            return sorted([bp for bp in library.filter("vehicle.*")
                           if bp.has_attribute("number_of_wheels") and
                           bp.get_attribute("number_of_wheels").as_int() == 4], key=lambda bp: bp.id)
        walkers = list(library.filter("walker.pedestrian.*"))
        def child(bp):
            """Match the same child assets used by ordinary environment collection."""
            return ((bp.has_attribute("age") and bp.get_attribute("age").as_str().lower() == "child")
                    or (not bp.has_attribute("age") and bp.id.rsplit(".", 1)[-1] in CHILD_MODELS))
        return sorted([bp for bp in walkers if child(bp) == (kind == "child")], key=lambda bp: bp.id)

    def _ground(self, location, vehicle):
        """Validate actual road/sidewalk ground using a short vertical scene ray."""
        carla = self.env.carla
        hits = self.env.world.cast_ray(
            carla.Location(x=location.x, y=location.y, z=location.z + 3),
            carla.Location(x=location.x, y=location.y, z=location.z - 3))
        labels = (carla.CityObjectLabel.Roads,) if vehicle else (
            carla.CityObjectLabel.Roads, carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Ground)
        for hit in hits:
            if hit.label in labels:
                return carla.Location(x=location.x, y=location.y, z=hit.location.z + 0.5)
        return None

    def _clear(self, location):
        """Reject nearby dynamic actors before relying on simulator spawn collision checks."""
        for actor in self.env.world.get_actors():
            if actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                radius = math.hypot(actor.bounding_box.extent.x, actor.bounding_box.extent.y)
                if actor.get_location().distance(location) < radius + 4:
                    return False
        return True

    def _spawn(self):
        """Try bounded route placements, failing explicitly instead of dropping exposure."""
        env, carla = self.env, self.env.carla
        blueprints = self._blueprints()
        if not blueprints:
            raise RuntimeError(f"No blueprints for staged {self.metadata['kind']} hazard")
        upper = min(50.0, env.route.length - 10)
        if upper < 20:
            raise RuntimeError("Staged hazards require at least 30 metres of route")
        vehicle = self.metadata["kind"] == "vehicle"
        for attempt in range(20):
            along = float(self.rng.uniform(20, upper))
            point = env.route.at(along)
            # Route-plan waypoints retain road height, lane identity and junction flags.
            waypoint = min((wp for wp, _ in env.route_plan), key=lambda wp:
                          (wp.transform.location.x - point[0]) ** 2 +
                          (wp.transform.location.y - point[1]) ** 2)
            if waypoint.is_junction:
                continue
            transform = waypoint.transform
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            side = int(self.rng.choice((-1, 1)))
            offset = float(self.rng.uniform(-0.4, 0.4)) if vehicle else side * (waypoint.lane_width / 2 + 0.8)
            location = carla.Location(x=transform.location.x + offset * right.x,
                                      y=transform.location.y + offset * right.y,
                                      z=transform.location.z)
            location = self._ground(location, vehicle)
            if location is None or not self._clear(location):
                continue
            bp = blueprints[int(self.rng.integers(len(blueprints)))]
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "world_model_rl_hazard")
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            rotation = carla.Rotation(yaw=transform.rotation.yaw if vehicle else
                                      math.degrees(math.atan2(-side * right.y, -side * right.x)))
            actor = env.world.try_spawn_actor(bp, carla.Transform(location, rotation))
            if actor is None:
                continue
            env.actors.append(actor)
            self.actor = actor
            self.centre = transform.location
            self.forward = forward
            self.direction = carla.Vector3D(x=-side * right.x, y=-side * right.y, z=0)
            self.crossing_width = 2 * abs(offset)
            self.metadata.update(model=bp.id, actor_id=actor.id, placement_attempts=attempt + 1,
                                 route_metres=along, lane_offset_m=offset,
                                 location=[location.x, location.y, location.z],
                                 trigger_distance_m=float(self.rng.uniform(8, 20)),
                                 speed_mps=0.0 if vehicle else float(self.rng.uniform(0.8, 1.8)),
                                 finished=False)
            if vehicle:
                actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            else:
                actor.apply_control(carla.WalkerControl(speed=0.0))
            return
        raise RuntimeError(f"Could not place staged {self.metadata['kind']} hazard after 20 candidates")

    def tick(self):
        """Trigger and advance crossing controls before each collected physics tick."""
        if self.metadata["kind"] == "vehicle" or not self.actor.is_alive:
            return
        carla = self.env.carla
        if not self.metadata["triggered"]:
            position = self.env.ego.get_location()
            dx, dy = self.centre.x - position.x, self.centre.y - position.y
            along = dx * self.forward.x + dy * self.forward.y
            lateral = abs(dx * self.forward.y - dy * self.forward.x)
            if 0 < along <= self.metadata["trigger_distance_m"] and lateral <= 5:
                self.metadata.update(triggered=True, trigger_frame=self.env.frame + 1)
        speed = 0.0
        if self.metadata["triggered"] and not self.metadata["finished"]:
            speed = self.metadata["speed_mps"]
            self.elapsed += self.env.cfg.environment.fixed_delta_seconds
            self.metadata["finished"] = self.elapsed * speed >= self.crossing_width
        self.actor.apply_control(carla.WalkerControl(direction=self.direction, speed=speed))

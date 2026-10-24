"""Dedicated synchronous CARLA session; owns only the actors it creates."""

import math
import queue
import random
import sys
import time
from pathlib import Path

import numpy as np

from .control import apply_action
from .data import split_for
from .geometry import Route, crosses_line
from .observations import CHILD_MODELS, OracleExtractor, load_provider, perception_caution
from .rewards import RewardCalculator
from .schema import EVENTS, SCHEMA_VERSION


class CarlaDrivingEnv:
    def __init__(self, cfg):
        import carla

        self.cfg, self.carla = cfg, carla
        cfg.validate()
        e = cfg.environment
        agents_path = str(Path(__file__).resolve().parents[1] / "PythonAPI" / "carla")
        if agents_path not in sys.path:
            sys.path.insert(0, agents_path)
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self.client = carla.Client(e.host, e.port)
        self.client.set_timeout(e.timeout)
        self.client_version = self.client.get_client_version()
        self.server_version = self.client.get_server_version()
        if self.client_version != self.server_version:
            raise RuntimeError(f"Client/server versions differ: {self.client_version}/{self.server_version}")
        self.world = self.client.get_world()
        if self.world.get_map().name.rsplit("/", 1)[-1] != e.town:
            self.world = self.client.load_world(e.town)
        self.map = self.world.get_map()
        self.original_settings = self.world.get_settings()
        self.original_weather = self.world.get_weather()
        self.actors, self.controllers, self.sensors = [], [], []
        self.staged_hazard = None
        self.provider = None
        self.tm = None
        self.closed = False
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = e.fixed_delta_seconds
            settings.no_rendering_mode = e.no_rendering_mode
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)
            self.tm = self.client.get_trafficmanager(e.tm_port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_global_distance_to_leading_vehicle(2.5)
            self.planner = GlobalRoutePlanner(self.map, 2.0)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _destroy_actors(self):
        if self.provider is not None:
            self.provider.close()
            self.provider = None
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
        for controller in self.controllers:
            if controller.is_alive:
                controller.stop()
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()
        self.actors, self.controllers, self.sensors = [], [], []
        self.staged_hazard = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.tm is not None:
                # This checkout's TM must be shut down before episode/world teardown.
                self.tm.set_synchronous_mode(True)
                self.tm.shut_down()
        finally:
            try:
                self._destroy_actors()
            finally:
                self.world.set_weather(self.original_weather)
                self.world.apply_settings(self.original_settings)

    def _weather(self, name):
        carla = self.carla
        if name == "ClearNight":
            return carla.WeatherParameters(cloudiness=10, sun_altitude_angle=-25)
        if name == "HardRainNight":
            return carla.WeatherParameters(cloudiness=90, precipitation=85,
                precipitation_deposits=70, wetness=90, sun_altitude_angle=-25,
                fog_density=20, fog_distance=0, wind_intensity=40)
        if name == "FogNight":
            return carla.WeatherParameters(cloudiness=70, wetness=50,
                sun_altitude_angle=-25, fog_density=70, fog_distance=0)
        value = getattr(carla.WeatherParameters, name, None)
        if value is None:
            raise ValueError(f"Unknown CARLA weather preset {name}")
        return value

    def _spawn_ego_and_route(self, rng):
        import networkx as nx

        e = self.cfg.environment
        spawns = self.map.get_spawn_points()
        if len(spawns) < 2:
            raise RuntimeError("Map needs at least two spawn points")
        blueprint = self.world.get_blueprint_library().find(e.ego_model)
        blueprint.set_attribute("role_name", "world_model_rl")
        fixed = e.start_spawn is not None
        for _ in range(1 if fixed else 100):
            start, goal = (e.start_spawn, e.goal_spawn) if fixed else rng.sample(range(len(spawns)), 2)
            if not (0 <= start < len(spawns) and 0 <= goal < len(spawns)) or start == goal:
                raise ValueError("Invalid start/goal spawn indices")
            try:
                plan = self.planner.trace_route(spawns[start].location, spawns[goal].location)
                route = Route([[wp.transform.location.x, wp.transform.location.y] for wp, _ in plan])
            except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
                if fixed:
                    raise ValueError("Selected spawn pair has no connected driving route")
                continue
            if not fixed and not e.min_route_metres <= route.length <= e.max_route_metres:
                continue
            ego = self.world.try_spawn_actor(blueprint, spawns[start])
            if ego is None:
                continue
            self.ego, self.route, self.route_plan = ego, route, plan
            self.start_spawn, self.goal_spawn = start, goal
            self.actors.append(ego)
            return
        raise RuntimeError("Could not spawn ego on a connected route with the requested lengths")

    def _spawn_traffic(self, rng):
        e, carla = self.cfg.environment, self.carla
        library = self.world.get_blueprint_library()
        vehicles = [bp for bp in library.filter("vehicle.*") if not bp.has_attribute("number_of_wheels")
                    or bp.get_attribute("number_of_wheels").as_int() == 4]
        if e.vehicles and not vehicles:
            raise RuntimeError("No four-wheel vehicle blueprints")
        spawns = list(self.map.get_spawn_points())
        rng.shuffle(spawns)
        # Place traffic near the route first, while retaining seeded variety.
        spawns.sort(key=lambda point: self.route.project([point.location.x, point.location.y])[1])
        self.vehicle_count = 0
        for transform in spawns:
            if self.vehicle_count >= e.vehicles:
                break
            bp = rng.choice(vehicles)
            bp.set_attribute("role_name", "world_model_rl_npc")
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                self.actors.append(actor)
                actor.set_autopilot(True, e.tm_port)
                self.tm.update_vehicle_lights(actor, True)
                self.vehicle_count += 1
        walkers = list(library.filter("walker.pedestrian.*"))
        children = [bp for bp in walkers if (bp.has_attribute("age") and
                    bp.get_attribute("age").as_str().lower() == "child") or (
                    not bp.has_attribute("age") and bp.id.rsplit(".", 1)[-1] in CHILD_MODELS)]
        adults = [bp for bp in walkers if bp not in children]
        if e.pedestrians and (not children or not adults):
            raise RuntimeError("Child and adult pedestrian assets are required")
        child_target = max(1, min(e.pedestrians - 1, round(e.child_fraction * e.pedestrians)))
        self.child_count = self.pedestrian_count = 0
        for attempt in range(e.pedestrians * 30):
            if self.pedestrian_count >= e.pedestrians:
                break
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            if attempt < e.pedestrians * 20 and self.route.project([location.x, location.y])[1] > 35:
                continue
            child = self.child_count < child_target
            bp = rng.choice(children if child else adults)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(bp, carla.Transform(location))
            if actor is None:
                continue
            controller = self.world.try_spawn_actor(library.find("controller.ai.walker"), carla.Transform(), actor)
            if controller is None:
                actor.destroy()
                continue
            self.actors.extend([actor, controller])
            self.controllers.append(controller)
            self.child_count += int(child)
            self.pedestrian_count += 1
        if self.vehicle_count != e.vehicles or self.pedestrian_count != e.pedestrians:
            raise RuntimeError(f"Requested {e.vehicles} vehicles/{e.pedestrians} walkers; spawned "
                               f"{self.vehicle_count}/{self.pedestrian_count}. Reduce counts explicitly.")

    def _build_controls(self):
        """Cache lane-specific traffic light stopping lines, including off-route lines."""
        self.controls = []
        for actor in self.world.get_actors():
            if actor.type_id.startswith("traffic.traffic_light"):
                for waypoint in actor.get_stop_waypoints():
                    p = waypoint.transform.location
                    s, distance, index = self.route.project([p.x, p.y])
                    vector = self.route.vectors[index] / self.route.lengths[index]
                    forward = waypoint.transform.get_forward_vector()
                    f = np.array([forward.x, forward.y])
                    on_route = distance < waypoint.lane_width * 0.6 and np.dot(vector, f) > 0.8
                    self.controls.append({"kind": "light", "actor": actor, "s": s,
                        "centre": np.array([p.x, p.y]), "forward": f,
                        "half_width": waypoint.lane_width / 2, "on_route": on_route})

    def upcoming_controls(self):
        """Return the nearest route-relevant traffic light stopping line."""
        result = {"light": None}
        front_s = self.route.progress + self.ego.bounding_box.extent.x
        for control in sorted(self.controls, key=lambda item: item["s"]):
            if control["on_route"] and -3 <= control["s"] - front_s <= 80 and result[control["kind"]] is None:
                result[control["kind"]] = control
        return result

    @property
    def remaining_time_seconds(self):
        """Return the active driving budget; paused waits do not spend decisions."""
        e = self.cfg.environment
        return max(0, e.max_steps - self.active_steps) * e.decision_dt

    def _policy_observation(self):
        observation = self.provider.observe(self, self.oracle).validate()
        if self.cfg.environment.observation_provider != "ground_truth":
            # Rewards still use independent CARLA truth; the actor's child target uses predictions.
            observation = perception_caution(observation, self.cfg.environment)
        # The episode clock is runtime state, including for future camera providers.
        observation.continuous[15] = self.remaining_time_seconds
        return observation

    def _attach_sensors(self):
        self.event_queue, self.imu_queue = queue.Queue(), queue.Queue()
        self.pending_events = []
        library, carla = self.world.get_blueprint_library(), self.carla
        for kind, blueprint in (("collision", "sensor.other.collision"),
                                ("illegal_lane", "sensor.other.lane_invasion"),
                                ("imu", "sensor.other.imu")):
            bp = library.find(blueprint)
            if kind == "imu":
                bp.set_attribute("sensor_tick", "0")
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
            self.actors.append(sensor)
            self.sensors.append(sensor)
            if kind == "imu":
                sensor.listen(self.imu_queue.put)
            elif kind == "collision":
                sensor.listen(lambda event: self.event_queue.put(("collision", event.frame, 1)))
            else:
                sensor.listen(lambda event: self.event_queue.put(("illegal_lane", event.frame,
                    int(any("Solid" in str(marking.type) for marking in event.crossed_lane_markings)))))

    def _tick(self, settle=True):
        frame = self.world.tick(self.cfg.environment.timeout)
        deadline = time.monotonic() + self.cfg.environment.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"IMU frame barrier timed out at {frame}")
            try:
                sample = self.imu_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise RuntimeError(f"IMU frame barrier timed out at {frame}") from exc
            if sample.frame == frame:
                break
            if sample.frame > frame:
                raise RuntimeError("Another client appears to be ticking this world")
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError("World frame changed outside this collector")
        self.frame, self.sim_time = frame, snapshot.timestamp.elapsed_seconds
        return self._events_at_boundary(frame) if settle else None

    def _events_at_boundary(self, frame):
        time.sleep(self.cfg.environment.sensor_grace_seconds)
        events = self.pending_events
        self.pending_events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        current = {"collision": 0, "illegal_lane": 0}
        for kind, event_frame, value in events:
            if event_frame > frame:
                self.pending_events.append((kind, event_frame, value))
            elif event_frame <= self.committed_frame:
                raise RuntimeError("Late event arrived after its transition was committed; discard this episode")
            else:
                current[kind] = max(current[kind], value)
        if self.world.get_snapshot().frame != frame:
            raise RuntimeError("World frame changed outside this collector")
        return current

    def reset(self, episode_id=0, seed=None, hazard_seed=None):
        """Reset a real episode, optionally staging a collection-only hazard actor."""
        self._destroy_actors()
        e = self.cfg.environment
        seed = self.cfg.training.seed + episode_id if seed is None else seed
        rng = random.Random(seed)
        self.tm.set_random_device_seed(seed)
        self.world.set_pedestrians_seed(seed)
        self.world.set_pedestrians_cross_factor(0.2)
        weather_name = e.weather[episode_id % len(e.weather)]
        weather = self._weather(weather_name)
        self.world.set_weather(weather)
        self._spawn_ego_and_route(rng)
        physics = self.ego.get_physics_control()
        friction_factor = max(0.5, 1 - 0.35 * weather.precipitation / 100 - 0.15 * weather.wetness / 100)
        wheels = physics.wheels
        for wheel in wheels:
            wheel.tire_friction *= friction_factor
        physics.wheels = wheels
        self.ego.apply_physics_control(physics)
        lights = self.carla.VehicleLightState.Position | self.carla.VehicleLightState.LowBeam
        self.ego.set_light_state(self.carla.VehicleLightState(lights))
        if hazard_seed is not None:
            from .collection import StagedHazard
            self.staged_hazard = StagedHazard(self, hazard_seed)
        self._spawn_traffic(rng)
        self.world.tick(e.timeout)
        for controller in self.controllers:
            controller.start()
            target = self.world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(rng.uniform(0.8, 1.8))
        self._build_controls()
        self.previous_action = np.zeros(3, np.float32)
        self._attach_sensors()
        self.committed_frame = -1
        # Let the spawned ego settle without collecting transitions.
        for _ in range(10):
            self.ego.apply_control(self.carla.VehicleControl(brake=1.0))
            self._tick()
        self.ego.apply_control(self.carla.VehicleControl())
        self.committed_frame = self.frame
        self.extractor = OracleExtractor(self)
        self.reward_calculator = RewardCalculator(self.cfg)
        self.steps = self.active_steps = 0
        self.oracle, self.measurement = self.extractor.measure()
        self.provider = load_provider(e.observation_provider, self)
        self.provider.reset()
        observation = self._policy_observation()
        self.stuck_time = self.offroad_time = 0.0
        self.ended = False
        self.expert = None
        self.metadata = {"schema": SCHEMA_VERSION, "config": self.cfg.to_dict(),
            "episode_id": episode_id, "split": split_for(episode_id, self.cfg.training.seed),
            "seed": seed, "town": self.map.name, "weather": weather_name,
            "client_version": self.client_version, "server_version": self.server_version,
            "observation_source": self.provider.source, "start_spawn": self.start_spawn,
            "goal_spawn": self.goal_spawn, "route_points": self.route.points.tolist(),
            "route_length_m": self.route.length, "vehicles": self.vehicle_count,
            "pedestrians": self.pedestrian_count, "children": self.child_count,
            "ego_friction_factor": friction_factor, "collector": "unset"}
        if self.staged_hazard is not None:
            kind = self.staged_hazard.metadata["kind"]
            self.metadata["vehicles"] += int(kind == "vehicle")
            self.metadata["pedestrians"] += int(kind != "vehicle")
            self.metadata["children"] += int(kind == "child")
        from .rewards import REWARD_VERSION
        self.metadata["reward_version"] = REWARD_VERSION
        return observation, {"frame": self.frame, "sim_time": self.sim_time}

    def expert_action(self, rng=None):
        if self.expert is None:
            from agents.navigation.basic_agent import BasicAgent
            self.expert = BasicAgent(self.ego, map_inst=self.map, grp_inst=self.planner,
                                    opt_dict={"dt": self.cfg.environment.decision_dt})
            self.expert.set_global_plan(self.route_plan)
        self.expert.set_target_speed(float(self.oracle.continuous[12]) * 3.6)
        action = self.expert.run_step()
        requested = np.array([action.steer, action.throttle, action.brake], np.float32)
        if rng is not None:
            requested += rng.normal(0, [0.06, 0.12, 0.05], 3)
            if rng.random() < 0.03:
                requested = np.array([rng.uniform(-0.8, 0.8), rng.random(), rng.random()], np.float32)
        return np.clip(requested, [-1, 0, 0], [1, 1, 1]).astype(np.float32)

    def finish_episode(self):
        """Catch detectable delayed final events before an episode is committed to disk."""
        time.sleep(self.cfg.environment.sensor_grace_seconds)
        if self.pending_events:
            raise RuntimeError("Future-frame events remain at episode end; discard this episode")
        while True:
            try:
                _, frame, value = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if value and frame <= self.committed_frame:
                raise RuntimeError("Late final event detected; discard this episode")
            if frame > self.committed_frame:
                raise RuntimeError("World advanced outside this collector at episode end")

    def _traffic_events(self, before_front, after_front):
        """Count red-light front-bumper crossings at each physics tick."""
        incoming = {"red_light": 0}
        for control in self.controls:
            if crosses_line(before_front, after_front, control["centre"],
                            control["forward"], control["half_width"]):
                if control["actor"].get_state() == self.carla.TrafficLightState.Red:
                    incoming["red_light"] += 1
        return incoming

    def step(self, requested):
        """Apply one decision and spend the deadline only outside legitimate light waits."""
        if self.ended:
            raise RuntimeError("Call reset after the episode ends")
        e, carla = self.cfg.environment, self.carla
        previous = self.previous_action.copy()
        applied = apply_action(requested, previous, e)
        self.previous_action = applied
        self.ego.apply_control(carla.VehicleControl(steer=float(applied[0]), throttle=float(applied[1]),
                                                   brake=float(applied[2])))
        before = self.measurement
        front = before["front"]
        control_events = {"red_light": 0}
        for _ in range(e.action_repeat):
            if getattr(self, "staged_hazard", None) is not None:
                self.staged_hazard.tick()
            self._tick(settle=False)
            if self.controls:
                # Keep crossing checks at physics rate, including light phase changes.
                bumper = self.ego.get_transform().transform(carla.Location(x=self.ego.bounding_box.extent.x))
                next_front = np.array([bumper.x, bumper.y])
                found = self._traffic_events(front, next_front)
                for name, value in found.items():
                    control_events[name] += value
                front = next_front
        incoming = self._events_at_boundary(self.frame)
        incoming.update(control_events)
        # Actor queries, visibility rays and observation assembly occur once per decision.
        self.oracle, self.measurement = self.extractor.measure()
        after = self.measurement
        distance_metres = float(np.linalg.norm(after["position"] - before["position"]))
        incoming["success"] = float(after["success"] and not incoming["collision"])
        self.offroad_time = self.offroad_time + e.decision_dt if after["offroad"] else 0.0
        if after["speed"] >= 0.2 or after["progress_distance"] > 0.01 or after["blocked"]:
            self.stuck_time = 0.0
        else:
            self.stuck_time += e.decision_dt
        self.steps += 1
        self.active_steps += int(not after["blocked"])
        self.oracle.continuous[15] = self.remaining_time_seconds
        task_ended = bool(incoming["collision"] or incoming["success"] or self.offroad_time >= e.offroad_seconds)
        incoming["time_limit"] = float(not task_ended and self.active_steps >= e.max_steps)
        terminated = bool(task_ended or incoming["time_limit"])
        truncated = bool(not terminated and self.stuck_time >= e.stuck_seconds)
        reason = ("collision" if incoming["collision"] else "success" if incoming["success"] else
                  "offroad" if task_ended else "time_limit" if incoming["time_limit"] else
                  "stuck" if truncated else "running")
        components, events = self.reward_calculator.step(before, after, incoming,
            e.decision_dt, applied, previous)
        self.ended = terminated or truncated
        self.committed_frame = self.frame
        observation = self._policy_observation()
        return observation, float(sum(components.values())), terminated, truncated, {
            "frame": self.frame, "sim_time": self.sim_time, "applied_action": applied,
            "events": events, "components": components, "terminal_reason": reason,
            "route_completion": min(1.0, self.route.high_water / self.route.length),
            "distance_metres": distance_metres, "acceleration_mps2": self.reward_calculator.comfort.acceleration,
            "jerk_mps3": self.reward_calculator.comfort.jerk,
            "steering_rate_per_s": float(abs(applied[0] - previous[0]) / e.decision_dt),
            "lane_offset": self.measurement["lane_offset"], "heading_error": self.measurement["heading_error"]}

"""Dedicated synchronous CARLA environment, with no physical vehicle API."""

import importlib
import logging
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .driving import (MIN_EPISODE_RETURN, WAYPOINT_SPACING_M, DrivingConfig, DrivingState, RewardConfig,
                      SafetySupervisor, action_to_controls, compute_reward, signal_wait)
from .inactivity import InactivityWatchdog
from .navigation import Route, crosswalk_polygons
from .observations import pack_observation
from .sensors import CameraConfig, RadarConfig, SensorRig, associate_depth
from .weather import scenario_weather


class _UnsupportedRoute(RuntimeError):
    pass


def ego_coordinates(transform, location):
    matrix = np.asarray(transform.get_inverse_matrix())
    return (matrix @ [location.x, location.y, location.z, 1.0])[:3]


class CarlaEnv:
    """Owns spawned actors and restores settings; requires a dedicated server."""

    def __init__(self, cfg, scenario=None, *, scenarios=None, start_episode=0):
        import carla

        self.cfg, self.carla, self.scenario = cfg.validate(), carla, scenario or {}
        if scenarios is not None and not scenarios:
            raise ValueError("At least one collection scenario is required")
        self.scenarios, self.scenario_episode = scenarios, start_episode
        if scenarios is not None:
            self.scenario = scenarios[start_episode % len(scenarios)]
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(cfg.timeout)
        if self.client.get_client_version() != self.client.get_server_version():
            raise RuntimeError("CARLA Python client and server versions must match")
        self.world = self.client.get_world()
        if self.world.get_map().name.rsplit("/", 1)[-1] != cfg.town:
            raise RuntimeError(f"Expected {cfg.town}; launch that map on a dedicated server first")
        self.map = self.world.get_map()
        self.settings, self.weather = self.world.get_settings(), self.world.get_weather()
        self.actors, self.rig, self.provider = [], None, None
        self.traffic_manager = None
        self.walker_controllers = []
        self.owns_settings = False
        self.closed = False
        self.rng = np.random.default_rng(cfg.seed)
        self.supervisor = SafetySupervisor(DrivingConfig(**cfg.driving))
        self.reward_config = RewardConfig(**cfg.reward)
        self.inactivity = InactivityWatchdog(cfg.inactivity_timeout_s, cfg.inactivity_min_progress_m,
                                            cfg.blocked_timeout_s, cfg.inactivity_grace_s)
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                raise RuntimeError("Server already synchronous: refuse to take over another client session")
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = cfg.dt
            settings.no_rendering_mode = False
            settings.substepping = True
            settings.max_substep_delta_time = 0.01
            settings.max_substeps = 10
            self.world.apply_settings(settings)
            self.owns_settings = True
            self.world.set_weather(scenario_weather(carla, self.scenario, cfg.weather))
            if cfg.perception != "ground_truth":
                module, name = cfg.provider.split(":")
                self.provider = getattr(importlib.import_module(module), name)()
        except BaseException:
            self.close()
            raise

    def _destroy(self):
        if self.rig:
            self.rig.close()
            self.rig = None
        for controller in getattr(self, "walker_controllers", []):
            try:
                controller.stop()
            except RuntimeError:
                continue
        self.walker_controllers = []
        manager = getattr(self, "traffic_manager", None)
        for actor, spec in getattr(self, "moving", []):
            if manager is not None and actor.type_id.startswith("vehicle.") and spec.get("speed_mps", 0) > 0:
                try:
                    actor.set_autopilot(False, manager.get_port())
                except RuntimeError:
                    continue
        self.moving = []
        for actor in reversed(self.actors):
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                continue
        self.actors.clear()

    def reset(self):
        try:
            if self.scenarios is not None:
                self.scenario = self.scenarios[self.scenario_episode % len(self.scenarios)]
                self.route_visit = self.scenario_episode // len(self.scenarios)
                self.episode_seed = int(np.random.SeedSequence([
                    self.cfg.seed, self.scenario.get("seed", self.cfg.seed), self.scenario_episode
                ]).generate_state(1)[0])
                self.rng = np.random.default_rng(self.episode_seed)
                self.world.set_weather(scenario_weather(self.carla, self.scenario, self.cfg.weather))
                self.scenario_episode += 1
            return self._reset()
        except BaseException:
            self._destroy()
            raise

    def _reset(self):
        self._destroy()
        if self.provider and hasattr(self.provider, "reset"):
            self.provider.reset()
        self.supervisor.reset()
        self.steps, self.timed_steps, self.previous_action, self.previous_acceleration = 0, 0, (0., 0.), 0.
        self.episode_return = 0.0
        self.crossing_entries = set()
        self.demo_pending = list(self.scenario.get("encounters", []))
        self.demo_walkers = []
        carla = self.carla
        spawns = self.map.get_spawn_points()
        start = self.scenario.get("route_start", self.cfg.route_start)
        end = self.scenario.get("route_end", self.cfg.route_end)
        if not 0 <= start < len(spawns) or not 0 <= end < len(spawns) or start == end:
            raise ValueError("Route spawn indices must be distinct and available on map")
        start, end, waypoints, self.route, self.crossings = self._select_route(spawns, start, end)
        self.route_start, self.route_end = start, end
        ego_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "dreamerv3_attention")
        self.ego = self.world.spawn_actor(ego_bp, spawns[start])
        self.actors.append(self.ego)
        # Actor transforms are cached from snapshots; a new actor reads as the
        # world origin until the first tick, misplacing ego-relative hazards.
        self.world.tick()
        self._cache_traffic_stops([wp for wp, _ in waypoints])
        if (self.scenario.get("requires_traffic_light") and not any(
                30. <= position <= self.route.length - 10. for _, position in self.traffic_stops)):
            raise _UnsupportedRoute("Unsupported scenario: selected route has no usable traffic-light stop")
        self.capabilities = {"map": self.map.name, "route_crossings": len(self.crossings),
                             "crosswalk_map_annotations": bool(self.map.get_crosswalks()),
                             "scenario": self.scenario.get("name", "basic")}
        self._spawn_hazards()
        # Walker controllers need their parent's first snapshot before start().
        self.world.tick()
        self._start_npcs()
        self.rig = SensorRig(self.world, self.ego,
                             CameraConfig(**{"sensor_tick": self.cfg.dt, **self.cfg.camera}),
                             RadarConfig(**{"sensor_tick": self.cfg.dt, **self.cfg.radar}))
        # Tick before reading, so all policy information refers to one snapshot.
        for _ in range(3):
            frame = self.world.tick()
        self.state, observation = self._observe(frame)
        self.oracle_before = replace(self.oracle_state)
        self.reward_progress_highwater = min(self.route.length, max(0.0, self.oracle_state.route_progress_m))
        self.inactivity.reset(self.oracle_state.route_progress_m)
        return observation

    def _select_route(self, spawns, start, end):
        """Cycle a seeded pool of supported routes during train collection."""
        agents = str(Path(__file__).resolve().parents[1] / "PythonAPI" / "carla")
        if agents not in sys.path:
            sys.path.insert(0, agents)
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        if not hasattr(self, "route_planner"):
            self.route_planner = GlobalRoutePlanner(self.map, 2.)
        short_route = "route_length_m" in self.scenario
        randomize = self.scenario.get("randomize_routes", self.cfg.randomize_routes)
        if not isinstance(randomize, bool):
            raise ValueError("scenario randomize_routes must be a boolean")
        varied = (getattr(self, "scenarios", None) is not None and
                  self.cfg.split == "train" and randomize)
        if self.scenario.get("fixed_route") or (
                not short_route and not varied and not self.scenario.get("requires_traffic_light")):
            waypoints = self.route_planner.trace_route(spawns[start].location, spawns[end].location)
            route, crossings = self._route_geometry(waypoints)
            if route.length < float(self.scenario.get("minimum_route_length_m", 0.)):
                raise _UnsupportedRoute("Unsupported scenario: selected route is shorter than the demo minimum")
            return start, end, waypoints, route, crossings

        import networkx as nx
        from .evaluation import scenario_records

        if not hasattr(self, "route_pools"):
            self.route_pools = {}
        name = self.scenario.get("name", "basic")
        if name not in self.route_pools:
            reserved = set()
            for record in scenario_records(self.cfg):
                if record["split"] != "train":
                    a, b = record["route_start"], record["route_end"]
                    reserved.update(((a, b), (b, a)))
            rng = np.random.default_rng(np.random.SeedSequence([
                self.cfg.seed, self.scenario.get("seed", self.cfg.seed)]))
            count = len(spawns)
            # Nearly straight signal routes are rare among arbitrary endpoints.
            candidate_limit = 2048 if self.scenario.get("requires_traffic_light") else 128
            candidates = rng.permutation(count * (count - 1))[:candidate_limit]
            pool = []
            starts = set()
            pool_limit = 16 if varied else 1
            for candidate in candidates:
                a, b = divmod(int(candidate), count - 1)
                b += b >= a
                if (a, b) in reserved:
                    continue
                # Different destinations can crop to the same road prefix.
                if short_route and a in starts:
                    continue
                try:
                    waypoints = self.route_planner.trace_route(spawns[a].location, spawns[b].location)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                try:
                    waypoints = self._crop_scenario_route(waypoints)
                    route, crossings = self._route_geometry(waypoints)
                except (ValueError, _UnsupportedRoute):
                    continue
                minimum_length = float(self.scenario["route_length_m"]) if short_route else 30.
                if route.length < minimum_length:
                    continue
                if route.length < float(self.scenario.get("minimum_route_length_m", 0.)):
                    continue
                pool.append((a, b, waypoints, route, crossings))
                starts.add(a)
                if len(pool) == pool_limit:
                    break
            if not pool:
                reason = "no valid route" if short_route else "no valid varied route"
                raise RuntimeError(f"Unsupported scenario {name}: {reason} in {candidate_limit} candidates")
            if len(pool) == 1 and varied:
                logging.getLogger(__name__).warning("Scenario %s has only one valid route variant", name)
            self.route_pools[name] = pool
        selected = (self.route_pools[name][self.route_visit % len(self.route_pools[name])] if varied else
                    self.route_pools[name][0])
        selected[3].index = 0
        return selected

    def _crop_scenario_route(self, waypoints):
        """Limit opt-in pilot routes before validating geometry and caching signals."""
        if "route_length_m" not in self.scenario:
            return waypoints
        raw_length = self.scenario["route_length_m"]
        if isinstance(raw_length, bool) or not isinstance(raw_length, (int, float)):
            raise ValueError("scenario route_length_m must be finite and positive")
        length = float(raw_length)
        if not math.isfinite(length) or length <= 0:
            raise ValueError("scenario route_length_m must be finite and positive")
        if len(waypoints) < 2:
            raise _UnsupportedRoute("Unsupported scenario: route is too short")
        points = np.asarray([(wp.transform.location.x, wp.transform.location.y) for wp, _ in waypoints])
        cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if cumulative[-1] < length:
            raise _UnsupportedRoute("Unsupported scenario: route is shorter than requested length")
        stop = int(np.searchsorted(cumulative, length, side="left"))
        cropped = waypoints[:min(len(waypoints), stop + 1)]
        if len(cropped) < 2:
            raise _UnsupportedRoute("Unsupported scenario: route is too short")
        if any(getattr(wp, "is_junction", False) for wp, _ in cropped):
            raise _UnsupportedRoute("Unsupported scenario: cropped route contains a junction")
        return cropped

    def _route_geometry(self, waypoints):
        if len(waypoints) < 2:
            raise ValueError("Route needs at least two waypoints")
        route = Route([(wp.transform.location.x, wp.transform.location.y) for wp, _ in waypoints],
                      lane_widths=[getattr(wp, "lane_width", float("nan")) for wp, _ in waypoints],
                      junctions=[getattr(wp, "is_junction", False) for wp, _ in waypoints])
        crossings = route.crossings(crosswalk_polygons(self.map.get_crosswalks()))
        if self.scenario.get("requires_crosswalk") and not crossings:
            raise _UnsupportedRoute("Unsupported scenario: selected map/route has no usable crosswalk annotations")
        if self.scenario.get("requires_speed_limit_transition"):
            limits = set()
            for landmark in self.map.get_all_landmarks_of_type("274"):
                position, offset, _ = route.project(
                    (landmark.transform.location.x, landmark.transform.location.y), local=False)
                if 0 < position < route.length and abs(offset) < 5:
                    limits.add(landmark.value)
            if len(limits) < 2:
                raise _UnsupportedRoute("Unsupported scenario: selected route has no annotated speed-limit transition")
        category = self.scenario.get("category", "straight")
        headings = np.unwrap(np.arctan2(route.delta[:, 1], route.delta[:, 0]))
        if ("max_heading_change_rad" in self.scenario and
                np.ptp(headings) > self.scenario["max_heading_change_rad"]):
            raise _UnsupportedRoute("Unsupported scenario: route exceeds maximum heading change")
        if self.scenario.get("category") == "straight" and np.ptp(headings) > 0.2:
            raise _UnsupportedRoute("Unsupported scenario: selected route is not straight")
        if category == "curved" and np.ptp(headings) < 0.2:
            raise _UnsupportedRoute("Unsupported scenario: selected route has no curve")
        if category == "intersection" and not any(wp.is_junction for wp, _ in waypoints):
            raise _UnsupportedRoute("Unsupported scenario: selected route has no intersection")
        if self.scenario.get("requires_traffic_light"):
            if not hasattr(self, "route_light_waypoints"):
                self.route_light_waypoints = [
                    (light, light.get_stop_waypoints()) for light in self.world.get_actors()
                    if light.type_id.startswith("traffic.traffic_light")]
            stops = self._route_traffic_stops([wp for wp, _ in waypoints], route, self.route_light_waypoints)
            if not any(30. <= position <= route.length - 10. for _, position in stops):
                raise _UnsupportedRoute("Unsupported scenario: route has no usable traffic-light stop")
        return route, crossings

    def _cache_traffic_stops(self, waypoints):
        """Match stop geometry to each occurrence of its lane on this route."""
        self.traffic_stops = self._route_traffic_stops(waypoints, self.route)

    def _route_traffic_stops(self, waypoints, route, light_waypoints=None):
        """Share signal matching between candidate selection and live detection."""
        points = np.asarray([(wp.transform.location.x, wp.transform.location.y) for wp in waypoints])
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.05]
        waypoints = [wp for wp, retain in zip(waypoints, keep) if retain]
        stops = []
        if light_waypoints is None:
            light_waypoints = [(light, light.get_stop_waypoints()) for light in self.world.get_actors()
                               if light.type_id.startswith("traffic.traffic_light")]
        for light, stop_waypoints in light_waypoints:
            positions = []
            for stop in stop_waypoints:
                point = np.asarray([stop.transform.location.x, stop.transform.location.y])
                forward = stop.transform.get_forward_vector()
                direction = np.asarray([forward.x, forward.y])
                for index, delta in enumerate(route.delta):
                    if not any((wp.road_id, wp.section_id, wp.lane_id) ==
                               (stop.road_id, stop.section_id, stop.lane_id)
                               for wp in waypoints[index:index+2]):
                        continue
                    length = route.lengths[index]
                    if np.dot(delta / length, direction) < 0.5:
                        continue
                    fraction = float(np.dot(point-route.points[index], delta) / length**2)
                    # Reject collinear points beyond either route endpoint too.
                    if not 0.0 <= fraction <= 1.0:
                        continue
                    projected = route.points[index] + fraction*delta
                    if np.linalg.norm(point-projected) > min(1.0, stop.lane_width/4.):
                        continue
                    position = float(route.cumulative[index] + fraction*length)
                    if not any(abs(position-other) < 0.1 for other in positions):
                        positions.append(position)
            stops.extend((light, position) for position in positions)
        return sorted(stops, key=lambda item: item[1])

    def _observe_traffic_light(self, state, bumper):
        config = self.supervisor.config
        restart_metadata = (config.restart_assist_enabled and
                            self.cfg.perception == "ground_truth")
        speed = max(0., state.speed_mps)
        predicted = speed + config.max_acceleration_mps2*self.cfg.dt
        lookahead = max(30., config.crossing_buffer_m +
                        predicted**2/(2.*config.red_light_deceleration_mps2) + speed*self.cfg.dt + 10.)
        restart_candidates = []
        tracked_id = self.supervisor._restart_light_id
        stop_phase_seen = False
        for light, position in self.traffic_stops:
            distance = position-state.route_progress_m-bumper
            if restart_metadata and -3. * config.restart_stop_distance_m <= distance <= lookahead:
                phase = light.get_state()
                phase_name = ("Red" if phase == self.carla.TrafficLightState.Red else
                              "Yellow" if phase == self.carla.TrafficLightState.Yellow else
                              "Green" if phase == self.carla.TrafficLightState.Green else None)
                restart_candidates.append((light, distance, phase_name))
            if not 0. <= distance <= lookahead:
                continue
            phase = light.get_state()
            if not stop_phase_seen:
                if phase == self.carla.TrafficLightState.Red:
                    state.red_light, state.red_light_distance_m = True, distance
                    stop_phase_seen = True
                elif phase == self.carla.TrafficLightState.Yellow:
                    state.yellow_light, state.yellow_light_distance_m = True, distance
                    state.yellow_light_confidence = 1.0
                    stop_phase_seen = True
            if stop_phase_seen and not restart_metadata:
                break
        if restart_metadata and restart_candidates:
            tracked = next((item for item in restart_candidates if str(item[0].id) == tracked_id), None)
            forward = [item for item in restart_candidates if item[1] >= 0.]
            selected = tracked or (min(forward, key=lambda item: item[1]) if forward else None)
            if selected:
                light, distance, phase_name = selected
                state.traffic_light_id = str(light.id)
                state.traffic_light_phase = phase_name
                state.traffic_light_distance_m = distance

    def _demo_encounters(self, progress):
        """Start each configured encounter once as the ego advances."""
        while self.demo_pending and progress >= self.demo_pending[0]["trigger_m"]:
            event = self.demo_pending.pop(0)
            self._spawn_demo_actor(event, progress)
        remaining = []
        for actor, origin, direction, distance in self.demo_walkers:
            if not actor.is_alive:
                continue
            location = actor.get_location()
            moved = ((location.x - origin.x) * direction.x + (location.y - origin.y) * direction.y)
            if moved >= distance:
                actor.apply_control(self.carla.WalkerControl(direction=direction, speed=0.))
            else:
                remaining.append((actor, origin, direction, distance))
        self.demo_walkers = remaining

    def _spawn_demo_actor(self, event, progress):
        kind = event["kind"]
        library = self.world.get_blueprint_library()
        child_ids = {"walker.pedestrian.%04d" % i for i in (9, 10, 11, 12, 13, 14, 48, 49)}
        candidates = list(library.filter("vehicle.*" if kind == "vehicle" else "walker.pedestrian.*"))
        if kind != "vehicle":
            child = kind == "child"
            candidates = [bp for bp in candidates if (bp.id in child_ids or
                          (bp.has_attribute("age") and str(bp.get_attribute("age")) == "child")) == child]
        candidates.sort(key=lambda bp: bp.id)
        if not candidates:
            raise RuntimeError(f"Demonstration cannot spawn {kind}: no matching blueprint")
        bp = candidates[int(self.rng.integers(len(candidates)))]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        for ahead in (30., 28., 26., 24., 22.):
            s = min(progress + ahead, self.route.length - 5.)
            i = min(int(np.searchsorted(self.route.cumulative, s, side="right") - 1), len(self.route.delta) - 1)
            if self.route.junctions is not None and self.route.junctions[i]:
                continue
            fraction = (s - self.route.cumulative[i]) / self.route.lengths[i]
            point = self.route.points[i] + fraction * self.route.delta[i]
            road = self.map.get_waypoint(self.carla.Location(x=float(point[0]), y=float(point[1]),
                                                             z=self.ego.get_location().z),
                                         lane_type=self.carla.LaneType.Driving)
            if road is None or road.is_junction:
                continue
            if kind == "vehicle":
                transform = self.carla.Transform(road.transform.location + self.carla.Location(z=.5),
                                                 road.transform.rotation)
            else:
                unit = self.route.delta[i] / self.route.lengths[i]
                side = 1. if kind == "adult" else -1.
                nx, ny = -unit[1] * side, unit[0] * side
                curb = self.carla.Location(x=float(point[0] + nx * 4.5), y=float(point[1] + ny * 4.5),
                                            z=road.transform.location.z)
                sidewalk = self.map.get_waypoint(curb, lane_type=self.carla.LaneType.Sidewalk)
                if sidewalk is None or sidewalk.transform.location.distance(curb) > 3.:
                    continue
                origin = sidewalk.transform.location + self.carla.Location(z=.5)
                if (origin.x - point[0]) * nx + (origin.y - point[1]) * ny < 2.:
                    continue
                transform = self.carla.Transform(origin)
                direction = self.carla.Vector3D(x=float(-nx), y=float(-ny), z=0.)
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                continue
            self.actors.append(actor)
            if kind == "vehicle":
                if self.traffic_manager is None:
                    self.traffic_manager = self.client.get_trafficmanager()
                    self.traffic_manager.set_synchronous_mode(True)
                    self.traffic_manager.set_random_device_seed(self.cfg.seed)
                actor.set_autopilot(True, self.traffic_manager.get_port())
                self.moving.append((actor, {"speed_mps": 3.}))
            else:
                actor.apply_control(self.carla.WalkerControl(direction=direction, speed=1.))
                self.demo_walkers.append((actor, origin, direction, 9.))
            return
        raise RuntimeError(f"Demonstration cannot spawn {kind} on the route ahead")

    def _spawn_hazards(self):
        self.moving = []
        self.walker_controllers = []
        self.required_pedestrians = []
        background = self.scenario.get("background_traffic", True)
        if not isinstance(background, bool):
            raise ValueError("scenario background_traffic must be a boolean")
        library = self.world.get_blueprint_library()
        self.world.set_pedestrians_seed(getattr(self, "episode_seed", self.cfg.seed))
        # Filter 0 excludes ROAD navmesh areas but permits CROSSWALK areas.
        self.world.set_pedestrians_cross_factor(0.0)
        sidewalks, visible_sidewalks = [], []
        ego_transform = self.ego.get_transform()
        authored = list(self.scenario.get("actors", []))
        needs_sidewalks = background or any(
            spec.get("kind") == "pedestrian" and
            not (spec.get("at_crossing") and not spec.get("speed_mps", 0))
            for spec in authored)
        needs_visible_sidewalks = self.scenario.get("requires_pedestrians", False) and any(
            spec.get("kind") == "pedestrian" and
            not (spec.get("at_crossing") and not spec.get("speed_mps", 0))
            for spec in authored)
        if needs_sidewalks:
            # Navigation draws cover the whole map. Required walkers need their
            # own candidate pool, even after the optional nearby pool is full.
            for _ in range(16384 if needs_visible_sidewalks else 128):
                point = self.world.get_random_location_from_navigation()
                if point is not None and point.distance(ego_transform.location) <= 60.:
                    if len(sidewalks) < 24:
                        sidewalks.append(point)
                    if needs_visible_sidewalks:
                        x, y, _ = ego_coordinates(ego_transform, point)
                        if 3. <= x <= 14. and 2. <= abs(y) < 7.:
                            visible_sidewalks.append(point)
                    if len(sidewalks) == 24 and (not needs_visible_sidewalks or len(visible_sidewalks) >= 8):
                        break
        if needs_sidewalks and not sidewalks:
            logging.getLogger(__name__).warning("No nearby sidewalk navigation: skipping roaming pedestrians")
        defaults = [
            {"kind": "vehicle", "x": 35, "y": 0, "speed_mps": 2},
            {"kind": "pedestrian", "x": 25, "y": 6, "speed_mps": 1.2},
            {"kind": "pedestrian", "child": True, "x": 30, "y": -6, "speed_mps": 1.0},
        ]
        required = [(spec, "authored") for spec in authored]
        for default in defaults if background else []:
            if not any(spec.get("kind", "vehicle") == default["kind"] and
                       (default["kind"] != "pedestrian" or
                        bool(spec.get("child")) == bool(default.get("child"))) for spec in authored):
                required.append((default, "default"))

        def random_placement(kind):
            x = float(self.rng.uniform(8., 55.))
            if kind == "pedestrian":
                y = float(self.rng.choice((-1., 1.)) * self.rng.uniform(4., 10.))
            else:
                y = float(self.rng.choice((-3.5, 0., 3.5)))
            return x, y

        kinds = (["vehicle"] * int(self.rng.integers(1, 3)) +
                 ["pedestrian"] * int(self.rng.integers(5, 11)))
        if not background:
            kinds = []
        self.rng.shuffle(kinds)
        extra_count = len(kinds)
        extras = []
        for kind in kinds:
            x, y = random_placement(kind)
            extras.append(({"kind": kind, "x": x, "y": y,
                            "speed_mps": float(self.rng.uniform(.9, 1.4) if kind == "pedestrian"
                                               else self.rng.uniform(1.5, 4.))},
                           "extra"))
        spawned_extra = 0
        for spec, origin in required + extras:
            kind = spec.get("kind", "vehicle")
            required_pedestrian = (self.scenario.get("requires_pedestrians", False) and
                                   origin == "authored" and kind == "pedestrian")
            required_vehicle = (self.scenario.get("requires_vehicles", False) and
                                origin == "authored" and kind == "vehicle")
            candidates = list(library.filter("walker.pedestrian.*" if kind == "pedestrian" else "vehicle.*"))
            if kind == "pedestrian":
                child_ids = {"walker.pedestrian.%04d" % i for i in (9, 10, 11, 12, 13, 14, 48, 49)}
                candidates = [bp for bp in candidates if
                              (bp.id in child_ids or (bp.has_attribute("age") and
                               str(bp.get_attribute("age")) == "child")) == bool(spec.get("child"))]
            candidates = sorted(candidates, key=lambda bp: bp.id)
            if not candidates:
                reason = ("Unsupported scenario: no authored child blueprint" if spec.get("child") else
                          "No actor blueprint")
                raise RuntimeError(reason)
            bp = candidates[int(self.rng.integers(len(candidates)))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            local = self.carla.Location(x=float(spec.get("x", 20)), y=float(spec.get("y", 0)), z=1.)
            location = ego_transform.transform(local)
            occupied_crossing = kind == "pedestrian" and spec.get("at_crossing") and not spec.get("speed_mps", 0)
            if occupied_crossing:
                if not self.crossings:
                    if required_pedestrian:
                        raise RuntimeError("Unsupported scenario: required pedestrian has no marked crossing")
                    logging.getLogger(__name__).warning("Skipping occupied crossing actor: no marked route crossing")
                    continue
                _, entry, exit = self.crossings[0]
                centre_s = (entry+exit)/2.
                index = min(int(np.searchsorted(self.route.cumulative, centre_s)-1), len(self.route.delta)-1)
                fraction = (centre_s-self.route.cumulative[index])/self.route.lengths[index]
                point = self.route.points[index]+fraction*self.route.delta[index]
                road = self.map.get_waypoint(self.carla.Location(x=float(point[0]), y=float(point[1]), z=location.z))
                location = self.carla.Location(x=float(point[0]), y=float(point[1]), z=road.transform.location.z+0.5)
            actor = None
            attempts = 1 if origin == "authored" else 12
            if kind == "pedestrian" and not occupied_crossing and not sidewalks:
                if required_pedestrian:
                    raise RuntimeError("Unsupported scenario: required pedestrian has no sidewalk navigation")
                continue
            nearby_sidewalks = sorted(sidewalks, key=lambda point: point.distance(location))
            if required_pedestrian and not occupied_crossing:
                nearby_sidewalks = sorted(visible_sidewalks, key=lambda point: point.distance(location))
                if not nearby_sidewalks:
                    raise RuntimeError("Unsupported scenario: required pedestrian has no nearby visible sidewalk")
                attempts = len(nearby_sidewalks)
            for attempt in range(attempts):
                if attempt:
                    if not occupied_crossing and not required_pedestrian:
                        x, y = random_placement(kind)
                        local = self.carla.Location(x=x, y=y, z=1.)
                        location = ego_transform.transform(local)
                if kind == "vehicle":
                    road = self.map.get_waypoint(location, lane_type=self.carla.LaneType.Driving)
                    if road is None or road.is_junction or road.transform.location.distance(location) > 5.:
                        continue
                    transform = self.carla.Transform(road.transform.location + self.carla.Location(z=.5),
                                                     road.transform.rotation)
                elif occupied_crossing:
                    # Retries must never leave the marked crossing.
                    transform = self.carla.Transform(location)
                else:
                    point = (nearby_sidewalks[attempt] if origin == "authored" else
                             sidewalks[int(self.rng.integers(len(sidewalks)))])
                    transform = self.carla.Transform(point + self.carla.Location(z=.5))
                actor = self.world.try_spawn_actor(bp, transform)
                if actor is not None:
                    break
            if actor is None:
                if required_vehicle:
                    raise RuntimeError("Unsupported scenario: required vehicle could not spawn")
                if required_pedestrian:
                    raise RuntimeError("Unsupported scenario: required pedestrian could not spawn")
                if origin != "extra":
                    logging.getLogger(__name__).warning(
                        "Skipping %s %s (%s) in scenario %s after %s failed spawn attempt(s)",
                        origin, kind, bp.id, self.scenario.get("name", "basic"), attempts)
                continue
            self.actors.append(actor)
            self.moving.append((actor, spec))
            if required_pedestrian:
                self.required_pedestrians.append(actor)
            spawned_extra += int(origin == "extra")
        self.actor_spawn_report = {"requested_extra": extra_count, "spawned_extra": spawned_extra}
        if spawned_extra < extra_count:
            logging.getLogger(__name__).warning("Spawned %s of %s optional random actors after retries",
                                               spawned_extra, extra_count)

    def _start_npcs(self):
        """Use CARLA navigation instead of forcing straight-line motion."""
        walkers = []
        if self.traffic_manager is not None:
            self.traffic_manager.set_random_device_seed(getattr(self, "episode_seed", self.cfg.seed))
        for actor, spec in self.moving:
            speed = float(spec.get("speed_mps", 0))
            if actor.type_id.startswith("vehicle."):
                if speed <= 0:
                    actor.apply_control(self.carla.VehicleControl(brake=1., hand_brake=True))
                    continue
                if self.traffic_manager is None:
                    self.traffic_manager = self.client.get_trafficmanager()
                    self.traffic_manager.set_synchronous_mode(True)
                    self.traffic_manager.set_random_device_seed(getattr(self, "episode_seed", self.cfg.seed))
                manager = self.traffic_manager
                manager.ignore_lights_percentage(actor, 0.)
                manager.ignore_signs_percentage(actor, 0.)
                manager.ignore_walkers_percentage(actor, 0.)
                manager.ignore_vehicles_percentage(actor, 0.)
                manager.distance_to_leading_vehicle(actor, 5.)
                manager.auto_lane_change(actor, False)
                manager.random_left_lanechange_percentage(actor, 0.)
                manager.random_right_lanechange_percentage(actor, 0.)
                manager.update_vehicle_lights(actor, True)
                actor.set_autopilot(True, manager.get_port())
            elif speed > 0:
                required = any(actor is item for item in getattr(self, "required_pedestrians", []))
                target = self.world.get_random_location_from_navigation()
                if target is None:
                    if required:
                        raise RuntimeError("Unsupported scenario: required pedestrian has no navigation target")
                    continue
                bp = self.world.get_blueprint_library().find("controller.ai.walker")
                controller = self.world.try_spawn_actor(bp, self.carla.Transform(), attach_to=actor)
                if controller is None:
                    if required:
                        raise RuntimeError("Unsupported scenario: required pedestrian controller unavailable")
                    logging.getLogger(__name__).warning("Walker controller unavailable: pedestrian stays on sidewalk")
                    continue
                self.actors.append(controller)
                self.walker_controllers.append(controller)
                walkers.append((actor, controller, speed, target))
        self._update_npcs()
        if walkers:
            self.world.tick()
        for actor, controller, speed, target in walkers:
            controller.start()
            # Native start disables collisions; preserve collision telemetry.
            actor.set_collisions(True)
            controller.set_max_speed(min(speed, 1.4))
            controller.go_to_location(target)

    def _update_npcs(self):
        alive = []
        for actor, spec in self.moving:
            try:
                # TrafficManager can destroy stuck NPCs between ticks.
                if not actor.is_alive:
                    continue
                if actor.type_id.startswith("vehicle.") and spec.get("speed_mps", 0) > 0:
                    limit = float(actor.get_speed_limit())
                    target = min(float(spec["speed_mps"])*3.6, .8*limit) if math.isfinite(limit) else 0.
                    self.traffic_manager.set_desired_speed(actor, max(0., target))
            except RuntimeError as exc:
                # is_alive itself can raise after the actor proxy is cleared.
                if "trying to operate on a destroyed actor" not in str(exc):
                    raise
                logging.getLogger(__name__).warning("Dropping destroyed NPC %s (%s)", actor.id, actor.type_id)
                continue
            alive.append((actor, spec))
        self.moving = alive

    def _actor_states(self):
        """Read NPC state before using it in hazards or the red-light queue."""
        for actor in self.world.get_actors():
            if actor.id == self.ego.id or not actor.type_id.startswith(("vehicle.", "walker.pedestrian.")):
                continue
            try:
                if not actor.is_alive:
                    continue
                location, velocity = actor.get_location(), actor.get_velocity()
                transform, box = actor.get_transform(), actor.bounding_box
            except RuntimeError as exc:
                if "trying to operate on a destroyed actor" not in str(exc):
                    raise
                continue
            yield actor, location, velocity, transform, box

    def _radar_hazard(self, name, readings, transform, progress, speed):
        """Select a relevant obstacle; retain raw detections for the policy."""
        readings = readings[np.isfinite(readings).all(axis=1)]
        extent = self.ego.bounding_box.extent
        candidates = []
        for point in readings:
            x, y = point[:2]
            world_point = transform.transform(self.carla.Location(
                x=float(x), y=float(y), z=float(point[2])))
            road = self.map.get_waypoint(world_point, project_to_road=False,
                                         lane_type=self.carla.LaneType.Driving)
            road_transform = getattr(road, "transform", None)
            # Downward radar rays also hit pavement. Compare against the road
            # plane at the hit, so grade and vehicle pitch do not hide obstacles.
            # Unknown road geometry retains the detection conservatively.
            if road_transform is not None and abs(ego_coordinates(road_transform, world_point)[2]) <= .1:
                continue
            if name in ("left", "right"):
                if abs(x) > extent.x + 3.:
                    continue
                clearance = max(0., abs(float(y)) - extent.y)
            else:
                if name == "front":
                    if x < 0.:
                        continue
                    # Keep immediate body-corridor obstacles even when the
                    # driver has not yet aligned with the planned bend.
                    stopping = speed * self.cfg.dt + speed**2 / (
                        2. * self.supervisor.config.red_light_deceleration_mps2)
                    immediate = x <= extent.x + max(3., stopping) and abs(y) <= extent.y + .5
                    position, lateral, _ = self.route.project((world_point.x, world_point.y), local=False)
                    if not immediate and (position < progress or abs(lateral) > extent.y + .5):
                        continue
                clearance = max(0., float(np.linalg.norm(point[:2])) - extent.x)
            candidates.append((clearance, point))
        return min(candidates, key=lambda item: item[0]) if candidates else None

    def _observe(self, frame):
        bundle = self.rig.get(frame, timeout=self.cfg.sensor_timeout)
        snapshot = self.world.get_snapshot()
        if snapshot.frame != frame:
            raise RuntimeError("Another client ticked CARLA; dedicated synchronous ownership lost")
        any_valid = any(bundle.valid.get(name, False) for name in ("rgb", "depth", "front", "left", "right", "rear"))
        if any_valid and abs(bundle.timestamp-snapshot.timestamp.elapsed_seconds) > 1e-5:
            raise ValueError("Sensor timestamp differs from the requested simulation state")
        failures = {*self.cfg.drop_sensors, *self.scenario.get("drop_sensors", []),
                    *self.scenario.get("stale_sensors", [])}
        for name in failures:
            if name == "rgb":
                bundle.rgb = None
            elif name == "depth":
                bundle.depth = None
            else:
                bundle.radars.pop(name, None)
            bundle.valid[name] = False
        transform, velocity = self.ego.get_transform(), self.ego.get_velocity()
        progress, lateral, heading = self.route.project((transform.location.x, transform.location.y))
        yaw = math.radians(transform.rotation.yaw)
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        weather = self.world.get_weather()
        speed_limit = self.ego.get_speed_limit() / 3.6
        waypoint = self.map.get_waypoint(transform.location)
        physical_waypoint = self.map.get_waypoint(
            transform.location, project_to_road=False, lane_type=self.carla.LaneType.Driving)
        lane_width, in_junction = self.route.lane_geometry(progress, waypoint.lane_width)
        departure_margin = 0.5 if in_junction else 0.0
        sensors_valid = bundle.valid.get("rgb", False) and all(bundle.valid.get(side, False)
                                                              for side in ("front", "left", "right", "rear"))
        state = DrivingState(speed_mps=speed, speed_limit_mps=speed_limit if speed_limit > 0 else None,
                             lane_error_m=lateral,
                             heading_error_rad=math.atan2(math.sin(yaw-heading), math.cos(yaw-heading)),
                             route_progress_m=progress, rain=weather.precipitation/100., wetness=weather.wetness/100.,
                             lane_width_m=lane_width,
                             night=float(weather.sun_altitude_angle < 0),
                             visibility_m=max(5., 100.*(1.-weather.fog_density/110.)),
                             lane_departure=abs(lateral) > lane_width/2. + departure_margin,
                             offroad=physical_waypoint is None,
                             observations_valid=sensors_valid)
        bumper = self.ego.bounding_box.extent.x
        crossing = next((c for c in self.crossings if c[2] >= progress - bumper), None)
        if crossing:
            state.crossing_id, entry, _ = crossing
            state.crossing_distance_m = entry - progress - bumper
        rows = []
        # Privileged actor states form only the explicitly selected GT provider.
        # Rewards and metrics retain oracle state separately in all modes.
        actors = list(self._actor_states())
        for actor, location, av, actor_transform, box in actors:
            relative = ego_coordinates(transform, location)
            distance = float(np.linalg.norm(relative))
            if distance > 80.:
                continue
            pedestrian = actor.type_id.startswith("walker.")
            child = actor.attributes.get("age") == "child" or actor.type_id in {
                "walker.pedestrian.%04d" % i for i in (9, 10, 11, 12, 13, 14, 48, 49)}
            relative_velocity = np.asarray(transform.get_inverse_matrix())[:3, :3] @ [
                av.x-velocity.x, av.y-velocity.y, av.z-velocity.z]
            if relative[0] > 0 and abs(relative[1]) < 2.5:
                state.front_hazard_m = min(state.front_hazard_m or float("inf"),
                                           max(0., relative[0]-bumper-box.extent.x))
                state.vehicle_ahead |= actor.type_id.startswith("vehicle.")
            if abs(relative[0]) < 5 and abs(relative[1]) > 1:
                clearance = max(0., abs(float(relative[1])) - self.ego.bounding_box.extent.y - box.extent.y)
                side = "left_hazard_m" if relative[1] < 0. else "right_hazard_m"
                current = getattr(state, side)
                setattr(state, side, min(current if current is not None else float("inf"), clearance))
                state.side_hazard_m = min(state.side_hazard_m if state.side_hazard_m is not None else
                                          float("inf"), clearance)
            if relative[0] < 0 and abs(relative[1]) < 3:
                state.rear_hazard_m = min(state.rear_hazard_m or float("inf"), abs(relative[0]))
            risk = float(pedestrian and -3 < relative[0] < max(15., speed*3.) and abs(relative[1]) < 7.)
            state.pedestrian_risk = max(state.pedestrian_risk, risk)
            state.child_pedestrian_risk = max(state.child_pedestrian_risk, risk * child)
            state.pedestrian_conflict |= bool(pedestrian and 0 < relative[0] < max(6., speed*2.) and
                                              abs(relative[1]) < 3.)
            if crossing and pedestrian:
                actor_s, actor_lateral, _ = self.route.project(
                    (location.x, location.y), local=False)
                state.crossing_occupied |= bool(crossing[1]-2 < actor_s < crossing[2]+2 and abs(actor_lateral) < 5)
            row = np.zeros(16, np.float32)
            row[:3], row[3:6] = relative / 100., relative_velocity / 30.
            row[6:11] = [float(pedestrian), float(child), 1., distance/100., 1.]
            # GT geometry remains identified separately from camera depth.
            # Project the actor box into the depth camera mounted with the RGB.
            camera_transform = self.rig.sensors[1].get_transform()
            vertices = box.get_world_vertices(actor_transform)
            camera_points = np.asarray([ego_coordinates(camera_transform, vertex) for vertex in vertices])
            visible = camera_points[:, 0] > 0.1
            if visible.all() and bundle.depth is not None:
                camera = self.rig.camera
                focal = camera.width / (2*math.tan(math.radians(camera.fov)/2))
                uv = np.column_stack((camera.width/2+focal*camera_points[:, 1]/camera_points[:, 0],
                                      camera.height/2-focal*camera_points[:, 2]/camera_points[:, 0]))
                image_box = np.r_[uv.min(0), uv.max(0)].reshape(1, 4)
                depth, uncertainty, valid = associate_depth(bundle.depth, image_box)
                expected = float(np.linalg.norm(ego_coordinates(camera_transform, location)))
                if valid[0] and abs(depth[0]-expected) < max(3., expected*0.25):
                    row[12:15] = [depth[0]/100., 1., uncertainty[0]/100.]
            rows.append((distance, row))
        self._observe_traffic_light(state, self.ego.bounding_box.location.x + bumper)
        signal_distance = state.red_light_distance_m if state.red_light else state.yellow_light_distance_m
        if (state.red_light or state.yellow_light) and signal_distance is not None:
            # Build the stationary queue backwards from the stop line.
            queue_end = state.route_progress_m + bumper + signal_distance
            vehicles = []
            for actor, loc, velocity_actor, _, box in actors:
                if not actor.type_id.startswith("vehicle."):
                    continue
                if math.hypot(velocity_actor.x, velocity_actor.y) > .2:
                    continue
                actor_s, lateral, _ = self.route.project((loc.x, loc.y), local=False)
                if abs(lateral) < 2.5 and state.route_progress_m < actor_s < queue_end:
                    vehicles.append((actor_s, box.extent.x))
            for position, extent in sorted(vehicles, reverse=True):
                if queue_end - (position + extent) > 8.:
                    break
                queue_end = position - extent
            state.red_light_queue = bool(vehicles and queue_end <
                                         state.route_progress_m + bumper + signal_distance and
                                         0 <= queue_end - (state.route_progress_m + bumper) <= 8.)
        collisions = self.rig.consume_collision(frame)
        state.collision = bool(collisions)
        state.pedestrian_collision = any(event["other_actor_type"].startswith("walker.") for event in collisions)
        self.oracle_state = replace(state)
        self.radar_hazards = {}
        objects = (np.stack([row for _, row in sorted(rows, key=lambda pair: pair[0])]) if rows else
                   np.zeros((0, 16), np.float32))
        if self.provider:
            prediction = self.provider.predict(bundle)
            prediction.validate_for(bundle)
            if self.cfg.perception == "external":
                # Only measured ego speed/navigation goal survive; no actor/map safety oracle.
                state = DrivingState(speed_mps=speed, observations_valid=bundle.valid.get("rgb", False),
                                     route_progress_m=progress, speed_limit_confidence=0.)
                required = {"speed_limit_mps", "speed_limit_confidence", "lane_error_m", "heading_error_rad",
                            "crossing_id", "crossing_distance_m", "crossing_occupied", "red_light",
                            "red_light_distance_m", "yellow_light", "yellow_light_distance_m",
                            "yellow_light_confidence", "visibility_m", "observations_valid"}
                missing_required = not required <= set(prediction.driving or {})
            else:
                missing_required = False
            fields = prediction.driving or {}
            if self.cfg.perception == "mixed":
                fields = {k: v for k, v in fields.items() if k in self.cfg.mixed_fields}
            forbidden = {"route_progress_m", "speed_mps", "collision", "pedestrian_collision", "offroad",
                         "lane_width_m", "traffic_light_id", "traffic_light_phase",
                         "traffic_light_distance_m",
                         "red_light_violation", "crossing_violation"}
            if forbidden & fields.keys():
                raise ValueError("External providers may replace perception fields only")
            state = replace(state, **fields)
            confidence = state.yellow_light_confidence
            distance = state.yellow_light_distance_m
            yellow_valid = (isinstance(state.yellow_light, (bool, np.bool_)) and
                            isinstance(confidence, (int, float, np.number)) and
                            not isinstance(confidence, (bool, np.bool_)) and
                            math.isfinite(confidence) and 0. <= confidence <= 1. and
                            (not state.yellow_light or (confidence > 0. and not state.red_light)) and
                            (distance is None or (isinstance(distance, (int, float, np.number)) and
                             not isinstance(distance, (bool, np.bool_)) and
                             math.isfinite(distance) and distance >= 0.)))
            state.observations_valid &= yellow_valid
            state.observations_valid &= not missing_required and sensors_valid
            objects = prediction.objects.copy()
            if not prediction.child_labels_authored:
                objects[:, 7] = 0.
                state.child_pedestrian_risk = 0.
            if prediction.depth_m is not None:
                objects[:, 12] = np.where(prediction.depth_valid, prediction.depth_m/100., 0.)
                objects[:, 13] = prediction.depth_valid
            if prediction.uncertainty is not None:
                uncertainty = prediction.uncertainty
                if uncertainty.ndim == 2:
                    uncertainty = np.max(uncertainty, axis=1)
                objects[:, 14] = np.where(prediction.uncertainty_valid, uncertainty/100., 0.)
            if prediction.object_valid is not None:
                objects = objects[prediction.object_valid]
            for row in objects:
                x, y = row[:2]*100.
                if x > 0 and abs(y) < 2.5:
                    state.front_hazard_m = min(state.front_hazard_m or float("inf"), max(0., float(x)-bumper))
                    state.vehicle_ahead |= bool(row[6] <= .5 and row[8] > .5 and
                                                np.linalg.norm(row[:3]) <= .8)
                if row[6] > 0.5 and -3 < x < max(15., speed*3.) and abs(y) < 7.:
                    state.pedestrian_risk = 1.
                    state.child_pedestrian_risk = max(state.child_pedestrian_risk, float(row[7] > 0.5))
                    state.pedestrian_conflict |= bool(0 < x < max(6., speed*2.) and abs(y) < 3.)
        # Side/rear/front radar hazards may influence safety in every mode.
        radars = {}
        for name, readings in bundle.radars.items():
            readings = readings.copy()
            if len(readings):
                if self.cfg.sensor_noise_std:
                    readings[:, :3] += self.rng.normal(0, self.cfg.sensor_noise_std, readings[:, :3].shape)
                hazard = self._radar_hazard(name, readings, transform, progress, speed)
                if hazard is not None:
                    clearance, point = hazard
                    self.radar_hazards[name] = {"point_ego_m": point[:3].tolist(), "clearance_m": clearance}
                    field = ("front_hazard_m" if name == "front" else
                             "rear_hazard_m" if name == "rear" else "side_hazard_m")
                    current = getattr(state, field)
                    setattr(state, field, min(current if current is not None else float("inf"), clearance))
                    if name in ("left", "right"):
                        field = ("left" if point[1] < 0. else "right") + "_hazard_m"
                        current = getattr(state, field)
                        setattr(state, field, min(current if current is not None else float("inf"), clearance))
            radars[name] = readings
        critical = [state.speed_mps, state.speed_limit_confidence, state.lane_error_m, state.heading_error_rad,
                    state.rain, state.wetness, state.night, state.visibility_m, state.pedestrian_risk]
        state.observations_valid &= bool(np.isfinite(critical).all())
        context = np.asarray([state.speed_mps/30., (state.speed_limit_mps or 0)/30., state.speed_limit_confidence,
                              state.lane_error_m/4., state.heading_error_rad/math.pi,
                              (self.route.length-progress)/1000.,
                              state.rain, state.wetness, state.night, state.visibility_m/100.,
                              (state.crossing_distance_m if state.crossing_distance_m is not None else 100.)/100.,
                              float(state.crossing_occupied), float(state.red_light), state.pedestrian_risk,
                              state.child_pedestrian_risk, float(state.observations_valid)], np.float32)
        navigation = []
        for ahead in (5., 15., 30.):
            point_s = min(self.route.length, progress+ahead)
            index = min(int(np.searchsorted(self.route.cumulative, point_s)-1), len(self.route.delta)-1)
            fraction = (point_s-self.route.cumulative[index])/self.route.lengths[index]
            point = self.route.points[index]+fraction*self.route.delta[index]
            relative = ego_coordinates(transform, self.carla.Location(
                x=float(point[0]), y=float(point[1]), z=transform.location.z))
            token = np.zeros(16, np.float32)
            token[:3], token[15] = relative/100., -1.
            if state.red_light and state.red_light_distance_m is not None:
                token[3], token[4] = state.red_light_distance_m / 100., 1.
            if state.yellow_light:
                token[5] = 1.
                token[6] = (state.yellow_light_distance_m / 100.
                            if state.yellow_light_distance_m is not None else -1.)
                token[7] = state.yellow_light_confidence
            navigation.append(token)
        objects = np.concatenate((np.asarray(navigation), objects[:self.cfg.max_objects]), axis=0)
        observation = pack_observation(bundle.rgb, context, objects, radars, max_objects=self.cfg.max_objects+3,
                                       model_size=(self.cfg.image_size, self.cfg.image_size), radar_valid=bundle.valid)
        self.last_bundle, self.frame = bundle, frame
        return state, observation

    def step(self, action):
        before = self.state
        safety_before = asdict(before)
        radar_before = self.radar_hazards
        oracle_before = self.oracle_before
        if getattr(self, "demo_pending", None) or getattr(self, "demo_walkers", None):
            self._demo_encounters(oracle_before.route_progress_m)
        result = self.supervisor.step(before, action, self.cfg.dt)
        steer, throttle, brake = action_to_controls(result.action)
        if result.reverse:  # a negative longitudinal means throttle in reverse here, not brake
            throttle, brake = -result.action[1], 0.0
        self.ego.apply_control(self.carla.VehicleControl(steer=steer, throttle=throttle, brake=brake,
                                                         reverse=result.reverse))
        self._update_npcs()
        self.steps += 1
        state, observation = self._observe(self.world.tick())
        oracle = self.oracle_state
        if (oracle_before.crossing_id == oracle.crossing_id and oracle_before.crossing_distance_m is not None and
                oracle.crossing_distance_m is not None):
            oracle.crossing_violation = (oracle_before.crossing_distance_m >= 0 > oracle.crossing_distance_m and
                                         result.crossing_phase not in ("resume", "passed"))
        oracle.red_light_violation = bool(oracle_before.red_light and oracle_before.red_light_distance_m is not None and
                                          oracle_before.route_progress_m+oracle_before.red_light_distance_m <
                                          oracle.route_progress_m)
        success = (oracle.route_progress_m >= self.route.length - 3. and
                   not oracle.collision and not oracle.pedestrian_collision and not oracle.offroad)
        termination_reason = next((reason for reason, active in (
            ("pedestrian_collision", oracle.pedestrian_collision),
            ("collision", oracle.collision), ("offroad", oracle.offroad),
            ("lane_deviation", abs(oracle.lane_error_m) > 6.),
            ("red_light_violation", oracle.red_light_violation),
            ("crossing_violation", oracle.crossing_violation), ("success", success),
        ) if active), None)
        terminated = termination_reason is not None
        highwater_before = min(self.route.length, max(0.0, self.reward_progress_highwater))
        highwater_after = max(highwater_before, min(self.route.length, max(0.0, oracle.route_progress_m)))
        waypoints_reached = (int(highwater_after // WAYPOINT_SPACING_M) -
                             int(highwater_before // WAYPOINT_SPACING_M))
        new_progress_m = highwater_after - highwater_before
        self.reward_progress_highwater = highwater_after
        can_move = (before.observations_valid and state.observations_valid and
                    result.target_speed_mps > 0.5 and result.crossing_phase != "stopped" and
                    not (before.red_light or state.red_light or
                         before.pedestrian_conflict or state.pedestrian_conflict) and
                    not {"invalid_action", "invalid_observation", "speed_target_brake"}.intersection(
                        result.intervention_reasons))
        wait = signal_wait(oracle, self.supervisor.config)
        if not wait:
            self.timed_steps = getattr(self, "timed_steps", 0) + 1
        watchdog_reason = (None if wait else
                           self.inactivity.update(oracle.route_progress_m, self.cfg.dt, can_move=can_move))
        truncation_reason = None
        if not terminated:
            if getattr(self, "timed_steps", 0) >= self.cfg.max_episode_steps:
                truncation_reason = "time_limit"
            else:
                truncation_reason = watchdog_reason
        truncated = truncation_reason is not None
        reward = compute_reward(oracle_before, result.action, oracle,
                                 self.cfg.dt, terminated=terminated, truncated=truncated, success=success,
                                 previous_action=self.previous_action, previous_acceleration=self.previous_acceleration,
                                 waypoints_reached=waypoints_reached, new_progress_m=new_progress_m,
                                 target_speed_mps=result.target_speed_mps,
                                 pause_time=wait,
                                 mandated_stop=(wait or result.crossing_phase == "stopped" or
                                                before.pedestrian_conflict or not before.observations_valid),
                                 config=self.reward_config)
        self.episode_return = getattr(self, "episode_return", 0.0) + reward.reward
        reward_limit = self.episode_return < MIN_EPISODE_RETURN and not terminated
        if reward_limit:
            terminated, truncated, truncation_reason = True, False, None
        self.previous_acceleration = (state.speed_mps-before.speed_mps)/self.cfg.dt
        self.oracle_before = replace(oracle)
        self.previous_action, self.state = result.action, state
        info = {"frame": self.frame, "timestamp": self.last_bundle.timestamp, "applied_action": result.action,
                "safety_before": safety_before, "radar_before": radar_before,
                "scenario": self.scenario.get("name", "basic"),
                "route_start": getattr(self, "route_start", self.scenario.get("route_start", self.cfg.route_start)),
                "route_end": getattr(self, "route_end", self.scenario.get("route_end", self.cfg.route_end)),
                "episode_seed": getattr(self, "episode_seed", self.cfg.seed),
                "weather": self.scenario.get("weather", self.cfg.weather),
                "requested_action": list(map(float, action)), "interventions": result.intervention_reasons,
                "crossing_phase": result.crossing_phase, "target_speed_mps": result.target_speed_mps,
                "restart_assist_phase": result.restart_assist_phase,
                "restart_assist_event": result.restart_assist_event,
                "restart_assist_elapsed_s": result.restart_assist_elapsed_s,
                "waypoints_reached": waypoints_reached, "new_progress_m": new_progress_m,
                "reward_components": reward.components, "success": success,
                "truncation_reason": truncation_reason,
                "termination_reason": "reward_limit" if reward_limit else termination_reason,
                "episode_return": self.episode_return,
                "policy_observations_valid": state.observations_valid, "perception_source": self.cfg.perception,
                "external_fields": self.cfg.mixed_fields if self.cfg.perception == "mixed" else self.cfg.perception,
                "sensor_valid": dict(self.last_bundle.valid),
                "route_length_m": self.route.length, **asdict(oracle)}
        if self.cfg.actor_longitudinal_bc_weight:
            # Labels describe the next policy state, aligned with RSSM's
            # reconstructed posterior. Oracle-only fields never enter them.
            valid = bool(state.observations_valid and not (terminated or truncated) and
                         math.isfinite(state.speed_mps))
            target = self.supervisor.longitudinal_reference(state, self.cfg.dt) if valid else 0.
            if not math.isfinite(target):
                target, valid = 0., False
            info["longitudinal_target"] = target
            info["longitudinal_target_valid"] = valid
        return observation, reward.reward, terminated, truncated, info

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._destroy()
            if self.provider and hasattr(self.provider, "close"):
                self.provider.close()
        finally:
            try:
                if getattr(self, "traffic_manager", None) is not None:
                    self.traffic_manager.set_synchronous_mode(False)
            finally:
                if self.owns_settings:
                    self.world.apply_settings(self.settings)
                    self.world.set_weather(self.weather)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Reproducible CARLA scenario selection and lightweight episode metrics."""

from dataclasses import asdict, is_dataclass, replace
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional

from .weather import validate_weather_overrides


SPLITS = ("train", "validation", "held_out")

# The current environment can author routes, weather, actor placements and
# sensor drops. Other categories remain explicit catalog entries with reasons;
# evaluation reports them as unsupported instead of claiming coverage.
_SCENARIO_TEMPLATES = [
    {"name": "straight", "category": "straight"},
    {"name": "curved", "category": "curved"},
    {"name": "intersection", "category": "intersection"},
    {"name": "moving_traffic", "category": "moving_traffic",
     "actors": [{"kind": "vehicle", "x": 22, "y": 0, "speed_mps": 3}]},
    {"name": "static_obstacle", "category": "static_obstacle",
     "actors": [{"kind": "vehicle", "x": 20, "y": 0, "speed_mps": 0}]},
    {"name": "pedestrian_approach", "category": "pedestrian_approaching",
     "actors": [{"kind": "pedestrian", "x": 12, "y": 4, "yaw": 180, "speed_mps": 1}]},
    {"name": "pedestrian_crossing", "category": "pedestrian_crossing",
     "actors": [{"kind": "pedestrian", "x": 18, "y": -5, "yaw": 90, "speed_mps": 1.2}]},
    {"name": "child_pedestrian", "category": "child_pedestrian",
     "actors": [{"kind": "pedestrian", "child": True, "x": 18, "y": -5,
                 "yaw": 90, "speed_mps": 1.2}]},
    {"name": "front_hazard", "category": "front_hazard",
     "actors": [{"kind": "vehicle", "x": 10, "y": 0}]},
    {"name": "side_hazard", "category": "side_hazard",
     "actors": [{"kind": "vehicle", "x": 0, "y": 3}]},
    {"name": "rear_hazard", "category": "rear_hazard",
     "actors": [{"kind": "vehicle", "x": -7, "y": 0}]},
    {"name": "occupied_crossing", "category": "occupied_crossing", "requires_crosswalk": True,
     "actors": [{"kind": "pedestrian", "at_crossing": True, "yaw": 90, "speed_mps": 0}]},
    {"name": "unoccupied_crossing", "category": "unoccupied_crossing", "requires_crosswalk": True},
    {"name": "speed_limit_transition", "category": "speed_limit_transition",
     "requires_speed_limit_transition": True},
    {"name": "heavy_rain", "category": "heavy_rain", "weather": "HardRainNoon"},
    {"name": "low_light", "category": "low_light", "weather": "ClearNight"},
    {"name": "combined_weather_hazard", "category": "combined_weather_hazard",
     "weather": "HardRainNight"},
    {"name": "missing_rgb", "category": "missing_sensor", "drop_sensors": ["rgb"]},
    {"name": "stale_sensor", "category": "stale_sensor", "stale_sensors": ["rgb"]},
]


def scenario_catalog(cfg=None) -> List[dict]:
    """Return deterministic scenario definitions across all three data splits."""
    seed = int(getattr(cfg, "seed", 0))
    route_start = int(getattr(cfg, "route_start", 0))
    route_end = int(getattr(cfg, "route_end", 10))
    records = []
    for split_index, split in enumerate(SPLITS):
        for index, template in enumerate(_SCENARIO_TEMPLATES):
            start = route_start + split_index * 40 + index * 2
            end = route_end + split_index * 40 + index * 2
            record = copy.deepcopy(template)
            record.update(name=f"{split}_{template['name']}", split=split,
                          seed=seed + split_index * 10000 + index,
                          route_start=start, route_end=end,
                          supported=template.get("supported", True))
            records.append(record)
    return records


def _validate_catalog(records: List[dict]) -> None:
    names, seeds, routes = set(), set(), set()
    for record in records:
        required = {"name", "split", "seed", "route_start", "route_end", "category"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Scenario missing fields: {sorted(missing)}")
        if record["split"] not in SPLITS:
            raise ValueError(f"Unknown scenario split: {record['split']}")
        if record["route_start"] == record["route_end"]:
            raise ValueError(f"Scenario route must have distinct endpoints: {record['name']}")
        if "route_length_m" in record:
            length = record["route_length_m"]
            if (isinstance(length, bool) or not isinstance(length, (int, float)) or
                    not math.isfinite(length) or length <= 0):
                raise ValueError("scenario route_length_m must be finite and positive")
        if not isinstance(record.get("background_traffic", True), bool):
            raise ValueError("scenario background_traffic must be a boolean")
        for requirement in ("requires_traffic_light", "requires_pedestrians", "requires_vehicles"):
            if requirement in record and not isinstance(record[requirement], bool):
                raise ValueError(f"scenario {requirement} must be a boolean")
        if record.get("requires_pedestrians") and not any(
                actor.get("kind") == "pedestrian" for actor in record.get("actors", [])):
            raise ValueError("scenario requires_pedestrians needs authored pedestrian actors")
        if record.get("requires_vehicles") and not any(
                actor.get("kind", "vehicle") == "vehicle" for actor in record.get("actors", [])):
            raise ValueError("scenario requires_vehicles needs authored vehicle actors")
        validate_weather_overrides(record)
        if "randomize_routes" in record and not isinstance(record["randomize_routes"], bool):
            raise ValueError("scenario randomize_routes must be a boolean")
        if "max_heading_change_rad" in record:
            heading = record["max_heading_change_rad"]
            if (isinstance(heading, bool) or not isinstance(heading, (int, float)) or
                    not math.isfinite(heading) or heading <= 0):
                raise ValueError("scenario max_heading_change_rad must be finite and positive")
        route = (record["route_start"], record["route_end"])
        name_key = (record["split"], record["name"])
        if name_key in names or record["seed"] in seeds or route in routes:
            raise ValueError("Scenario names, seeds, and routes must be unique across splits")
        names.add(name_key)
        seeds.add(record["seed"])
        routes.add(route)


def scenario_records(cfg) -> List[dict]:
    """Read all splits so collection can reserve evaluation route pairs.

    A JSON catalog may be a list of records or ``{"scenarios": [...]}``.
    Records must explicitly declare split, seed and distinct route endpoints.
    """
    path = getattr(cfg, "scenario_file", "")
    if path:
        payload = json.loads(Path(path).read_text())
        records = payload.get("scenarios", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("Scenario file must contain a list of scenario objects")
        records = [dict(record) for record in records]
        _validate_catalog(records)
    else:
        records = scenario_catalog(cfg)
    return records


def scenario_suite(cfg) -> List[dict]:
    """Return the configured split from a JSON catalog or built-in catalog."""
    return [record for record in scenario_records(cfg) if record["split"] == cfg.split]


def collection_scenarios(cfg) -> List[dict]:
    """Resolve collection names in order, restricted to the configured split."""
    names = cfg.scenarios or [cfg.scenario]
    catalog = {record["name"]: record for record in scenario_suite(cfg)}
    selected = []
    for name in names:
        if name == "basic":
            selected.append({"name": "basic", "weather": cfg.weather})
            continue
        if name not in catalog:
            raise ValueError(f"Scenario {name!r} is absent from selected split {cfg.split!r}")
        record = catalog[name]
        if not record.get("supported", True):
            raise ValueError(f"Unsupported scenario: {name}")
        selected.append(record)
    return selected


def episode_metrics(infos: Iterable[dict], dt_s: float = 0.05) -> dict:
    """Summarize an episode's environment info stream without simulator imports."""
    rows = list(infos)
    if not rows:
        return {"steps": 0, "duration_s": 0.0, "route_completion": 0.0,
                "distance_traveled_m": 0.0, "progress_per_s": 0.0}
    metrics = {
        "steps": len(rows), "duration_s": 0.0, "route_completion": 0.0,
        "distance_traveled_m": 0.0, "progress_per_s": 0.0,
        "lane_departures": 0, "collision_events": 0, "pedestrian_incidents": 0,
        "traffic_violations": 0, "minimum_relevant_separation_m": None,
        "crossings_encountered": 0, "crossing_violations": 0,
        "crossing_compliance_rate": 0.0,
        "red_light_violations": 0, "speed_violation_events": 0,
        "red_light_observation_steps": 0, "yellow_light_observation_steps": 0,
        "pedestrian_encounter_steps": 0,
        "pedestrian_conflict_events": 0,
        "speed_violation_duration_s": 0.0, "speed_excess_mean_mps": 0.0,
        "speed_excess_integral_mps_s": 0.0,
        "speed_tracking_error_mean_mps": 0.0, "unnecessary_stopping_s": 0.0,
        "unnecessary_stop_events": 0, "steering_change_mean": 0.0,
        "longitudinal_change_mean": 0.0, "acceleration_rms_mps2": 0.0,
        "jerk_rms_mps3": 0.0,
        "supervisor_intervention_steps": 0, "intervention_reasons": {}, "missing_observation_steps": 0,
        "missing_observation_fail_safe_steps": 0,
        "policy_inference_latency_mean_s": 0.0,
        "policy_inference_latency_p95_s": 0.0,
    }
    previous = None
    previous_progress = None
    previous_lane_departure = previous_collision = previous_ped_collision = False
    previous_ped_conflict = False
    previous_speed_violation = previous_stopped = False
    previous_crossing = None
    previous_steer = previous_longitudinal = None
    previous_accel = None
    step_durations = []
    inference_latencies = []
    forward_progress = speed_excess_sum = tracking_sum = 0.0
    steering_changes, longitudinal_changes, accelerations, jerks = [], [], [], []
    min_separation = None

    for row in rows:
        metrics["red_light_observation_steps"] += int(bool(row.get("red_light")))
        metrics["yellow_light_observation_steps"] += int(bool(row.get("yellow_light")))
        metrics["pedestrian_encounter_steps"] += int(max(
            float(row.get("pedestrian_risk") or 0.), float(row.get("child_pedestrian_risk") or 0.)) > 0.)
        timestamp = row.get("timestamp")
        if previous is None:
            dt = max(0.0, float(dt_s))
        else:
            prior_ts = previous.get("timestamp")
            delta = (float(timestamp) - float(prior_ts)
                     if timestamp is not None and prior_ts is not None else dt_s)
            dt = delta if math.isfinite(delta) and delta > 0 else max(0.0, float(dt_s))
        metrics["duration_s"] += dt
        step_durations.append(dt)

        progress = row.get("route_progress_m")
        if progress is not None and math.isfinite(float(progress)):
            if previous_progress is not None:
                delta_progress = float(progress) - previous_progress
                forward_progress += max(0.0, delta_progress)
            previous_progress = float(progress)
        route_length = row.get("route_length_m")
        if route_length and float(route_length) > 0 and progress is not None:
            metrics["route_completion"] = max(metrics["route_completion"],
                                               min(1.0, max(0.0, float(progress) / float(route_length))))

        lane_departure = bool(row.get("lane_departure", False))
        if lane_departure and not previous_lane_departure:
            metrics["lane_departures"] += 1
        previous_lane_departure = lane_departure
        collision = bool(row.get("collision", False))
        ped_collision = bool(row.get("pedestrian_collision", False))
        if collision and not previous_collision:
            metrics["collision_events"] += 1
        if ped_collision and not previous_ped_collision:
            metrics["pedestrian_incidents"] += 1
        previous_collision, previous_ped_collision = collision, ped_collision
        ped_conflict = bool(row.get("pedestrian_conflict", False))
        if ped_conflict and not previous_ped_conflict:
            metrics["pedestrian_conflict_events"] += 1
        previous_ped_conflict = ped_conflict
        crossing_violation = bool(row.get("crossing_violation", False))
        red_violation = bool(row.get("red_light_violation", False))
        metrics["crossing_violations"] += int(crossing_violation and not (
            previous and previous.get("crossing_violation")))
        metrics["red_light_violations"] += int(red_violation and not (previous and previous.get("red_light_violation")))

        crossing_id = row.get("crossing_id")
        crossing_distance = row.get("crossing_distance_m")
        if (crossing_id is not None and crossing_id != previous_crossing and
                crossing_distance is not None and crossing_distance >= 0):
            metrics["crossings_encountered"] += 1
        previous_crossing = crossing_id
        for key in ("front_hazard_m", "side_hazard_m", "rear_hazard_m",
                    "pedestrian_separation_m", "minimum_separation_m"):
            separation = row.get(key)
            if separation is not None and math.isfinite(float(separation)) and float(separation) >= 0:
                min_separation = float(separation) if min_separation is None else min(min_separation, float(separation))

        speed = float(row.get("speed_mps", 0.0))
        limit = row.get("speed_limit_mps")
        if limit is not None and math.isfinite(float(limit)) and float(limit) > 0:
            excess = max(0.0, speed - float(limit))
        else:
            excess = 0.0
        over = excess > 1e-6
        if over and not previous_speed_violation:
            metrics["speed_violation_events"] += 1
        if over:
            metrics["speed_violation_duration_s"] += dt
            speed_excess_sum += excess * dt
        metrics["traffic_violations"] = (metrics["crossing_violations"] +
                                         metrics["red_light_violations"] +
                                         metrics["speed_violation_events"])
        previous_speed_violation = over
        target = row.get("target_speed_mps")
        if target is not None and math.isfinite(float(target)):
            tracking_sum += abs(speed - float(target)) * dt

        mandated_stop = (bool(row.get("red_light", False)) or
                         bool(row.get("pedestrian_conflict", False)) or
                         bool(row.get("crossing_occupied", False)) or
                         row.get("crossing_phase") == "stopped" or
                         (crossing_distance is not None and 0 <= crossing_distance <=
                          2.0 + max(0.0, speed) ** 2 / 6.0))
        stopped = speed < 0.5
        unnecessary = stopped and not mandated_stop
        if unnecessary:
            metrics["unnecessary_stopping_s"] += dt
        if unnecessary and not previous_stopped:
            metrics["unnecessary_stop_events"] += 1
        previous_stopped = unnecessary

        reasons = row.get("interventions", ()) or ()
        metrics["supervisor_intervention_steps"] += int(bool(reasons))
        for reason in reasons:
            metrics["intervention_reasons"][str(reason)] = metrics["intervention_reasons"].get(str(reason), 0) + 1
        if not bool(row.get("policy_observations_valid", row.get("observations_valid", True))):
            metrics["missing_observation_steps"] += 1
            action = row.get("applied_action", (0, 0))
            if len(action) >= 2 and float(action[1]) <= -0.99:
                metrics["missing_observation_fail_safe_steps"] += 1
        latency = row.get("policy_inference_latency_s")
        if latency is not None and math.isfinite(float(latency)) and float(latency) >= 0:
            inference_latencies.append(float(latency))

        action = row.get("applied_action")
        if action is not None and len(action) >= 2:
            steer, longitudinal = float(action[0]), float(action[1])
            if previous_steer is not None:
                steering_changes.append(abs(steer - previous_steer))
                longitudinal_changes.append(abs(longitudinal - previous_longitudinal))
            previous_steer, previous_longitudinal = steer, longitudinal
        if previous is not None:
            acceleration = (speed - float(previous.get("speed_mps", speed))) / max(dt, 1e-6)
            accelerations.append(acceleration)
            if previous_accel is not None:
                jerks.append((acceleration - previous_accel) / max(dt, 1e-6))
            previous_accel = acceleration
        previous = row

    metrics["distance_traveled_m"] = sum(max(0.0, float(row.get("speed_mps", 0.0))) * dt
                                          for row, dt in zip(rows, step_durations))
    metrics["progress_per_s"] = forward_progress / max(metrics["duration_s"], 1e-6)
    metrics["speed_excess_mean_mps"] = speed_excess_sum / max(metrics["speed_violation_duration_s"], 1e-6)
    metrics["speed_excess_integral_mps_s"] = speed_excess_sum
    metrics["speed_tracking_error_mean_mps"] = tracking_sum / max(metrics["duration_s"], 1e-6)
    if metrics["crossings_encountered"]:
        metrics["crossing_compliance_rate"] = max(
            0.0, 1.0 - metrics["crossing_violations"] / metrics["crossings_encountered"])
    metrics["minimum_relevant_separation_m"] = min_separation
    metrics["steering_change_mean"] = _mean(steering_changes)
    metrics["longitudinal_change_mean"] = _mean(longitudinal_changes)
    metrics["acceleration_rms_mps2"] = _rms(accelerations)
    metrics["jerk_rms_mps3"] = _rms(jerks)
    metrics["policy_inference_latency_mean_s"] = _mean(inference_latencies)
    if inference_latencies:
        ordered_latencies = sorted(inference_latencies)
        metrics["policy_inference_latency_p95_s"] = ordered_latencies[
            max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)]
    return metrics


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def evaluate(cfg, agent, env_factory=None, episodes: Optional[int] = None,
             deterministic: bool = True, *, fixed_scenario=False, trace_path=None, observer=None,
             action_source="actor") -> dict:
    """Run selected scenarios; unsupported cases are recorded without claims.

    ``env_factory`` receives ``(cfg, scenario)`` and returns a reset/step/close
    environment. The agent follows ``initial()`` and ``act(obs, carry, bool)``.
    """
    if env_factory is None:
        from .environment import CarlaEnv
        env_factory = CarlaEnv
    explicit_selection = cfg.scenario != "basic" or bool(cfg.scenarios)
    if fixed_scenario or explicit_selection:
        suite = collection_scenarios(cfg)
        if fixed_scenario and len(suite) != 1:
            raise ValueError("Fixed evaluation requires exactly one scenario")
        suite = [{"category": "basic", "split": cfg.split, "seed": cfg.seed,
                  "route_start": cfg.route_start, "route_end": cfg.route_end, **record} for record in suite]
    else:
        suite = scenario_suite(cfg)
    if not suite:
        raise ValueError(f"No evaluation scenarios available for split {cfg.split!r}")
    if episodes is not None:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        suite = suite[:episodes]
    results, unsupported, frames = [], [], 0
    for episode, scenario in enumerate(suite):
        if not scenario.get("supported", True):
            unsupported.append({"name": scenario["name"], "category": scenario["category"],
                                "split": scenario["split"], "seed": scenario["seed"],
                                "route": [scenario["route_start"], scenario["route_end"]],
                                "reason": scenario.get("unsupported_reason", "catalogued unsupported")})
            continue
        run_cfg = replace(cfg, seed=scenario["seed"])
        from .training import seed_all

        seed_all(run_cfg.seed)
        env = None
        evaluation_started = time.perf_counter()
        try:
            env = env_factory(run_cfg, scenario)
            observation = env.reset()
            carry = agent.initial()
            if observer is not None:
                observer.publish(env, observation, {"frames": frames, "episode": episode,
                                                    "total_episodes": len(suite), "scenario": scenario["name"]},
                                 reset=True)
            infos = []
            terminated = truncated = False
            while not (terminated or truncated):
                inference_started = time.perf_counter()
                action, carry = agent.act(observation, carry, deterministic)
                inference_latency = time.perf_counter() - inference_started
                observation, reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                info["policy_inference_latency_s"] = inference_latency
                if hasattr(carry, "set"):
                    import torch

                    carry_device = carry.device or carry.get("state").device
                    previous_action = info.get("applied_action", action)
                    carry.set("previous_action", torch.as_tensor(
                        previous_action, dtype=torch.float32, device=carry_device))
                    carry.set("is_init", torch.zeros((1,), dtype=torch.bool,
                                                       device=carry_device))
                infos.append(info)
                frames += 1
                if observer is not None:
                    observer.publish(env, observation, {**info, "frames": frames, "episode": episode,
                                                        "total_episodes": len(suite), "scenario": scenario["name"],
                                                        "reward": reward, "terminated": terminated,
                                                        "truncated": truncated, "action_source": action_source})
                if trace_path is not None:
                    with Path(trace_path).open("a") as trace:
                        trace.write(json.dumps({"scenario": scenario["name"], **info}, allow_nan=False) + "\n")
            metrics = episode_metrics(infos, cfg.dt)
            metrics["stopped_fraction"] = sum(row.get("speed_mps", 0.) < 0.1 for row in infos) / max(1, len(infos))
            metrics["action_override_fraction"] = sum(
                any(abs(a-b) > 1e-5 for a, b in zip(row.get("requested_action", [0, 0]),
                                                   row.get("applied_action", [0, 0])))
                for row in infos) / max(1, len(infos))
            metrics["throttle_request_block_fraction"] = sum(
                row.get("requested_action", [0, 0])[1] > 0.1 and
                row.get("applied_action", [0, 0])[1] <= 0 for row in infos) / max(1, sum(
                    row.get("requested_action", [0, 0])[1] > 0.1 for row in infos))
            length = infos[-1].get("route_length_m", 0.) if infos else 0.
            metrics["required_average_speed_mps"] = length / (cfg.max_episode_steps * cfg.dt)
            metrics["evaluation_wall_time_s"] = time.perf_counter() - evaluation_started
            metrics["evaluation_steps_per_wall_s"] = len(infos) / max(metrics["evaluation_wall_time_s"], 1e-6)
            results.append({"name": scenario["name"], "category": scenario["category"],
                            "split": scenario["split"], "seed": scenario["seed"],
                            "route": [infos[-1].get("route_start", scenario["route_start"]),
                                      infos[-1].get("route_end", scenario["route_end"])],
                            "scenario": scenario,
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "success": bool(infos[-1].get("success", False)), "metrics": metrics})
        except (RuntimeError, ValueError) as exc:
            unsupported_markers = ("Unsupported scenario", "Scenario actor placement failed", "No actor blueprint",
                                   "Route spawn indices must be distinct and available")
            if not any(marker in str(exc) for marker in unsupported_markers):
                raise
            unsupported.append({"name": scenario["name"], "category": scenario["category"],
                                "split": scenario["split"], "seed": scenario["seed"],
                                "route": [scenario["route_start"], scenario["route_end"]],
                                "reason": str(exc)})
        finally:
            if env is not None:
                env.close()
    checkpoint = getattr(agent, "checkpoint_path", getattr(agent, "checkpoint_id", None))
    training_sanity = fixed_scenario or (explicit_selection and cfg.split == "train")
    return {"scope": "fixed_training_sanity" if training_sanity else "split_evaluation",
            "split": cfg.split, "episodes": results, "unsupported_scenarios": unsupported,
            "summary": aggregate_metrics(results), "config": _config_dict(cfg),
            "checkpoint_provenance": checkpoint}


def aggregate_metrics(episodes: Iterable[dict]) -> dict:
    episodes = list(episodes)
    if not episodes:
        return {"episode_count": 0}
    keys = [key for key, value in episodes[0]["metrics"].items()
            if isinstance(value, (int, float)) and value is not None]
    summary = {"episode_count": len(episodes)}
    for key in keys:
        values = [float(ep["metrics"][key]) for ep in episodes if ep["metrics"].get(key) is not None]
        summary[key + "_mean"] = _mean(values)
    summary["success_rate"] = _mean([float(ep["success"]) for ep in episodes])
    return summary


def _config_dict(cfg):
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def compare_results(baseline: dict, candidate: dict,
                    metric_keys: Optional[Iterable[str]] = None, *, fixed_scenario=False) -> dict:
    """Compare matched measured runs, allowing only attention to differ."""
    def policy_metadata(result):
        policy = result.get("policy")
        if policy is None:
            legacy_speed = result.get("controller_speed_mps")
            if result.get("scope") == "fixed_training_sanity" and legacy_speed is not None:
                policy = {"type": "route_controller", "target_speed_mps": legacy_speed}
            else:
                policy = {"type": "actor"}
        if not isinstance(policy, dict) or policy.get("type") not in ("actor", "route_controller"):
            raise ValueError("Comparison requires recognized actor or route_controller policy metadata")
        fixed = policy.get("type") == "route_controller" and set(policy) == {"type", "target_speed_mps"}
        dynamic = (policy.get("type") == "route_controller" and
                   policy == {"type": "route_controller", "speed_mode": "speed_limit"})
        if ((policy["type"] == "actor" and set(policy) != {"type"}) or
                (policy["type"] == "route_controller" and not (fixed or dynamic))):
            raise ValueError("Policy metadata has missing or unsupported fields")
        if fixed:
            speed = policy.get("target_speed_mps")
            if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or
                    not math.isfinite(speed) or speed <= 0):
                raise ValueError("Route controller metadata requires a positive finite target speed")
        return policy

    base_policy = policy_metadata(baseline)
    candidate_policy = policy_metadata(candidate)
    policy_comparison = base_policy.get("type") != candidate_policy.get("type")
    if policy_comparison and {base_policy["type"], candidate_policy["type"]} != {"actor", "route_controller"}:
        raise ValueError("Policy comparison is supported only between actor and route_controller")
    base_cfg, cand_cfg = dict(baseline.get("config", {})), dict(candidate.get("config", {}))
    if fixed_scenario:
        if any(result.get("scope") != "fixed_training_sanity" for result in (baseline, candidate)):
            raise ValueError("Fixed comparison requires fixed-scenario results")
        for settings in (base_cfg, cand_cfg):
            for key in ("steps", "checkpoint_every", "device", "warmup_frames", "warmup_policy",
                        "world_model_warmup_updates", "capacity", "actor_bc_weight", "expert_episode_interval",
                        "actor_steering_bc_weight", "actor_longitudinal_bc_weight"):
                settings.pop(key, None)
    elif policy_comparison:
        for settings in (base_cfg, cand_cfg):
            for key in ("steps", "checkpoint_every", "device", "warmup_frames", "warmup_policy",
                        "world_model_warmup_updates", "capacity", "actor_bc_weight", "expert_episode_interval",
                        "actor_steering_bc_weight", "actor_longitudinal_bc_weight"):
                settings.pop(key, None)
        if base_cfg != cand_cfg:
            raise ValueError("Policy comparison requires identical task configs")
    elif (not isinstance(base_cfg.get("attention"), bool) or
            not isinstance(cand_cfg.get("attention"), bool) or
            base_cfg["attention"] == cand_cfg["attention"]):
        raise ValueError("Comparison requires different attention settings")
    else:
        base_cfg.pop("attention", None)
        cand_cfg.pop("attention", None)
    if base_cfg != cand_cfg:
        raise ValueError("Matched comparison requires configs identical except attention")
    def match_key(ep):
        scenario_json = json.dumps(ep.get("scenario", {}), sort_keys=True)
        return ep["name"], ep["seed"], tuple(ep["route"]), scenario_json

    base = {match_key(ep): ep for ep in baseline.get("episodes", [])}
    cand = {match_key(ep): ep for ep in candidate.get("episodes", [])}
    if not base or base.keys() != cand.keys():
        raise ValueError("Matched comparison requires the same nonempty scenarios, seeds and routes")
    rows = []
    for key in sorted(base):
        bmetrics, cmetrics = base[key]["metrics"], cand[key]["metrics"]
        selected = list(metric_keys) if metric_keys is not None else [
            name for name, value in bmetrics.items()
            if isinstance(value, (int, float)) and cmetrics.get(name) is not None]
        delta = {name: float(cmetrics[name]) - float(bmetrics[name]) for name in selected}
        rows.append({"scenario": key[0], "seed": key[1], "route": list(key[2]),
                     "delta_candidate_minus_baseline": delta})
    summary = {}
    for name in rows[0]["delta_candidate_minus_baseline"]:
        summary[name + "_mean_delta"] = _mean([row["delta_candidate_minus_baseline"][name] for row in rows])
    return {"episode_count": len(rows), "rows": rows, "summary": summary,
            "checkpoint_provenance": {"baseline": baseline.get("checkpoint_provenance"),
                                      "candidate": candidate.get("checkpoint_provenance")},
            "policies": {"baseline": base_policy, "candidate": candidate_policy},
            "interpretation": "Measured deltas only; metric direction and safety significance require review."}


def audit_trace(path: Path) -> dict:
    """Summarize existing JSONL control traces without inferring causality."""
    groups = {}
    all_rows = []
    malformed = []
    issues = []

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    for line_number, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            malformed.append({"line": line_number, "error": str(exc)})
            continue
        groups.setdefault(str(row.get("scenario", "unknown")), []).append((line_number, row))
        all_rows.append((line_number, row))

    def summarize(entries):
        stationary = clear_idle = mandated = invalid = safety_blocked = 0
        known_validity = known_clear = speed_steps = 0
        weak_throttle = positive_throttle_requests = brake_request = 0
        stationary_weak_throttle = 0
        steer_mismatch, long_mismatch = [], []
        requested_steering, applied_steering, requested_longitudinal, applied_longitudinal = [], [], [], []
        lanes, headings, speeds, targets, rewards, reasons = [], [], [], [], {}, {}
        accelerations = []
        previous = None
        missing = set()
        def number(row, field, line_number):
            value = row.get(field)
            if value is None:
                missing.add(field)
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append({"line": line_number, "scenario": row.get("scenario"),
                               "field": field, "issue": "expected a finite number"})
                return None
            return float(value)

        for line_number, row in entries:
            speed = number(row, "speed_mps", line_number)
            target = number(row, "target_speed_mps", line_number)
            lane = number(row, "lane_error_m", line_number)
            heading = number(row, "heading_error_rad", line_number)
            timestamp = number(row, "timestamp", line_number)
            valid = row.get("policy_observations_valid")
            validity_field = "policy_observations_valid"
            if valid is None:
                valid = row.get("observations_valid")
                validity_field = "observations_valid"
            sensors = row.get("sensor_valid")
            if valid is not None and not isinstance(valid, bool):
                issues.append({"line": line_number, "scenario": row.get("scenario"),
                               "field": validity_field, "issue": "expected a boolean"})
                valid = None
            sensor_known = isinstance(sensors, dict) and bool(sensors) and all(
                isinstance(value, bool) for value in sensors.values())
            if sensors is not None and not sensor_known:
                issues.append({"line": line_number, "scenario": row.get("scenario"),
                               "field": "sensor_valid", "issue": "expected a nonempty map of booleans"})
            if not isinstance(row.get("reward_components"), dict):
                missing.add("reward_components")
            if valid is None and not sensor_known:
                missing.add("observation_validity")
            known = valid is not None or sensor_known
            sensor_invalid = sensor_known and not all(sensors.values())
            is_invalid = valid is False or sensor_invalid
            if known:
                known_validity += 1
                invalid += int(is_invalid)
            if speed is not None:
                speeds.append(float(speed))
                speed_steps += 1
            if target is not None:
                targets.append(float(target))
            if lane is not None:
                lanes.append(lane)
            if heading is not None:
                headings.append(heading)
            interventions = row.get("interventions")
            if interventions is None:
                missing.add("interventions")
            elif not isinstance(interventions, list):
                issues.append({"line": line_number, "scenario": row.get("scenario"),
                               "field": "interventions", "issue": "expected a list"})
                interventions = []
            for reason in interventions or []:
                if isinstance(reason, str):
                    reasons[reason] = reasons.get(reason, 0) + 1
            safe_stop = (row.get("crossing_phase") == "stopped" or
                         "crosswalk_stop" in (interventions or []) or
                         "red_light_stop" in (interventions or []) or
                         (target is not None and target <= 0.1 and
                          (row.get("red_light") is True or row.get("pedestrian_conflict") is True)))
            if safe_stop:
                mandated += 1
            safety_blocked_step = safe_stop or any(
                reason in (interventions or [])
                for reason in ("pedestrian_conflict_stop", "front_hazard_brake"))
            safety_blocked += int(safety_blocked_step)
            is_stationary = speed is not None and speed < 0.1
            stationary += int(is_stationary)
            clear = known and not is_invalid and not safety_blocked_step
            known_clear += int(clear)
            is_clear_idle = is_stationary and clear and target is not None and target > 0.5
            clear_idle += int(is_clear_idle)
            requested, applied = row.get("requested_action"), row.get("applied_action")
            values = []
            for field, action in (("requested_action", requested), ("applied_action", applied)):
                if action is None:
                    missing.add(field)
                    values.append(None)
                    continue
                if not isinstance(action, (list, tuple)) or len(action) != 2:
                    issues.append({"line": line_number, "scenario": row.get("scenario"),
                                   "field": field, "issue": "expected a finite two-element action"})
                    values.append(None)
                    continue
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                       not math.isfinite(value) for value in action):
                    issues.append({"line": line_number, "scenario": row.get("scenario"),
                                   "field": field, "issue": "contains a non-finite or nonnumeric action value"})
                    values.append(None)
                else:
                    values.append([float(value) for value in action])
            requested, applied = values
            if requested is not None and applied is not None:
                requested_steering.append(requested[0])
                applied_steering.append(applied[0])
                requested_longitudinal.append(requested[1])
                applied_longitudinal.append(applied[1])
                steer_mismatch.append(applied[0] - requested[0])
                long_mismatch.append(applied[1] - requested[1])
                positive_throttle_requests += int(requested[1] > 0)
                is_weak = requested[1] > 0 and 0 < applied[1] < 0.15
                weak_throttle += int(is_weak)
                stationary_weak_throttle += int(is_weak and is_clear_idle)
                brake_request += int(requested[1] < 0)
            components = row.get("reward_components")
            if isinstance(components, dict):
                for name, value in components.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        issues.append({"line": line_number, "scenario": row.get("scenario"),
                                       "field": f"reward_components.{name}", "issue": "expected a finite number"})
                        continue
                    total = rewards.get(name, 0.) + float(value)
                    if math.isfinite(total):
                        rewards[name] = total
                    else:
                        issues.append({"line": line_number, "scenario": row.get("scenario"),
                                       "field": f"reward_components.{name}", "issue": "aggregate overflow"})
            frame = row.get("frame")
            if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int)):
                issues.append({"line": line_number, "scenario": row.get("scenario"),
                               "field": "frame", "issue": "expected an integer frame"})
                frame = None
            if previous and speed is not None and previous["speed"] is not None:
                same_episode = (row.get("scenario") == previous["scenario"] and
                                row.get("episode") == previous["episode"])
                frame_contiguous = (frame is None or previous["frame"] is None or
                                    frame == previous["frame"] + 1)
                dt = timestamp - previous["timestamp"] if timestamp is not None and previous["timestamp"] is not None else None
                time_contiguous = dt is None or 0 < dt <= 1.0
                if same_episode and frame_contiguous and time_contiguous and dt is not None:
                    accelerations.append((speed - previous["speed"]) / dt)
            previous = {"scenario": row.get("scenario"), "episode": row.get("episode"),
                        "frame": frame, "timestamp": timestamp, "speed": speed}
        def stats(values):
            if not values:
                return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
            ordered = sorted(values)
            middle = len(ordered) // 2
            median = ordered[middle] if len(ordered) % 2 else (ordered[middle-1] + ordered[middle]) / 2
            return {"count": len(values), "mean": sum(value / len(values) for value in values), "median": median,
                    "min": ordered[0], "max": ordered[-1]}
        count = len(entries)
        return {"steps": count, "stationary_steps": stationary,
                "stationary_fraction": stationary / max(1, count),
                "clear_road_stationary_steps": clear_idle,
                "clear_road_observation_steps": known_clear,
                "clear_road_stationary_fraction": clear_idle / max(1, known_clear),
                "mandated_stop_steps": mandated, "mandated_stop_fraction": mandated / max(1, count),
                "safety_blocked_steps": safety_blocked,
                "safety_blocked_fraction": safety_blocked / max(1, count),
                "invalid_observation_steps": invalid, "invalid_observation_fraction": invalid / max(1, count),
                "known_observation_validity_steps": known_validity,
                "unknown_observation_validity_steps": count - known_validity,
                "invalid_fraction_of_known_observations": invalid / max(1, known_validity),
                "measured_speed_steps": speed_steps,
                "speed_mps": stats(speeds), "safe_target_speed_mps": stats(targets),
                "lane_error_m": stats(lanes), "heading_error_rad": stats(headings),
                "requested_steering": stats(requested_steering), "applied_steering": stats(applied_steering),
                "applied_minus_requested_steering": stats(steer_mismatch),
                "requested_signed_longitudinal": stats(requested_longitudinal),
                "applied_signed_longitudinal": stats(applied_longitudinal),
                "applied_minus_requested_longitudinal": stats(long_mismatch),
                "weak_positive_throttle_steps": weak_throttle,
                "weak_positive_throttle_fraction_of_requests": weak_throttle / max(1, positive_throttle_requests),
                "clear_road_stationary_weak_positive_throttle_steps": stationary_weak_throttle,
                "positive_throttle_request_steps": positive_throttle_requests,
                "brake_request_steps": brake_request, "supervisor_reasons": reasons,
                "reward_component_totals": rewards,
                "acceleration_mps2": stats(accelerations), "unavailable_fields": sorted(missing)}

    per_scenario = {name: summarize(rows) for name, rows in sorted(groups.items())}
    overall = summarize(all_rows)
    return {"source": str(Path(path)), "steps": len(all_rows), "malformed_lines": malformed, "field_issues": issues,
            "overall": overall, "scenarios": per_scenario,
            "criteria": {"stationary_speed_below_mps": 0.1,
                         "clear_road_target_above_mps": 0.5,
                         "weak_positive_throttle_requested_above": 0.0,
                         "weak_positive_throttle_applied_range": [0.0, 0.15],
                         "acceleration_max_timestamp_gap_s": 1.0,
                         "acceleration_frame_gap": 1},
            "interpretation": "No-override or weak-command evidence does not identify a throttle, controller, or vehicle-physics cause."}

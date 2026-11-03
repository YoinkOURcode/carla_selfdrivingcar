"""Local dashboard and bounded, on-demand destination driving."""

import base64
import io
import json
import math
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from .driving import DrivingConfig, config_default_speed


_REWARD_LABELS = {
    "progress": "Progress", "lane": "Lane keeping", "lane_departure": "Lane departure",
    "collision": "Collision", "offroad": "Off-road", "waypoint": "Waypoint (15 m)",
    "lane_centering": "Lane centering", "pedestrian_collision": "Pedestrian collision",
    "red_light_violation": "Red light", "crossing_violation": "Crossing violation",
    "speeding": "Speeding", "target_tracking": "Target tracking",
    "steering_change": "Steering change", "acceleration_change": "Acceleration change",
    "jerk": "Jerk", "time": "Time", "idle": "Idle", "success": "Success",
    "termination": "Termination",
}


class EpisodeRewards:
    """Accumulate actual reward terms and concise measured episode values."""

    def __init__(self):
        self.components = {}
        self.values = {}
        self.steps = 0
        self.episode_return = 0.0
        self.reward_count = 0

    def reset(self):
        self.components.clear()
        self.values.clear()
        self.steps = 0
        self.episode_return = 0.0
        self.reward_count = 0

    def add(self, record, reward=None, dt_s=None):
        actual_reward = record.get("reward", reward)
        if actual_reward is not None:
            self.episode_return += float(actual_reward)
            self.reward_count += 1
        components = record.get("reward_components")
        if not isinstance(components, dict):
            return
        self.steps += 1
        for name, value in components.items():
            self.components[name] = self.components.get(name, 0.0) + float(value)

        def add_value(name, value):
            if value is not None and math.isfinite(float(value)):
                self.values[name] = self.values.get(name, 0.0) + float(value)

        add_value("waypoint", record.get("waypoints_reached"))
        add_value("progress", record.get("new_progress_m"))
        add_value("lane_centering", record.get("new_progress_m"))
        if record.get("lane_departure") is not None:
            add_value("lane_departure", int(bool(record["lane_departure"])))
        for component, field in (("collision", "collision"), ("offroad", "offroad"),
                                 ("pedestrian_collision", "pedestrian_collision"),
                                 ("red_light_violation", "red_light_violation"),
                                 ("crossing_violation", "crossing_violation"),
                                 ("success", "success")):
            if field in record and record[field] is not None:
                add_value(component, int(bool(record[field])))
        if (record.get("lane_error_m") is not None and
                math.isfinite(float(record["lane_error_m"]))):
            add_value("lane_error_samples", abs(record["lane_error_m"]))
            self.values["lane_error_count"] = self.values.get("lane_error_count", 0) + 1
        add_value("time", dt_s)
        if components.get("idle", 0.0) != 0.0:
            add_value("idle", dt_s)
        speed, limit = record.get("speed_mps"), record.get("speed_limit_mps")
        confidence = record.get("speed_limit_confidence", 1.0)
        if speed is not None and math.isfinite(float(speed)):
            if (limit is None or not math.isfinite(float(limit)) or float(limit) <= 0 or
                    confidence is None or not math.isfinite(float(confidence)) or
                    float(confidence) < DrivingConfig().min_speed_limit_confidence):
                limit = config_default_speed()
            excess = max(0.0, float(speed) - float(limit))
            self.values["speeding_max"] = max(self.values.get("speeding_max", 0.0), excess)
        target = record.get("target_speed_mps")
        if (speed is not None and target is not None and math.isfinite(float(speed)) and
                math.isfinite(float(target))):
            add_value("target_error", abs(float(speed) - float(target)))
            self.values["target_error_count"] = self.values.get("target_error_count", 0) + 1

    def snapshot(self):
        if not self.steps:
            return {"reward_breakdown": [],
                    "episode_return": self.episode_return if self.reward_count else None}
        rows = []
        for name, reward in self.components.items():
            value = "—"
            if name in ("waypoint", "lane_departure", "collision", "offroad", "pedestrian_collision",
                        "red_light_violation", "crossing_violation", "success"):
                if name in self.values:
                    unit = "checkpoints" if name == "waypoint" else "active steps"
                    value = f"{int(self.values[name])} {unit}"
            elif name == "progress" and "progress" in self.values:
                value = f"{self.values['progress']:.1f} m new"
            elif name == "lane_centering" and "lane_centering" in self.values:
                value = f"{self.values['lane_centering']:.1f} m"
            elif name == "speeding" and "speeding_max" in self.values:
                value = f"max {self.values['speeding_max']:.2f} m/s over limit"
            elif name == "target_tracking" and "target_error" in self.values:
                value = f"mean {self.values['target_error'] / self.values['target_error_count']:.2f} m/s error"
            elif name == "time" and "time" in self.values:
                value = f"{self.values['time']:.1f} s"
            elif name == "idle" and "idle" in self.values:
                value = f"{self.values['idle']:.1f} s"
            elif name == "lane" and "lane_error_samples" in self.values:
                value = f"mean {self.values['lane_error_samples'] / self.values['lane_error_count']:.2f} m offset"
            rows.append({"name": _REWARD_LABELS.get(name, name.replace("_", " ").title()),
                         "value": value, "reward": reward})
        return {"reward_breakdown": rows, "episode_return": self.episode_return}


def camera_objects(observation, attention, camera_matrix, intrinsics, size, max_objects, source):
    """Project policy object centres; projection alone does not prove visibility."""
    objects = []
    width, height = size
    for index in range(4, 4 + max_objects):
        if not observation["mask"][index]:
            continue
        row = observation["tokens"][index]
        if row[8] <= 0 or not np.isfinite(row).all():
            continue
        point = camera_matrix @ np.r_[row[:3] * 100., 1.]
        if point[0] <= 0.1:
            continue
        u = intrinsics["cx"] + intrinsics["fx"] * point[1] / point[0]
        v = intrinsics["cy"] - intrinsics["fy"] * point[2] / point[0]
        if not (0 <= u < width and 0 <= v < height):
            continue
        # FusionEncoder prepends the image before the context/object tokens.
        weight = float(attention[index + 1]) if attention is not None else None
        label = "pedestrian" if row[6] > 0.5 else "vehicle" if source == "ground_truth" else "object"
        objects.append({"slot": index - 4, "label": label, "x": float(u / width), "y": float(v / height),
                        "distance_m": float(np.linalg.norm(row[:3] * 100.)),
                        "confidence": float(row[10]), "attention": weight})
    return objects


def policy_objects(observation, attention, max_objects, source, projected, camera_available):
    """Describe valid policy object slots, including objects outside the image."""
    projected_slots = {obj["slot"] for obj in projected}
    rows = []
    for index in range(4, 4 + max_objects):
        if not observation["mask"][index]:
            continue
        row = observation["tokens"][index]
        if not np.isfinite(row).all():
            continue
        slot = index - 4
        rows.append({
            "slot": slot,
            "label": "pedestrian" if row[6] > .5 else "vehicle" if source == "ground_truth" else "object",
            "distance_m": float(np.linalg.norm(row[:3] * 100.)) if row[8] > 0 else None,
            "confidence": float(row[10]),
            "attention": float(attention[index + 1]) if attention is not None else None,
            "camera": ("unavailable" if not camera_available else
                       "in frame" if slot in projected_slots else "outside frame"),
        })
    return rows[:20], len(rows)


def radar_readings(observation, attention, max_objects):
    """Return valid radar tokens without assigning object identities."""
    start = 4 + max_objects
    readings = []
    for index in range(start, min(start + 32, len(observation["mask"]))):
        if not observation["mask"][index]:
            continue
        row = observation["tokens"][index]
        if not np.isfinite(row).all():
            continue
        readings.append({"distance_m": float(np.linalg.norm(row[:3]) * 100.),
                         "attention": float(attention[index + 1]) if attention is not None else None})
    return readings


def traffic_signal(observation):
    """Describe a traffic-signal state actually present in policy tokens."""
    if not observation["mask"][0] or not observation["mask"][2]:
        return None
    context, nav = observation["tokens"][0], observation["tokens"][2]
    if context[12] > .5:
        return {"state": "red", "distance_m": float(nav[3] * 100.) if nav[4] > .5 else None}
    if nav[5] > .5:
        return {"state": "yellow", "distance_m": float(nav[6] * 100.) if nav[6] >= 0 else None}
    return None


def snapshot(env, observation, attention, distance, info, mode):
    bundle = env.last_bundle
    image = None
    objects = []
    if bundle.rgb is not None and bundle.valid.get("rgb", False):
        buffer = io.BytesIO()
        Image.fromarray(bundle.rgb).save(buffer, format="JPEG", quality=80)
        image = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        calibration = bundle.calibration["sensors"]["rgb"]
        camera_matrix = (np.asarray(env.rig.sensors[0].get_transform().get_inverse_matrix()) @
                         np.asarray(env.ego.get_transform().get_matrix()))
        objects = camera_objects(observation, attention, camera_matrix, calibration["intrinsics_px"],
                                 calibration["resolution_px"], env.cfg.max_objects, env.cfg.perception)
    policy_rows, policy_count = policy_objects(observation, attention, env.cfg.max_objects,
                                                env.cfg.perception, objects, image is not None)
    control = env.ego.get_control()
    limit = env.state.speed_limit_mps
    return {"frame": bundle.frame, "simulation_s": bundle.timestamp, "image": image, "objects": objects,
            "policy_objects": policy_rows, "policy_object_count": policy_count,
            "radar_readings": radar_readings(observation, attention, env.cfg.max_objects),
            "traffic_signal": traffic_signal(observation),
            "mode": mode, "perception": env.cfg.perception, "attention_enabled": attention is not None,
            "speed_kmh": env.state.speed_mps * 3.6,
            "speed_limit_kmh": limit * 3.6 if limit is not None else None,
            "distance_m": distance, "route_length_m": env.route.length,
            "remaining_m": max(0., env.route.length - env.state.route_progress_m),
            "progress": float(np.clip(env.state.route_progress_m / env.route.length, 0, 1)),
            "controls": {"steer": control.steer, "throttle": control.throttle, "brake": control.brake},
            "interventions": list(info.get("interventions", [])),
            "termination_reason": info.get("termination_reason"),
            "truncation_reason": info.get("truncation_reason"),
            "lane_error_m": info.get("lane_error_m"),
            "safety_events": [name for name in ("collision", "pedestrian_collision", "offroad",
                                               "lane_departure", "red_light_violation", "crossing_violation")
                              if info.get(name)] + (["lane_deviation"]
                                  if abs(info.get("lane_error_m") or 0.) > 6. else []),
            "sensors_valid": dict(bundle.valid), "success": bool(info.get("success", False))}


class Viewer:
    def __init__(self, cfg, args):
        from .live import TrainingMonitor

        self.cfg, self.args = cfg, args
        self.training = TrainingMonitor(getattr(args, "training_run", None))
        self.rewards = EpisodeRewards()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = {"status": "idle", "message": "Choose a route and start driving.",
                      "route_start": cfg.route_start, "route_end": cfg.route_end,
                      "checkpoint_available": bool(args.checkpoint),
                      "checkpoint": args.checkpoint, "budget_seconds": args.budget_seconds}

    def update(self, **values):
        with self.lock:
            self.state = {**self.state, **values}

    def read(self):
        with self.lock:
            return self.state.copy()

    def start(self, request):
        budget = self.args.budget_seconds
        if budget is None or not math.isfinite(budget) or not 0 < budget <= 86400:
            raise ValueError("Launch viewer with --budget-seconds SECONDS in (0, 86400] to enable driving")
        start, end = request.get("start", self.cfg.route_start), request.get("destination", self.cfg.route_end)
        if type(start) is not int or type(end) is not int or min(start, end) < 0 or start == end:
            raise ValueError("Start and destination must be distinct nonnegative spawn indices")
        if not self.args.checkpoint:
            raise ValueError("Trained Dreamer driving requires --checkpoint PATH")
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("A drive is already active; stop it before starting another")
            self.stop_event.clear()
            self.state = {"status": "starting", "message": "Connecting to CARLA...",
                          "route_start": start, "route_end": end,
                          "checkpoint_available": bool(self.args.checkpoint),
                          "checkpoint": self.args.checkpoint, "budget_seconds": budget}
            self.thread = threading.Thread(target=self.drive, args=(start, end), daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join()

    def drive(self, start, end):
        from .__main__ import prepare_output, server
        from .environment import CarlaEnv
        from .evaluation import collection_scenarios
        from .training import Trainer, register_run, seed_all, workload_lock

        cfg = replace(self.cfg, device="cpu", route_start=start, route_end=end,
                      max_episode_steps=max(self.cfg.max_episode_steps, math.ceil(self.args.budget_seconds / self.cfg.dt)))
        output = None
        started = time.monotonic()
        deadline = started + self.args.budget_seconds
        try:
            scenario = None
            if cfg.demo:
                scenario = dict(cfg.demo)
                scenario.update(name=f"{cfg.town.lower()}_demonstration", weather=cfg.weather,
                                route_start=start, route_end=end, fixed_route=True,
                                background_traffic=False, requires_traffic_light=True,
                                category="curved")
                self.update(scenario=scenario["name"], message="Validating the selected demonstration route.")
            elif cfg.scenario != "basic" or cfg.scenarios:
                selected = collection_scenarios(cfg)
                if len(selected) != 1:
                    raise ValueError("Viewer driving requires exactly one configured scenario")
                scenario = selected[0]
                cfg = replace(cfg, seed=scenario["seed"], route_start=scenario["route_start"],
                              route_end=scenario["route_end"])
                self.update(scenario=scenario["name"], route_start=cfg.route_start, route_end=cfg.route_end,
                            message="Using the configured scenario route.")
            with workload_lock(cfg, launch_server=bool(self.args.ue4_root)):
                candidate = Path(self.args.output or (
                    "runs/dreamerv3_viewer_" + time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns()}"))
                prepare_output(candidate)
                output = candidate
                register_run(output, cfg, "viewer_dreamer", self.args.budget_seconds)
                cfg.save(output / "config.json")
                seed_all(cfg.seed)
                trainer = Trainer.restore(self.args.checkpoint, device="cpu")
                if (trainer.cfg.image_size, trainer.cfg.max_objects, trainer.cfg.attention) != (
                        cfg.image_size, cfg.max_objects, cfg.attention):
                    raise ValueError("Viewer image_size, max_objects and attention must match checkpoint config")
                agent = trainer.agent
                with server(cfg, self.args.ue4_root, deadline):
                    if self.stop_event.is_set() or time.monotonic() >= deadline:
                        self.update(status="stopped", message="Stopped before driving.")
                        return
                    with CarlaEnv(cfg, scenario=scenario) as env:
                        try:
                            self._episode(env, agent, deadline)
                        finally:
                            if hasattr(env, "ego") and env.ego.is_alive:
                                env.ego.apply_control(env.carla.VehicleControl(brake=1.))
        except Exception as error:
            self.update(status="error", message=f"{type(error).__name__}: {error}")
        finally:
            if output is not None:
                # Store final telemetry only; original recordings/checkpoints stay untouched.
                result = {k: v for k, v in self.read().items() if k != "image"}
                result.update(elapsed_s=time.monotonic() - started, checkpoint=self.args.checkpoint)
                (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def _episode(self, env, agent, deadline):
        self.rewards.reset()
        observation = env.reset()
        self.update(route_start=getattr(env, "route_start", env.cfg.route_start),
                    route_end=getattr(env, "route_end", env.cfg.route_end))
        carry = agent.initial()
        previous = env.ego.get_location()
        previous_xyz = np.asarray([previous.x, previous.y, previous.z])
        distance, info, done, truncated = 0., {}, False, False
        self.update(**self.rewards.snapshot())
        while True:
            tick_started = time.monotonic()
            if self.stop_event.is_set() or tick_started >= deadline:
                self.update(status="stopped", message="Stopped." if self.stop_event.is_set() else "Time budget reached.")
                break
            attention = None
            action, next_carry = agent.act(observation, carry, deterministic=True)
            weights = agent.encoder.last_attention
            attention = weights.cpu().numpy().reshape(-1) if weights is not None else None
            self.update(**snapshot(env, observation, attention, distance, info, "dreamer"))
            if done or truncated:
                status = "arrived" if info.get("success") else "ended"
                ending = info.get("termination_reason") or info.get("truncation_reason")
                reasons = {"lane_deviation": "Lane deviation exceeded 6 m.",
                           "collision": "Collision detected.",
                           "pedestrian_collision": "Pedestrian collision detected.",
                           "offroad": "Vehicle went off-road.",
                           "red_light_violation": "Red-light violation.",
                           "crossing_violation": "Crossing violation.",
                           "reward_limit": "Episode return fell below -500.",
                           "time_limit": "Episode step limit reached.",
                           "inactivity_timeout": "No route progress: inactivity timeout.",
                           "blocked_timeout": "Vehicle remained blocked: timeout."}
                reason = ("Destination reached." if info.get("success") else
                          reasons.get(ending, f"Episode ended: {ending or 'reason unavailable'}."))
                self.update(status=status, message=reason)
                break
            self.update(status="driving", message="Driving to destination.")
            observation, reward, done, truncated, info = env.step(action)
            self.rewards.add({**info, "reward": reward}, dt_s=env.cfg.dt)
            self.update(**self.rewards.snapshot())
            import torch

            carry = next_carry
            carry["previous_action"] = torch.as_tensor(info["applied_action"], device=agent.device)
            carry["is_init"] = torch.zeros(1, dtype=torch.bool, device=agent.device)
            position = env.ego.get_location()
            xyz = np.asarray([position.x, position.y, position.z])
            distance += float(np.linalg.norm(xyz - previous_xyz))
            previous_xyz = xyz
            self.stop_event.wait(max(0., env.cfg.dt - (time.monotonic() - tick_started)))


def make_server(viewer, port=8080):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path == "/":
                self.reply(200, Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(200, json.dumps(viewer.read(), allow_nan=False).encode())
            elif self.path == "/api/training":
                self.reply(200, json.dumps(viewer.training.read(), allow_nan=False).encode())
            else:
                self.reply(404, b'{"error":"Not found"}')

        def do_POST(self):
            authority = self.headers.get("Host", "")
            allowed = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            if authority not in allowed or self.headers.get("Origin", f"http://{authority}") != f"http://{authority}":
                self.reply(403, b'{"error":"Local same-origin requests only"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON request")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/start":
                    viewer.start(request)
                elif self.path == "/api/stop":
                    viewer.stop()
                else:
                    self.reply(404, b'{"error":"Not found"}')
                    return
                self.reply(200, b'{"ok":true}')
            except (ValueError, json.JSONDecodeError) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode())

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(cfg, args):
    if not 1 <= args.web_port <= 65535:
        raise ValueError("--web-port must be between 1 and 65535")
    viewer = Viewer(cfg, args)
    with make_server(viewer, args.web_port) as http:
        print(f"CARLA viewer: http://localhost:{http.server_port} (Ctrl+C to stop)", flush=True)
        try:
            http.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()

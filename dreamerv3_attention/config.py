"""Small, strict experiment configuration; all distances use metres."""

import json
import math
from dataclasses import InitVar, asdict, dataclass, field, replace
from pathlib import Path


@dataclass
class Config:
    seed: int = 7
    attention: bool = True
    device: str = "cpu"
    image_size: int = 64
    max_objects: int = 16
    hidden: int = 128
    belief: int = 128
    categoricals: int = 16
    classes: int = 16
    image_depth: int = 8
    bins: int = 255
    horizon: int = 15
    gamma: float = 0.997
    batch: int = 4
    sequence: int = 16
    capacity: int = 5000
    learning_rate: float = 4e-5
    warmup_updates: int = 100
    updates_per_step: float = 0.25
    warmup_frames: int = 128
    warmup_policy: str = "random"
    world_model_warmup_updates: int = 0
    actor_bc_weight: float = 0.0
    expert_episode_interval: int = 0
    actor_steering_bc_weight: float = 0.0
    actor_longitudinal_bc_weight: float = 0.0
    # Probability that a replay sequence gets weather-like image corruption in the
    # encoder input (reconstruction target stays clean). 0 disables it.
    pixel_augment: float = 0.0
    # False hides the camera image from the agent (tokens only) and drops the pixel reconstruction loss.
    use_pixels: bool = True
    # Weight of an extra world-model loss on the reconstructed near route point.
    route_aux_weight: float = 0.0
    steps: int = 10000
    max_episode_steps: int = 1000
    inactivity_timeout_s: float = 15.0
    inactivity_min_progress_m: float = 0.5
    blocked_timeout_s: float = 60.0
    inactivity_grace_s: float = 2.0
    inactivity_stationary_s: float = 7.0
    checkpoint_every: int = 500
    host: str = "127.0.0.1"
    port: int = 2000
    town: str = "Town01"
    dt: float = 0.05
    timeout: float = 10.0
    sensor_timeout: float = 0.5
    # Retired timing overrides are accepted and ignored when loading old run configs.
    traffic_light_red_seconds: InitVar[float | None] = None
    traffic_light_yellow_seconds: InitVar[float | None] = None
    route_start: int = 0
    route_end: int = 10
    weather: str = "ClearNoon"
    perception: str = "ground_truth"
    provider: str = ""
    mixed_fields: list[str] = field(default_factory=list)
    drop_sensors: list[str] = field(default_factory=list)
    sensor_noise_std: float = 0.0
    camera: dict = field(default_factory=dict)
    radar: dict = field(default_factory=dict)
    driving: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)
    scenario_file: str = ""
    demo: dict = field(default_factory=dict)
    viewer_checkpoint: str = ""
    scenario: str = "basic"
    scenarios: list[str] = field(default_factory=list)
    randomize_routes: bool = True
    split: str = "train"

    def validate(self):
        if not isinstance(self.demo, dict) or not isinstance(self.viewer_checkpoint, str):
            raise ValueError("demo must be an object and viewer_checkpoint must be a path string")
        if self.demo:
            encounters = self.demo.get("encounters")
            length = self.demo.get("minimum_route_length_m")
            if (not isinstance(length, (int, float)) or isinstance(length, bool) or
                    not math.isfinite(length) or length < 50 or not isinstance(encounters, list) or not encounters):
                raise ValueError("demo requires a minimum route length and encounters")
            previous = -1.
            for event in encounters:
                if not isinstance(event, dict):
                    raise ValueError("demo encounters must be objects")
                trigger = event.get("trigger_m")
                if (event.get("kind") not in ("vehicle", "adult", "child") or
                        not isinstance(trigger, (int, float)) or isinstance(trigger, bool) or
                        not math.isfinite(trigger) or trigger <= previous or trigger + 35 > length):
                    raise ValueError("demo encounters need ordered route triggers and valid actor kinds")
                previous = trigger
        if not isinstance(self.randomize_routes, bool):
            raise ValueError("randomize_routes must be a boolean")
        if self.warmup_policy not in ("random", "controller"):
            raise ValueError("warmup_policy must be random or controller")
        if not isinstance(self.world_model_warmup_updates, int) or self.world_model_warmup_updates < 0:
            raise ValueError("world_model_warmup_updates must be a nonnegative integer")
        if not math.isfinite(self.actor_bc_weight) or self.actor_bc_weight < 0:
            raise ValueError("actor_bc_weight must be finite and nonnegative")
        if not math.isfinite(self.actor_steering_bc_weight) or self.actor_steering_bc_weight < 0:
            raise ValueError("actor_steering_bc_weight must be finite and nonnegative")
        if not math.isfinite(self.actor_longitudinal_bc_weight) or self.actor_longitudinal_bc_weight < 0:
            raise ValueError("actor_longitudinal_bc_weight must be finite and nonnegative")
        if not math.isfinite(self.pixel_augment) or not 0 <= self.pixel_augment <= 1:
            raise ValueError("pixel_augment must be a probability")
        if not isinstance(self.use_pixels, bool):
            raise ValueError("use_pixels must be a boolean")
        if not math.isfinite(self.route_aux_weight) or self.route_aux_weight < 0:
            raise ValueError("route_aux_weight must be finite and nonnegative")
        if self.actor_bc_weight and self.warmup_policy != "controller":
            raise ValueError("actor_bc_weight requires controller warmup")
        if not isinstance(self.expert_episode_interval, int) or self.expert_episode_interval < 0:
            raise ValueError("expert_episode_interval must be a nonnegative integer")
        if self.expert_episode_interval and not self.actor_bc_weight:
            raise ValueError("Expert refresh requires actor_bc_weight")
        if (not isinstance(self.scenarios, list) or
                any(not isinstance(name, str) or not name for name in self.scenarios) or
                len(set(self.scenarios)) != len(self.scenarios)):
            raise ValueError("scenarios must be a list of unique nonempty scenario names")
        if self.scenarios and self.scenario != "basic":
            raise ValueError("Use scenarios or scenario, not both")
        for name in ("image_size", "hidden", "belief", "categoricals", "classes", "image_depth", "bins",
                     "batch", "sequence", "capacity", "steps", "horizon", "max_episode_steps", "checkpoint_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_size % 8 or self.hidden < 4 or self.bins < 3 or self.max_objects < 0:
            raise ValueError("image_size must be divisible by 8; hidden >= 4, bins >= 3, max_objects >= 0")
        if self.capacity < self.sequence or self.max_episode_steps < self.sequence:
            raise ValueError("capacity and episode length must accommodate sequence")
        if not 0 < self.dt <= 0.1 or not 0 < self.gamma < 1 or self.learning_rate <= 0:
            raise ValueError("invalid physics timestep, discount or learning rate")
        if self.updates_per_step < 0 or self.warmup_updates < 0 or self.warmup_frames < self.sequence:
            raise ValueError("invalid update schedule")
        if self.perception not in ("ground_truth", "external", "mixed"):
            raise ValueError("unknown perception mode")
        if self.perception != "ground_truth" and not self.provider:
            raise ValueError("external/mixed perception requires module:factory provider")
        if self.perception == "mixed" and not self.mixed_fields:
            raise ValueError("mixed mode requires explicit mixed_fields provenance")
        if self.split not in ("train", "validation", "held_out") or self.sensor_noise_std < 0:
            raise ValueError("invalid split or sensor noise")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if not 1 <= self.port <= 65533 or self.sensor_timeout <= 0 or self.timeout <= 0:
            raise ValueError("invalid port or sensor/network timeout")
        from .driving import DrivingConfig, RewardConfig
        from .inactivity import InactivityWatchdog
        from .sensors import CameraConfig, RadarConfig

        InactivityWatchdog(self.inactivity_timeout_s, self.inactivity_min_progress_m,
                           self.blocked_timeout_s, self.inactivity_grace_s, self.inactivity_stationary_s)
        CameraConfig(**self.camera)
        RadarConfig(**self.radar)
        driving = DrivingConfig(**self.driving)
        for name in ("pedestrian_speed_factor", "child_pedestrian_speed_factor", "vehicle_speed_factor"):
            factor = getattr(driving, name)
            if (isinstance(factor, bool) or not isinstance(factor, (int, float)) or
                    not math.isfinite(factor) or not 0. < factor <= 1.):
                raise ValueError(f"driving.{name} must be finite and in (0, 1]")
        if not isinstance(driving.restart_assist_enabled, bool):
            raise ValueError("driving.restart_assist_enabled must be a boolean")
        restart_limits = (driving.restart_stop_distance_m, driving.restart_stop_confirm_s,
                          driving.restart_green_grace_s, driving.restart_target_speed_mps,
                          driving.restart_max_throttle, driving.restart_timeout_s,
                          driving.restart_handover_s)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value <= 0
               for value in restart_limits):
            raise ValueError("Restart-assist limits must be positive finite numbers")
        if driving.restart_max_throttle > 1.0:
            raise ValueError("restart_max_throttle must be at most 1")
        for settings in (asdict(driving), asdict(RewardConfig(**self.reward))):
            if any(not math.isfinite(value) or value < 0 for value in settings.values()):
                raise ValueError("Driving/reward configuration values must be finite and nonnegative")
        red_deceleration = driving.red_light_deceleration_mps2
        if isinstance(red_deceleration, bool) or red_deceleration <= 0:
            raise ValueError("Red-light deceleration must be positive and finite")
        if isinstance(driving.max_speed_mps, bool):
            raise ValueError("max_speed_mps must be finite and nonnegative; 0 disables the cap")
        return self

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self.resolved()), indent=2) + "\n")

    def resolved(self):
        """Freeze defaults that affect simulator observations and rewards."""
        from .driving import DrivingConfig, RewardConfig
        from .sensors import CameraConfig, RadarConfig

        # Checkpoints saved while reverse recovery existed still carry its keys.
        driving = {k: v for k, v in self.driving.items() if not k.startswith("reverse_")}
        return replace(self, driving=asdict(DrivingConfig(**driving)),
                       reward=asdict(RewardConfig(**self.reward)),
                       camera=asdict(CameraConfig(**{"sensor_tick": self.dt, **self.camera})),
                       radar=asdict(RadarConfig(**{"sensor_tick": self.dt, **self.radar})))

"""Small, explicit configuration. Versions and values are stored without hashes."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EnvironmentConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    timeout: float = 30.0
    town: str = "Town01"
    ego_model: str = "vehicle.tesla.model3"
    vehicles: int = 20
    pedestrians: int = 20
    child_fraction: float = 0.35
    weather: list = field(default_factory=lambda: [
        "ClearNoon", "HardRainNoon", "ClearNight", "HardRainNight", "FogNight"])
    fixed_delta_seconds: float = 0.05
    action_repeat: int = 2
    max_steps: int = 1800
    stuck_seconds: float = 60.0
    offroad_seconds: float = 2.0
    start_spawn: int | None = None
    goal_spawn: int | None = None
    min_route_metres: float = 80.0
    max_route_metres: float = 600.0
    visibility_metres: float = 70.0
    field_of_view_degrees: float = 100.0
    child_caution_metres: float = 30.0
    child_lateral_metres: float = 10.0
    child_speed_mps: float = 4.17
    child_reaction_seconds: float = 1.0
    caution_deceleration: float = 2.5
    rain_speed_factor: float = 0.7
    night_speed_factor: float = 0.7
    fog_speed_factor: float = 0.5
    steering_rate: float = 2.0
    acceleration_command_rate: float = 2.5
    braking_command_rate: float = 4.0
    sensor_grace_seconds: float = 0.02
    observation_provider: str = "ground_truth"
    no_rendering_mode: bool = False

    @property
    def decision_dt(self):
        return self.fixed_delta_seconds * self.action_repeat


@dataclass
class RewardConfig:
    progress: float = 2.0
    success: float = 100.0
    collision: float = 100.0
    red_light: float = 20.0
    stop_sign: float = 10.0
    illegal_lane: float = 3.0
    offroad: float = 10.0
    wrong_way: float = 5.0
    speeding: float = 0.1
    unsafe_child: float = 1.0
    unsafe_weather: float = 0.5
    acceleration: float = 0.1
    jerk: float = 0.1
    steering_rate: float = 0.05
    time: float = 0.05
    comfort_filter_seconds: float = 0.3


@dataclass
class ModelConfig:
    embedding: int = 64
    attention_heads: int = 4
    deterministic: int = 128
    stochastic: int = 32
    hidden: int = 256
    learning_rate: float = 0.0003
    batch_size: int = 32
    sequence_length: int = 20
    burn_in: int = 5
    kl_weight: float = 0.1
    free_nats: float = 1.0
    prior_prediction_weight: float = 0.5
    gradient_clip: float = 100.0


@dataclass
class TrainingConfig:
    seed: int = 20261016
    device: str = "auto"
    bootstrap_episodes: int = 20
    pretrain_updates: int = 10000
    policy_updates: int = 100000
    policy_block_updates: int = 10000
    collection_episodes: int = 2
    refresh_updates: int = 200
    replay_steps: int = 100000
    batch_size: int = 256
    minimum_real_steps: int = 500
    learning_rate: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.005
    initial_alpha: float = 0.1
    imagination_horizon: int = 1
    imagination_starts: int = 1024
    synthetic_fraction: float = 0.25
    validation_branches: int = 128
    validation_episodes: int = 10
    encoding_batch_episodes: int = 16
    imagination_refresh_updates: int = 100
    checkpoint_updates: int = 1000
    failure_sequence_fraction: float = 0.5
    termination_fit_updates: int = 2000
    gate_normalized_rmse: float = 0.75
    gate_speed_mae: float = 2.0
    gate_lane_mae: float = 0.5
    gate_heading_mae: float = 0.2
    gate_reward_mae: float = 10.0
    gate_terminal_brier: float = 0.2
    gate_terminal_recall: float = 0.5
    gate_terminal_precision: float = 0.5
    gate_min_terminal_positive: int = 1
    gate_min_terminal_negative: int = 20
    behavior_cloning_updates: int = 0
    behavior_cloning_weight: float = 0.0
    behavior_cloning_teacher: str = "recorded"
    behavior_cloning_traffic_fraction: float = 0.0
    policy_observation_skip: bool = False


@dataclass
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        kinds = {"environment": EnvironmentConfig, "reward": RewardConfig,
                 "model": ModelConfig, "training": TrainingConfig}
        unknown = value.keys() - kinds.keys()
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        cfg = cls(**{name: kind(**value.get(name, {})) for name, kind in kinds.items()})
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path=None):
        return cls.from_dict(json.loads(Path(path).read_text()) if path else {})

    def validate(self):
        e, m, t = self.environment, self.model, self.training
        if not 0 < e.fixed_delta_seconds <= 0.1 or e.action_repeat < 1:
            raise ValueError("Physics dt must be in (0, 0.1] and action_repeat >= 1")
        if (e.start_spawn is None) != (e.goal_spawn is None):
            raise ValueError("Specify both start_spawn and goal_spawn, or neither")
        if not e.weather or e.vehicles < 0 or e.pedestrians < 0:
            raise ValueError("Need weather presets and nonnegative actor counts")
        if type(e.no_rendering_mode) is not bool or (e.no_rendering_mode and e.observation_provider != "ground_truth"):
            raise ValueError("No-rendering mode requires a ground-truth observation provider")
        if e.pedestrians == 1 or not 0 < e.child_fraction < 1:
            raise ValueError("Use zero or at least two pedestrians; child_fraction must be in (0, 1)")
        if not 0 < e.min_route_metres <= e.max_route_metres or e.max_steps < 1:
            raise ValueError("Invalid route lengths or max_steps")
        for value in (e.rain_speed_factor, e.night_speed_factor, e.fog_speed_factor):
            if not 0 < value <= 1:
                raise ValueError("Weather speed factors must be in (0, 1]")
        if min(e.steering_rate, e.acceleration_command_rate, e.braking_command_rate,
               e.visibility_metres, e.child_caution_metres, e.caution_deceleration,
               e.stuck_seconds, e.offroad_seconds, e.child_speed_mps, e.child_lateral_metres) <= 0:
            raise ValueError("Action rates, visibility and caution settings must be positive")
        if not 0 < e.field_of_view_degrees < 180 or min(e.sensor_grace_seconds, e.child_reaction_seconds) < 0:
            raise ValueError("Invalid field of view or sensor grace interval")
        if min(m.embedding, m.attention_heads, m.deterministic, m.stochastic, m.hidden) < 1 or m.embedding % m.attention_heads:
            raise ValueError("Attention heads must divide embedding; model sizes must be positive")
        if min(m.sequence_length, m.batch_size, t.batch_size, t.replay_steps,
               t.policy_block_updates, t.imagination_starts, t.validation_branches,
               t.validation_episodes, t.encoding_batch_episodes, t.imagination_refresh_updates,
               t.checkpoint_updates, t.termination_fit_updates) < 1:
            raise ValueError("Lengths, batch sizes and refresh interval must be positive")
        if m.burn_in < 0 or t.imagination_horizon not in (1, 3, 5):
            raise ValueError("Nonnegative burn-in; supported imagination horizons: 1, 3, 5")
        if not 0 < t.synthetic_fraction <= 0.5 or int(t.batch_size * t.synthetic_fraction) < 1 or not 0 < t.gamma < 1 or not 0 < t.tau <= 1:
            raise ValueError("Invalid synthetic fraction, gamma or tau")
        if min(t.pretrain_updates, t.refresh_updates, t.bootstrap_episodes,
               t.minimum_real_steps, t.policy_updates, t.collection_episodes) < 0:
            raise ValueError("Training budgets must be nonnegative")
        if not 0 <= t.failure_sequence_fraction <= 1:
            raise ValueError("failure_sequence_fraction must be in [0, 1]")
        if t.behavior_cloning_updates < 0 or t.behavior_cloning_weight < 0:
            raise ValueError("Behavior cloning budgets and weight must be nonnegative")
        if not 0 <= t.behavior_cloning_traffic_fraction <= 1:
            raise ValueError("behavior_cloning_traffic_fraction must be in [0, 1]")
        if t.behavior_cloning_teacher not in ("recorded", "geometry_v1", "geometry_v2", "geometry_v3"):
            raise ValueError("Unknown behavior cloning teacher")
        if t.behavior_cloning_teacher != "recorded" and e.ego_model != "vehicle.tesla.model3":
            raise ValueError("The geometric teacher is calibrated for the Tesla Model3 ego")
        if min(self.reward.comfort_filter_seconds, m.gradient_clip) <= 0:
            raise ValueError("Comfort filter duration and gradient clip must be positive")
        if min(m.learning_rate, t.learning_rate, t.initial_alpha) <= 0:
            raise ValueError("Learning rates and initial alpha must be positive")
        if min(vars(self.reward).values()) < 0:
            raise ValueError("Reward magnitudes must be nonnegative")


def require_compatible(saved, current):
    """Prevent silently mixing different simulator dynamics or reward labels."""
    for name in ("environment", "reward", "model"):
        before, after = dict(saved[name]), dict(current.to_dict()[name])
        if name == "environment":
            # Viewport rendering does not change physics or the structured oracle.
            # It is a runtime throughput option; camera providers cannot disable it.
            before.pop("no_rendering_mode", None)
            after.pop("no_rendering_mode", None)
        if before != after:
            raise ValueError(f"Checkpoint/dataset {name} differs from configuration; use its saved config")

"""Decision-rate task rewards and bounded comfort costs."""

import math

from .schema import EVENTS

REWARD_VERSION = "decision-comfort-v2"


class ComfortCost:
    """EMA longitudinal acceleration; each comfort term has a per-second ceiling."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.filtered = None
        self.acceleration = self.jerk = 0.0

    def step(self, before_acceleration, after_acceleration, applied, previous, dt, collision=False):
        w = self.cfg.reward
        if self.filtered is None:
            self.filtered = float(before_acceleration)
        old = self.filtered
        self.filtered += (1 - math.exp(-dt / w.comfort_filter_seconds)) * (float(after_acceleration) - old)
        self.acceleration = abs(self.filtered)
        self.jerk = abs(self.filtered - old) / dt
        if collision:
            self.filtered = None
            return dict(acceleration=0.0, jerk=0.0, steering_rate=0.0)
        steering = abs(float(applied[0] - previous[0])) / dt
        bounded = lambda value, threshold: min(1.0, max(0.0, value / threshold - 1.0) ** 2)
        return {"acceleration": -w.acceleration * dt * bounded(self.acceleration, 3.0),
                "jerk": -w.jerk * dt * bounded(self.jerk, 5.0),
                "steering_rate": -w.steering_rate * dt * min(1.0, (steering / self.cfg.environment.steering_rate) ** 2)}


class RewardCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.child_seconds = None
        self.child_initial_speed = 0.0
        self.comfort = ComfortCost(cfg)

    def step(self, before, after, incoming, dt, applied, previous_applied):
        """Evaluate one recorded decision interval."""
        w, e = self.cfg.reward, self.cfg.environment
        events = {name: float(incoming.get(name, 0)) for name in EVENTS}
        if after["child_near"]:
            if self.child_seconds is None:
                self.child_seconds, self.child_initial_speed = 0.0, before["speed"]
            self.child_seconds += dt
            envelope = max(e.child_speed_mps, self.child_initial_speed - e.caution_deceleration *
                           max(0, self.child_seconds - e.child_reaction_seconds))
            child_excess = max(0, after["speed"] - envelope)
        else:
            self.child_seconds = None
            child_excess = 0.0
        weather_excess = max(0, after["speed"] - after["weather_limit"])
        speeding = max(0, after["speed"] - after["speed_limit"])
        events.update(offroad=float(after["offroad"]), wrong_way=float(after["wrong_way"]),
                      unsafe_child=float(child_excess > 0.1),
                      unsafe_weather=float(weather_excess > 0.1 and after["weather_limit"] < after["speed_limit"]))
        comfort = self.comfort.step(before["longitudinal_acceleration"], after["longitudinal_acceleration"],
                                    applied, previous_applied, dt, bool(events["collision"]))
        components = {
            "progress": w.progress * after["progress_distance"], "success": w.success * events["success"],
            "collision": -w.collision * events["collision"], "red_light": -w.red_light * events["red_light"],
            "stop_sign": -w.stop_sign * events["stop_sign"], "illegal_lane": -w.illegal_lane * events["illegal_lane"],
            "offroad": -w.offroad * dt * events["offroad"], "wrong_way": -w.wrong_way * dt * events["wrong_way"],
            "speeding": -w.speeding * dt * speeding ** 2,
            "unsafe_child": -w.unsafe_child * dt * child_excess ** 2,
            "unsafe_weather": -w.unsafe_weather * dt * weather_excess ** 2 * (
                after["weather_limit"] < after["speed_limit"]),
            **comfort, "time": -w.time * dt}
        return components, events

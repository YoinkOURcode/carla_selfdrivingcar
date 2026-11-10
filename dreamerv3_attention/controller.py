"""A fixed-route diagnostic controller using the policy's packed observations."""

import math
from dataclasses import replace

import numpy as np

from .driving import (DrivingConfig, DrivingState, compute_target_speed, signal_stop_required,
                      signal_target_speed, signal_wait, yellow_can_stop)


class RouteController:
    """Pure pursuit steering and proportional speed control, before safety."""

    def __init__(self, target_speed=None, config=None):
        if target_speed is not None and (isinstance(target_speed, bool) or
                                         not math.isfinite(target_speed) or target_speed <= 0):
            raise ValueError("Controller speed must be positive and finite")
        self.target_speed = target_speed
        self.config = config or DrivingConfig()

    def initial(self):
        return None

    def act(self, observation, carry, deterministic=True):
        try:
            tokens, mask = np.asarray(observation["tokens"]), np.asarray(observation["mask"])
            if (tokens.ndim != 2 or tokens.shape[0] < 3 or tokens.shape[1] < 16 or
                    mask.ndim != 1 or len(mask) < 3):
                raise ValueError
            context, near, nav = tokens[0], tokens[1], tokens[2]
            if not np.all(np.isfinite(context[[0, 1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15]])):
                raise ValueError
            if not np.all(np.isfinite(nav)) or not np.all(np.isfinite(near[:2])):
                raise ValueError
        except (KeyError, TypeError, ValueError, IndexError):
            return np.asarray([0., -1.], dtype=np.float32), None
        # Context is token 0; navigation tokens 1..3 look 5, 15, 30 m ahead.
        if (not mask[:3].all() or context[15] < 0.5 or near[15] != -1 or nav[15] != -1):
            return np.asarray([0., -1.], dtype=np.float32), None
        # The 15 m target cuts across tight junction corners. Track the nearest
        # planned point so steering follows the approach lane into the turn.
        x, y = near[:2] * 100.
        steer_angle = math.atan2(2. * 2.8 * float(y), max(1., float(x*x+y*y)))
        steer = np.clip(steer_angle / 0.7, -1., 1.)
        speed = max(0., float(context[0]) * 30.)
        limit = float(context[1]) * 30.
        state = DrivingState(
            speed_mps=speed,
            speed_limit_mps=limit if limit > 0 else None,
            speed_limit_confidence=float(context[2]),
            rain=float(context[6]), wetness=float(context[7]), night=float(context[8]),
            visibility_m=float(context[9]) * 100.,
            pedestrian_risk=float(context[13]),
            child_pedestrian_risk=float(context[14]),
            red_light=bool(context[12] > 0.5),
            red_light_distance_m=float(nav[3]) * 100. if nav[4] > 0.5 else None,
            yellow_light=bool(nav[5] > 0.5),
            yellow_light_distance_m=float(nav[6]) * 100. if nav[6] >= 0 else None,
            yellow_light_confidence=float(nav[7]))
        state.vehicle_ahead = any(
            valid and np.isfinite(row).all() and row[8] > .5 and row[6] <= .5 and
            row[0] > 0 and abs(row[1]) < .025 and np.linalg.norm(row[:3]) <= .8
            for row, valid in zip(tokens[4:], mask[4:]))
        if state.yellow_light and not 0. < state.yellow_light_confidence <= 1.:
            return np.asarray([0., -1.], dtype=np.float32), None
        yellow_stop = None
        yellow_braking = False
        if state.yellow_light:
            yellow_stop = (carry.get("yellow_stop") if isinstance(carry, dict) else None)
            if yellow_stop is None:
                yellow_stop = yellow_can_stop(state, self.config)
            if not yellow_stop:
                state = replace(state, yellow_light=False)
            else:
                yellow_braking = signal_stop_required(state, state.yellow_light_distance_m, self.config)
        target = compute_target_speed(state, self.config)
        if yellow_stop:
            yellow_limit = signal_target_speed(state.yellow_light_distance_m, self.config)
            target = min(target, max(2., speed * .8), yellow_limit)
        if self.target_speed is not None:
            target = min(target, self.target_speed)
        longitudinal = (-1. if yellow_braking or signal_wait(state, self.config) else
                        np.clip(0.5 * (target - speed), -1., 1.))
        return np.asarray([steer, longitudinal], dtype=np.float32), {"yellow_stop": yellow_stop, "yellow_braking": yellow_braking}

"""Identical requested-to-applied control semantics in CARLA and imagination."""

import numpy as np


# Distances are from the front bumper to the relevant traffic light stopping line.
LIGHT_STOP_MARGIN_M = 2.5
LIGHT_WAIT_DISTANCE_M = 5.0
WAIT_SPEED_MPS = 0.3


def apply_action(requested, previous, cfg):
    requested = np.asarray(requested, np.float32)
    previous = np.asarray(previous, np.float32)
    if requested.shape[-1] != 3 or not np.isfinite(requested).all():
        raise ValueError("Action must have three finite components")
    request = np.clip(requested, [-1, 0, 0], [1, 1, 1])
    dt = cfg.decision_dt
    steer = previous[..., 0] + np.clip(request[..., 0] - previous[..., 0],
                                      -cfg.steering_rate * dt, cfg.steering_rate * dt)
    before = previous[..., 1] - previous[..., 2]
    desired = request[..., 1] - request[..., 2]
    rate = np.where(desired < before, cfg.braking_command_rate, cfg.acceleration_command_rate)
    longitudinal = before + np.clip(desired - before, -rate * dt, rate * dt)
    return np.stack([steer, np.maximum(longitudinal, 0), np.maximum(-longitudinal, 0)], -1).astype(np.float32)


def apply_action_tensor(requested, previous, cfg):
    import torch

    request = torch.cat([requested[..., :1].clamp(-1, 1), requested[..., 1:].clamp(0, 1)], -1)
    dt = cfg.decision_dt
    steer = previous[..., 0] + (request[..., 0] - previous[..., 0]).clamp(
        -cfg.steering_rate * dt, cfg.steering_rate * dt)
    before = previous[..., 1] - previous[..., 2]
    desired = request[..., 1] - request[..., 2]
    rate = torch.where(desired < before, cfg.braking_command_rate, cfg.acceleration_command_rate)
    difference = torch.minimum(torch.maximum(desired - before, -rate * dt), rate * dt)
    longitudinal = before + difference
    return torch.stack([steer, longitudinal.clamp_min(0), (-longitudinal).clamp_min(0)], -1)


def weather_limits(speed_limit, rain, wetness, fog, sun_altitude, cfg):
    """Explicit research heuristics, not a claim about safe real-world stopping speeds."""
    factors = [1.0]
    if max(rain, wetness) >= 20:
        factors.append(cfg.rain_speed_factor)
    if sun_altitude < 0:
        factors.append(cfg.night_speed_factor)
    if fog >= 20:
        factors.append(cfg.fog_speed_factor)
    return speed_limit * min(factors)


def visibility_range(rain, fog, sun_altitude, cfg):
    factor = (1 - 0.4 * rain / 100) * (1 - 0.7 * fog / 100) * (
        0.5 if sun_altitude < 0 else 1.0)
    return max(10.0, cfg.visibility_metres * factor)

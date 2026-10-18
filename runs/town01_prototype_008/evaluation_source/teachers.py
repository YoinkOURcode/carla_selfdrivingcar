"""Observed-state training labels. These functions are never driving-time overrides."""

import numpy as np

from .schema import OBJECT_OFFSET, SLOTS


def geometry_actions(arrays, calibrated=False, hold_stops=False):
    """A conservative pursuit/speed teacher on current structured training states only.

    It supplies counterfactual action labels, not measured actions or dynamics targets.
    Constants describe the configured Tesla-sized ego; no future states/events are read.
    """
    c = arrays["continuous"][:-1]
    speed = np.maximum(c[:, 0], 0)
    points = c[:, 20:OBJECT_OFFSET].reshape(-1, 5, 2)
    lookahead = np.clip(3 + .5 * speed, 5, 10)
    blend = ((lookahead - 5) / 5)[:, None]
    point = points[:, 0] * (1 - blend) + points[:, 1] * blend
    curvature = 2 * point[:, 1] / np.maximum(np.square(point).sum(-1), 1.)
    if calibrated:
        # Measured live Tesla wheelbase: 3.0046 m; front-wheel max steer: 70 deg.
        # CARLA's steering-curve speed is km/h, confirmed in the UE movement code.
        fraction = np.interp(speed * 3.6, [0, 20, 60, 120], [1., .9, .8, .7])
        steering = np.clip(-np.arctan(3.0 * curvature) / (np.deg2rad(70.) * fraction), -.8, .8)
    else:
        steering = np.clip(-np.arctan(2.9 * curvature) / .7, -.8, .8)

    curves = np.abs(2 * points[:, :3, 1] / np.maximum(np.square(points[:, :3]).sum(-1), 1.))
    lateral_acceleration = 1.8 * (1 - .4 * c[:, 16] / 100 - .2 * c[:, 17] / 100)
    curve_speed = np.sqrt(lateral_acceleration / np.maximum(curves.max(-1), .001))
    target = np.minimum(np.maximum(c[:, 12], 0), curve_speed)
    deceleration = 2.5 * (1 - .35 * c[:, 16] / 100 - .15 * c[:, 17] / 100)
    controls = arrays["controls"][:-1]
    light = arrays["light"][:-1]
    red_or_yellow = controls[:, 0] & np.isin(light, [2, 3]) & (c[:, 14] >= 0)
    light_speed = np.sqrt(2 * deceleration * np.maximum(c[:, 14] - 2.5, 0))
    target = np.where(red_or_yellow, np.minimum(target, light_speed), target)

    objects = c[:, OBJECT_OFFSET:].reshape(-1, SLOTS, 7)
    present = arrays["present"][:-1]
    classes = arrays["classes"][:-1]
    for slot in range(SLOTS):
        o = objects[:, slot]
        forward, left = o[:, 0], o[:, 1]
        vehicle = classes[:, slot] == 0
        clearance = forward - o[:, 4] - 2.4
        same_corridor = np.abs(left) < 1.35 + o[:, 5]
        # Pedestrians moving into the corridor are considered before reaching it.
        pedestrian_corridor = np.minimum(np.abs(left), np.abs(left + o[:, 3])) < 1.6 + o[:, 5]
        relevant = present[:, slot] & (forward > 0) & np.where(vehicle, same_corridor, pedestrian_corridor)
        lead_speed = np.maximum(speed + o[:, 2], 0)
        stopping_speed = np.sqrt(2 * deceleration * np.maximum(clearance - 2.5, 0))
        following_speed = np.minimum(lead_speed + stopping_speed, np.maximum(clearance - 2.5, 0) / 1.5)
        permitted = np.where(vehicle, following_speed, stopping_speed)
        target = np.where(relevant, np.minimum(target, permitted), target)

    error = target - speed
    braking = error < -.2
    throttle = np.where(braking, 0, np.clip(.25 * error + .12, 0, .75))
    brake = np.where(braking, np.clip(-.35 * error, 0, .8), 0)
    if hold_stops:
        # A speed servo's cruise feed-forward must not creep through a stopped lead
        # or red light once measured speed falls below its ordinary braking deadband.
        stopped_target = target <= .3
        throttle = np.where(stopped_target, 0, throttle)
        brake = np.where(stopped_target, np.maximum(brake, .35), brake)
    return np.stack([steering, throttle, brake], -1).astype(np.float32)

"""SI-unit, ego-relative observation contract. Actor identities never enter a network."""

from dataclasses import dataclass

import numpy as np


SCHEMA_VERSION = "carla-structured-v1"
WAYPOINTS = 5
VEHICLE_SLOTS = 6
PEDESTRIAN_SLOTS = 6
SLOTS = VEHICLE_SLOTS + PEDESTRIAN_SLOTS
OBJECT_FIELDS = ("forward_m", "left_m", "relative_forward_mps", "relative_left_mps",
                 "half_length_m", "half_width_m", "confidence")
CONTEXT_FIELDS = (
    "forward_speed_mps", "left_speed_mps", "forward_acceleration_mps2",
    "left_acceleration_mps2", "left_yaw_rate_radps", "previous_steering",
    "previous_throttle", "previous_brake", "lane_offset_left_m", "heading_error_sin",
    "heading_error_cos", "speed_limit_mps", "cautious_target_speed_mps",
    "route_remaining_m", "light_distance_m", "stop_sign_distance_m",
    "rain_percent", "wetness_percent", "fog_percent", "sun_altitude_degrees")
CONTEXT = len(CONTEXT_FIELDS)
OBJECT_OFFSET = CONTEXT + 2 * WAYPOINTS
CONTINUOUS = OBJECT_OFFSET + SLOTS * len(OBJECT_FIELDS)
LIGHT_CLASSES = ("none", "unknown", "red", "yellow", "green")
OBJECT_CLASSES = ("vehicle", "adult", "child")
OBS_KEYS = ("continuous", "present", "classes", "controls", "light", "ids")
EVENTS = ("collision", "red_light", "stop_sign", "illegal_lane", "offroad", "wrong_way",
          "unsafe_child", "unsafe_weather", "success")


def schema_description():
    return {"version": SCHEMA_VERSION, "frame": "x forward, y left; SI units; radians except sun altitude",
            "context": list(CONTEXT_FIELDS), "waypoint_distances_m": [5, 10, 20, 30, 40],
            "waypoint_fields": ["forward_m", "left_m"], "vehicle_slots": VEHICLE_SLOTS,
            "pedestrian_slots": PEDESTRIAN_SLOTS, "object_fields": list(OBJECT_FIELDS),
            "light_classes": list(LIGHT_CLASSES), "object_classes": list(OBJECT_CLASSES),
            "actions": ["steering [-1,1]", "throttle [0,1]", "brake [0,1]"],
            "identities": "numeric metadata only; no hashes; -1 means absent"}


@dataclass
class Observation:
    continuous: np.ndarray
    present: np.ndarray
    classes: np.ndarray
    controls: np.ndarray
    light: np.ndarray
    ids: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.zeros(CONTINUOUS, np.float32), np.zeros(SLOTS, bool),
                   np.zeros(SLOTS, np.int64), np.zeros(2, bool), np.array(0, np.int64),
                   np.full(SLOTS, -1, np.int64))

    def arrays(self):
        return {key: np.asarray(getattr(self, key)) for key in OBS_KEYS}

    def validate(self):
        shapes = {"continuous": (CONTINUOUS,), "present": (SLOTS,), "classes": (SLOTS,),
                  "controls": (2,), "light": (), "ids": (SLOTS,)}
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid observation {name}: expected finite {shape}, got {value.shape}")
        if np.any((self.classes < 0) | (self.classes >= len(OBJECT_CLASSES))):
            raise ValueError("Invalid object class")
        if not 0 <= int(self.light) < len(LIGHT_CLASSES):
            raise ValueError("Invalid traffic-light class")
        if not np.isin(self.present, [False, True]).all() or not np.isin(self.controls, [False, True]).all():
            raise ValueError("Presence fields must be binary masks")
        if not np.issubdtype(self.classes.dtype, np.integer) or not np.issubdtype(self.ids.dtype, np.integer):
            raise ValueError("Object classes/identities must be integers")
        if np.any(np.asarray(self.ids)[~np.asarray(self.present, bool)] != -1):
            raise ValueError("Absent object slots must have identity -1")
        objects = self.continuous[OBJECT_OFFSET:].reshape(SLOTS, len(OBJECT_FIELDS))[self.present.astype(bool)]
        if np.any(objects[:, 4:6] < 0) or np.any(objects[:, 6] < 0) or np.any(objects[:, 6] > 1):
            raise ValueError("Object extents must be nonnegative and confidence in [0, 1]")
        return self


def continuous_valid(present, controls):
    shape = (*present.shape[:-1], CONTINUOUS)
    valid = np.ones(shape, bool)
    valid[..., 14:16] = controls
    valid[..., OBJECT_OFFSET:] = np.repeat(present, len(OBJECT_FIELDS), axis=-1)
    return valid


@dataclass
class Normalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, episodes):
        """Training episodes only. Share object statistics across interchangeable slots."""
        values = np.concatenate([ep.arrays["continuous"] for ep in episodes])
        valid = np.concatenate([continuous_valid(ep.arrays["present"], ep.arrays["controls"])
                                for ep in episodes])
        count = valid.sum(axis=0)
        mean = (values * valid).sum(axis=0) / np.maximum(count, 1)
        var = ((values - mean) ** 2 * valid).sum(axis=0) / np.maximum(count, 1)
        objects = values[:, OBJECT_OFFSET:].reshape(-1, SLOTS, len(OBJECT_FIELDS))
        mask = valid[:, OBJECT_OFFSET:].reshape(objects.shape)
        n = np.maximum(mask.sum(axis=(0, 1)), 1)
        object_mean = (objects * mask).sum(axis=(0, 1)) / n
        object_var = ((objects - object_mean) ** 2 * mask).sum(axis=(0, 1)) / n
        mean[OBJECT_OFFSET:] = np.tile(object_mean, SLOTS)
        var[OBJECT_OFFSET:] = np.tile(object_var, SLOTS)
        # Constant and unseen dimensions retain physical units instead of huge amplification.
        scale = np.where(var > 1e-4, np.sqrt(var), 1.0)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def normalize(self, values, present, controls):
        return np.where(continuous_valid(present, controls),
                        (values - self.mean) / self.scale, 0).astype(np.float32)

    def denormalize(self, values):
        return values * self.scale + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value):
        result = cls(np.array(value["mean"], np.float32), np.array(value["scale"], np.float32))
        if result.mean.shape != (CONTINUOUS,) or result.scale.shape != (CONTINUOUS,) or not (
                np.isfinite(result.mean).all() and np.isfinite(result.scale).all() and
                (result.scale > 0).all()):
            raise ValueError("Invalid normalizer")
        return result

"""CARLA sensor setup, frame synchronization, and perception contracts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from threading import Condition
import time
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from .observations import FrameBundle


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    fov: float = 90.0
    sensor_tick: float = 0.05
    location: tuple[float, float, float] = (1.5, 0.0, 2.4)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll

    def __post_init__(self) -> None:
        if self.width != 1280 or self.height != 720:
            raise ValueError("native front camera resolution must be 1280x720")
        if not math.isfinite(self.fov) or not 1.0 <= self.fov < 179.0:
            raise ValueError("camera FOV must be in [1, 179) degrees")
        if not math.isfinite(self.sensor_tick) or self.sensor_tick <= 0:
            raise ValueError("camera sensor_tick must be positive")
        if len(self.location) != 3 or len(self.rotation) != 3:
            raise ValueError("camera mount location and rotation must have three values")
        if not np.isfinite((*self.location, *self.rotation)).all():
            raise ValueError("camera mount values must be finite")


@dataclass(frozen=True)
class RadarConfig:
    horizontal_fov: float = 30.0
    vertical_fov: float = 10.0
    range: float = 40.0
    points_per_second: int = 1500
    sensor_tick: float = 0.05

    def __post_init__(self) -> None:
        if (not math.isfinite(self.horizontal_fov) or not math.isfinite(self.vertical_fov)
                or not 0 < self.horizontal_fov <= 360 or not 0 < self.vertical_fov <= 180):
            raise ValueError("radar FOV must be positive and within its angular range")
        if (not math.isfinite(self.range) or not math.isfinite(self.sensor_tick)
                or self.range <= 0 or self.points_per_second <= 0 or self.sensor_tick <= 0):
            raise ValueError("radar range, rate and sensor_tick must be positive")


RADAR_SIDES = ("front", "left", "right", "rear")
RADAR_YAWS = {"front": 0.0, "left": -90.0, "right": 90.0, "rear": 180.0}
RADAR_LOCATIONS = {
    "front": (1.5, 0.0, 1.0),
    "left": (0.0, -0.8, 1.0),
    "right": (0.0, 0.8, 1.0),
    "rear": (-1.5, 0.0, 1.0),
}


@dataclass
class PerceptionPrediction:
    """Provider outputs; objects/context use 16 features and aligned metadata."""

    objects: np.ndarray
    context: np.ndarray | None = None
    object_valid: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    depth_m: np.ndarray | None = None
    driving: Mapping[str, Any] | None = None
    frame: int | None = None
    timestamp: float | None = None
    child_labels_authored: bool = False

    def __post_init__(self) -> None:
        self.objects = np.asarray(self.objects, dtype=np.float32)
        if self.objects.ndim != 2 or self.objects.shape[1] != 16:
            raise ValueError("objects must have shape [N, 16]")
        count = len(self.objects)
        if self.object_valid is None:
            self.object_valid = np.isfinite(self.objects).all(axis=1)
        else:
            self.object_valid = np.asarray(self.object_valid, dtype=np.bool_)
            if self.object_valid.shape != (count,):
                raise ValueError("object_valid must have shape [N]")
            self.object_valid &= np.isfinite(self.objects).all(axis=1)
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=np.float32)
            if self.uncertainty.shape not in ((count,), (count, 16)):
                raise ValueError("uncertainty must have shape [N] or [N, 16]")
        if self.depth_m is not None:
            self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
            if self.depth_m.shape != (count,):
                raise ValueError("depth_m must have shape [N]")
        if self.context is not None:
            self.context = np.asarray(self.context, dtype=np.float32)
            if self.context.shape != (16,):
                raise ValueError("context must have shape [16]")

    @property
    def uncertainty_valid(self) -> np.ndarray:
        if self.uncertainty is None:
            return np.zeros(len(self.objects), dtype=np.bool_)
        return (np.isfinite(self.uncertainty).all(axis=1)
                if self.uncertainty.ndim == 2 else np.isfinite(self.uncertainty))

    @property
    def depth_valid(self) -> np.ndarray:
        return (np.zeros(len(self.objects), dtype=np.bool_) if self.depth_m is None
                else np.isfinite(self.depth_m) & (self.depth_m > 0))

    def validate_for(self, bundle: FrameBundle, *, timestamp_tolerance: float = 1e-6) -> None:
        """Require predictions to identify the exact frame and timestamp."""
        if self.frame is None or self.timestamp is None:
            raise ValueError("perception prediction must include frame and timestamp")
        if self.frame != bundle.frame:
            raise ValueError(f"perception frame {self.frame} != observation frame {bundle.frame}")
        if abs(self.timestamp - bundle.timestamp) > timestamp_tolerance:
            raise ValueError("perception timestamp does not match observation timestamp")


@runtime_checkable
class PerceptionProvider(Protocol):
    def predict(self, bundle: FrameBundle) -> PerceptionPrediction: ...


class MockPerceptionProvider:
    """Deterministic external-provider stand-in, with no oracle actor access."""

    def predict(self, bundle: FrameBundle) -> PerceptionPrediction:
        return PerceptionPrediction(objects=np.zeros((0, 16), np.float32),
                                    object_valid=np.zeros((0,), np.bool_),
                                    driving={}, frame=bundle.frame,
                                    timestamp=bundle.timestamp)


def decode_carla_rgb(image: Any) -> np.ndarray:
    """Convert CARLA BGRA image/raw_data into HWC RGB uint8."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, [2, 1, 0]].copy()


def decode_carla_depth(image: Any) -> np.ndarray:
    """Decode CARLA packed BGRA depth into metres (float32)."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b, g, r = (bgra[:, :, i].astype(np.float32) for i in (0, 1, 2))
    encoded = r + 256.0 * g + 65536.0 * b
    return encoded * (1000.0 / 16777215.0)


def associate_depth(depth_m: np.ndarray | None,
                    boxes_xyxy: np.ndarray,
                    *, min_valid_fraction: float = 0.25,
                    visible_mask: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate object depth from aligned image-space boxes.

    Returns (median depth in metres, robust uncertainty in metres, valid).
    Uses the central 60% of each box to reduce boundary contamination. A
    semantic visible-pixel mask can exclude known occluded pixels; depth
    alone cannot determine whether a box is occluded. Invalid boxes return
    NaN depth/uncertainty and valid=False.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape [N, 4]")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in [0, 1]")
    depths = np.full(len(boxes), np.nan, dtype=np.float32)
    uncertainty = np.full(len(boxes), np.nan, dtype=np.float32)
    valid = np.zeros(len(boxes), dtype=np.bool_)
    if depth_m is None:
        return depths, uncertainty, valid
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must have shape [height, width]")
    if visible_mask is not None:
        visible = np.asarray(visible_mask, dtype=np.bool_)
        if visible.shape != depth.shape:
            raise ValueError("visible_mask must match depth_m shape")
    else:
        visible = None
    height, width = depth.shape
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if not np.isfinite([x1, y1, x2, y2]).all() or x2 <= x1 or y2 <= y1:
            continue
        # Ignore outer 20% on each edge, which tends to include unrelated pixels.
        dx, dy = 0.2 * (x2 - x1), 0.2 * (y2 - y1)
        left = max(0, int(math.floor(x1 + dx)))
        right = min(width, int(math.ceil(x2 - dx)))
        top = max(0, int(math.floor(y1 + dy)))
        bottom = min(height, int(math.ceil(y2 - dy)))
        if right <= left or bottom <= top:
            continue
        crop = depth[top:bottom, left:right]
        usable = np.isfinite(crop) & (crop > 0)
        if visible is not None:
            usable &= visible[top:bottom, left:right]
        values = crop[usable]
        if values.size / crop.size < min_valid_fraction or values.size == 0:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        depths[i] = median
        # Robust spread + penalty for sparse/occluded valid pixels.
        uncertainty[i] = max(1.4826 * mad, 0.01) / math.sqrt(values.size / crop.size)
        valid[i] = True
    return depths, uncertainty, valid


def radar_to_ego(readings: np.ndarray, *, location=(0.0, 0.0, 0.0),
                 rotation=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Convert CARLA [depth, azimuth, altitude, velocity] to ego [x,y,z,v].

    CARLA local sensor axes use x forward and y right. Rotation is Euler
    pitch/yaw/roll in degrees and follows ``carla.Transform.get_matrix``.
    Output's first four channels are xyz metres and radial velocity m/s.
    """
    rows = np.asarray(readings, dtype=np.float32)
    if rows.size == 0:
        return np.zeros((0, 4), np.float32)
    if rows.ndim != 2 or rows.shape[1] < 4:
        raise ValueError("readings must have shape [N, 4]")
    d, az, alt = rows[:, 0], rows[:, 1], rows[:, 2]
    xyz = np.stack((d * np.cos(alt) * np.cos(az),
                    d * np.cos(alt) * np.sin(az), d * np.sin(alt)), axis=1)
    pitch, yaw, roll = np.deg2rad(rotation)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
        [sp, -cp * sr, cp * cr],
    ])
    transformed = xyz @ matrix.T + np.asarray(location, dtype=np.float32)
    return np.column_stack((transformed, rows[:, 3])).astype(np.float32)


def effective_radar_coverage(config: RadarConfig, yaws: Mapping[str, float] = RADAR_YAWS,
                             samples: int = 360) -> np.ndarray:
    """Boolean 360-degree coverage by azimuth; accounts for FOV overlaps/gaps."""
    angles = np.arange(samples) * (360.0 / samples) - 180.0
    covered = np.zeros(samples, dtype=np.bool_)
    for yaw in yaws.values():
        delta = (angles - yaw + 180.0) % 360.0 - 180.0
        covered |= np.abs(delta) <= config.horizontal_fov / 2.0
    return covered


class SensorFrameBuffer:
    """Bounded, epoch-aware buffer; only exact/current frames are returned."""

    def __init__(self, capacity: int = 8):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._condition = Condition()
        self._data: dict[str, dict[int, tuple[float, Any]]] = defaultdict(dict)
        self._calibration: dict[str, Mapping[str, Any]] = {}
        self._collision_events: list[tuple[int, Mapping[str, Any]]] = []
        self._collision_sequence = 0
        self._collision_consumed_sequences: set[int] = set()
        self._collision_capacity = max(64, capacity * 16)
        self._epoch = 0
        self._highest_frame = -1
        self._min_frame = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset(self, start_frame: int | None = None) -> int:
        with self._condition:
            self._epoch += 1
            # CARLA frame IDs normally continue increasing across episodes.
            # A caller can supply the first expected frame if it resets CARLA.
            self._min_frame = (self._highest_frame + 1 if start_frame is None
                               else int(start_frame))
            self._data.clear()
            self._collision_events.clear()
            self._collision_consumed_sequences.clear()
            self._condition.notify_all()
            return self._epoch

    def set_calibration(self, sensor: str, calibration: Mapping[str, Any]) -> None:
        self._calibration[sensor] = dict(calibration)

    def add(self, sensor: str, frame: int, timestamp: float, data: Any,
            *, epoch: int | None = None) -> bool:
        with self._condition:
            if epoch is not None and epoch != self._epoch:
                return False
            if frame < self._min_frame:
                return False
            frames = self._data[sensor]
            if frames and frame < max(frames):
                return False  # reject out-of-order stale callbacks
            frames[frame] = (timestamp, data)
            self._highest_frame = max(self._highest_frame, frame)
            for old in sorted(frames)[:-self.capacity]:
                del frames[old]
            self._condition.notify_all()
            return True

    def add_collision(self, event: Mapping[str, Any]) -> bool:
        """Record an event by event frame; collision callbacks have no heartbeat."""
        frame = int(event["frame"])
        timestamp = float(event["timestamp"])
        with self._condition:
            if frame < self._min_frame:
                return False
            frames = self._data["collision"]
            frames[frame] = (timestamp, dict(event))
            self._highest_frame = max(self._highest_frame, frame)
            for old in sorted(frames)[:-self.capacity]:
                del frames[old]
            self._collision_sequence += 1
            self._collision_events.append((self._collision_sequence, dict(event)))
            if len(self._collision_events) > self._collision_capacity:
                self._collision_events = self._collision_events[-self._collision_capacity:]
                oldest = self._collision_events[0][0]
                self._collision_consumed_sequences = {
                    sequence for sequence in self._collision_consumed_sequences
                    if sequence >= oldest
                }
            self._condition.notify_all()
            return True

    def collision_events(self, frame: int) -> tuple[Mapping[str, Any], ...]:
        """Peek at unconsumed collision events with event frame <= ``frame``."""
        with self._condition:
            return tuple(dict(event) for sequence, event in self._collision_events
                         if sequence not in self._collision_consumed_sequences
                         and int(event["frame"]) <= frame)

    def consume_collision(self, frame: int) -> tuple[Mapping[str, Any], ...]:
        """Return and consume pending collisions up to this frame, retaining future events."""
        with self._condition:
            ready = [(sequence, event) for sequence, event in self._collision_events
                     if sequence not in self._collision_consumed_sequences
                     and int(event["frame"]) <= frame]
            if not ready:
                return ()
            events = tuple(dict(event) for _, event in ready)
            self._collision_consumed_sequences.update(sequence for sequence, _ in ready)
            return events

    def bundle(self, frame: int, *, timeout: float = 0.0,
               required: tuple[str, ...] = ("rgb",),
               timestamp_tolerance: float = 1e-6) -> FrameBundle:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not all(frame in self._data[name] for name in required):
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                self._condition.wait(remain)
            assembled = {key: values[frame] for key, values in self._data.items()
                         if frame in values}
            reference_timestamp = (assembled["rgb"][0] if "rgb" in assembled
                                   else next(iter(assembled.values()))[0]
                                   if assembled else 0.0)
            timestamps = {key: value[0] for key, value in assembled.items()}
            valid_sensors = {key for key, timestamp in timestamps.items()
                             if abs(timestamp - reference_timestamp) <= timestamp_tolerance}
            payload = {key: value[1] for key, value in assembled.items()
                       if key in valid_sensors}
            valid = {key: key in payload for key in ("rgb", "depth", *RADAR_SIDES)}
            valid["collision"] = "collision" in self._calibration
            radars = {side: payload[side] for side in RADAR_SIDES if side in payload}
            sensor_calibrations = {
                key: {"frame": None, "timestamp": None, "valid": False, **dict(details)}
                for key, details in self._calibration.items()
            }
            for key, timestamp in timestamps.items():
                sensor_calibrations.setdefault(key, {})
                sensor_calibrations[key].update({
                    "frame": frame, "timestamp": timestamp,
                    "valid": key in valid_sensors,
                })
            calibration = {
                "epoch": self._epoch,
                "units": {"position": "metres", "time": "seconds",
                          "radial_velocity": "metres/second", "angles": "radians"},
                "sensors": sensor_calibrations,
            }
            return FrameBundle(frame=frame, timestamp=reference_timestamp, rgb=payload.get("rgb"),
                               depth=payload.get("depth"), radars=radars,
                               valid=valid, calibration=calibration,
                               events=self.collision_events(frame))


class SensorRig:
    """Attach and manage RGB/depth/radar/collision sensors on a CARLA actor."""

    def __init__(self, world: Any, vehicle: Any, camera: CameraConfig | None = None,
                 radar: RadarConfig | None = None, *, capacity: int = 8):
        self.world, self.vehicle = world, vehicle
        self.camera = camera or CameraConfig()
        self.radar = radar or RadarConfig()
        self.buffer = SensorFrameBuffer(capacity)
        self.sensors: list[Any] = []
        self._closed = False
        try:
            self._attach()
        except Exception:
            self.close()
            raise

    def _attach(self) -> None:
        try:
            import carla
        except ImportError as exc:
            raise RuntimeError("CARLA Python API is required to create SensorRig") from exc
        library = self.world.get_blueprint_library()
        def make(kind: str, loc: tuple[float, float, float], rot: tuple[float, float, float], attrs: Mapping[str, Any]):
            bp = library.find(kind)
            for key, value in attrs.items():
                bp.set_attribute(key, str(value))
            transform = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                        carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            self.sensors.append(sensor)
            return sensor
        c = self.camera
        fx = c.width / (2.0 * math.tan(math.radians(c.fov) / 2.0))
        camera_calibration = {
            "extrinsics_ego": {"location_m": c.location, "rotation_pitch_yaw_roll_deg": c.rotation},
            "intrinsics_px": {"fx": fx, "fy": fx, "cx": (c.width - 1) / 2,
                              "cy": (c.height - 1) / 2},
            "resolution_px": (c.width, c.height), "fov_deg": c.fov,
            "sensor_tick_s": c.sensor_tick,
        }
        self.buffer.set_calibration("rgb", camera_calibration)
        self.buffer.set_calibration("depth", camera_calibration)
        rgb = make("sensor.camera.rgb", c.location, c.rotation,
                   {"image_size_x": c.width, "image_size_y": c.height, "fov": c.fov,
                    "sensor_tick": c.sensor_tick})
        rgb.listen(lambda image: self.buffer.add("rgb", image.frame, image.timestamp,
                                                  decode_carla_rgb(image)))
        depth = make("sensor.camera.depth", c.location, c.rotation,
                     {"image_size_x": c.width, "image_size_y": c.height, "fov": c.fov,
                      "sensor_tick": c.sensor_tick})
        depth.listen(lambda image: self.buffer.add("depth", image.frame, image.timestamp,
                                                    decode_carla_depth(image)))
        for side in RADAR_SIDES:
            yaw = RADAR_YAWS[side]
            loc = RADAR_LOCATIONS[side]
            rot = (0.0, yaw, 0.0)
            self.buffer.set_calibration(side, {
                "extrinsics_ego": {"location_m": loc, "rotation_pitch_yaw_roll_deg": rot},
                "horizontal_fov_deg": self.radar.horizontal_fov,
                "vertical_fov_deg": self.radar.vertical_fov,
                "range_m": self.radar.range,
                "sensor_tick_s": self.radar.sensor_tick,
                "output_coordinates": "ego_x_forward_y_right_z_up",
            })
            sensor = make("sensor.other.radar", loc, rot,
                          {"horizontal_fov": self.radar.horizontal_fov,
                           "vertical_fov": self.radar.vertical_fov,
                           "range": self.radar.range,
                           "points_per_second": self.radar.points_per_second,
                           "sensor_tick": self.radar.sensor_tick})
            sensor.listen(lambda data, side=side, loc=loc, rot=rot: self.buffer.add(
                side, data.frame, data.timestamp,
                radar_to_ego(np.array([[d.depth, d.azimuth, d.altitude, d.velocity]
                                       for d in data], dtype=np.float32), location=loc, rotation=rot),
                ))
        collision = make("sensor.other.collision", (0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0), {})
        self.buffer.set_calibration("collision", {"sensor_tick_s": 0.0})
        def on_collision(event: Any) -> None:
            other_actor = getattr(event, "other_actor", None)
            self.buffer.add_collision({
                "frame": int(event.frame),
                "timestamp": float(getattr(event, "timestamp", 0.0)),
                "other_actor_id": getattr(other_actor, "id", None),
                "other_actor_type": getattr(other_actor, "type_id", "unknown"),
            })
        collision.listen(on_collision)

    def get(self, frame: int, timeout: float = 0.1) -> FrameBundle:
        if self._closed:
            raise RuntimeError("SensorRig is closed")
        return self.buffer.bundle(frame, timeout=timeout,
                                  required=("rgb", "depth", *RADAR_SIDES))

    def collision_events(self, frame: int) -> tuple[Mapping[str, Any], ...]:
        return self.buffer.collision_events(frame)

    def consume_collision(self, frame: int) -> tuple[Mapping[str, Any], ...]:
        return self.buffer.consume_collision(frame)

    def reset(self, start_frame: int | None = None) -> None:
        """Start a new episode epoch; late callbacks from prior epochs are ignored."""
        self.buffer.reset(start_frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sensor in reversed(self.sensors):
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
        self.sensors.clear()

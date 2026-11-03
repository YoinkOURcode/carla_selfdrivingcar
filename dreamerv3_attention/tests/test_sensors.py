import numpy as np
import pytest
import sys
import time
from types import SimpleNamespace

from dreamerv3_attention.observations import FrameBundle, pack_observation
from dreamerv3_attention.sensors import (
    CameraConfig, MockPerceptionProvider, PerceptionPrediction, RadarConfig,
    RADAR_LOCATIONS, SensorFrameBuffer, associate_depth,
    decode_carla_depth, effective_radar_coverage, radar_to_ego,
)


def test_packing_resizes_rgb_and_masks_missing_and_overflow_tokens():
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    context = np.arange(16, dtype=np.float32)
    objects = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    packed = pack_observation(rgb, context, objects,
                              {"front": np.array([[2, 0, 0, -1]], np.float32)},
                              max_objects=2, radar_capacity=1, model_size=(3, 2))
    assert packed["pixels"].shape == (3, 2, 3)
    assert packed["pixels"].dtype == np.uint8
    assert packed["pixels"][0, 0, 0] == rgb[0, 0, 0]
    assert packed["tokens"].shape == (7, 16)
    assert packed["tokens"].dtype == np.float32
    assert packed["mask"].tolist() == [True, True, True, True, False, False, False]
    assert np.array_equal(packed["tokens"][0], context)
    assert np.allclose(packed["tokens"][3, :4], [0.02, 0, 0, -1 / 30])
    assert packed["radar_valid"].tolist() == [True, False, False, False]


def test_missing_rgb_and_nonfinite_rows_are_explicitly_invalid():
    packed = pack_observation(None, np.full(16, np.nan, np.float32),
                              np.array([[np.nan] * 16], np.float32),
                              max_objects=1, radar_capacity=0)
    assert packed["pixels"].sum() == 0
    assert not packed["pixels_valid"]
    assert not packed["mask"].any()


def test_decode_carla_depth_maps_zero_and_max_to_zero_and_1000m():
    # CARLA channels are BGRA; depth integer is R + 256*G + 65536*B.
    bgra = np.array([[[0, 0, 0, 255], [255, 255, 255, 255]]], dtype=np.uint8)
    class Image:
        height, width, raw_data = 1, 2, bgra.tobytes()
    decoded = decode_carla_depth(Image())
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, [[0.0, 1000.0]], atol=1e-4)


def test_depth_association_is_robust_and_marks_invalid_boxes_missing():
    depth = np.full((10, 10), 12.0, np.float32)
    depth[1:3, 1:3] = 40.0  # boundary contamination is outside central crop
    depth[:, :] = np.where(np.indices(depth.shape)[0] < 8, depth, np.nan)
    boxes = np.array([[0, 0, 10, 8], [20, 20, 30, 30]], np.float32)
    distances, uncertainty, valid = associate_depth(depth, boxes)
    assert valid.tolist() == [True, False]
    assert distances[0] == 12.0
    assert uncertainty[0] > 0 and np.isfinite(uncertainty[0])
    assert np.isnan(distances[1]) and np.isnan(uncertainty[1])
    missing = associate_depth(None, boxes)
    assert not missing[2].any()


def test_depth_association_can_use_explicit_visible_pixel_mask():
    depth = np.full((10, 10), 12.0, np.float32)
    depth[4:6, 4:6] = 40.0
    visible = np.ones_like(depth, dtype=np.bool_)
    visible[4:6, 4:6] = False
    distance, _, valid = associate_depth(depth, np.array([[0, 0, 10, 10]]),
                                         visible_mask=visible)
    assert valid.tolist() == [True]
    assert distance[0] == 12.0


def test_radar_transform_uses_mount_translation_and_full_yaw():
    local = np.array([[10.0, 0.0, 0.0, -2.0]], np.float32)
    transformed = radar_to_ego(local, location=(1, 2, 3), rotation=(0, 90, 0))
    assert np.allclose(transformed[0, :3], [1, 12, 3], atol=1e-5)
    assert transformed[0, 3] == -2


def test_radar_rotation_matches_carla_transform_matrix_without_server():
    carla = pytest.importorskip("carla")
    rotation = (10.0, 35.0, 20.0)
    point = np.array([[5.0, 0.0, 0.0, 0.0]], np.float32)
    actual = radar_to_ego(point, location=(1.0, 2.0, 3.0), rotation=rotation)[0, :3]
    transform = carla.Transform(carla.Location(x=1, y=2, z=3),
                                carla.Rotation(pitch=10, yaw=35, roll=20))
    matrix = np.asarray(transform.get_matrix(), dtype=np.float64)
    expected = matrix[:3, :3] @ np.array([5.0, 0.0, 0.0]) + matrix[:3, 3]
    assert np.allclose(actual, expected, atol=1e-5)

    # Explicit pitch/roll sign checks: CARLA positive pitch raises +x;
    # positive roll lowers +y in its local coordinate convention.
    pitched = radar_to_ego(point, rotation=(10, 0, 0))[0]
    assert pitched[2] > 0
    y_point = np.array([[5.0, np.pi / 2, 0.0, 0.0]], np.float32)
    rolled = radar_to_ego(y_point, rotation=(0, 0, 20))[0]
    assert rolled[2] < 0


def test_coverage_reports_fov_gaps_instead_of_claiming_full_coverage():
    coverage = effective_radar_coverage(RadarConfig(horizontal_fov=30))
    assert coverage.shape == (360,)
    # Inclusive FOV boundaries include 31 samples at 1-degree resolution.
    assert coverage.sum() == 124
    assert not coverage.all()


def test_radar_tokens_are_nearest_first_and_mounts_cover_each_direction():
    rows = np.array([[10, 0, 0, 0], [2, 0, 0, 0], [2.1, 0, 0, 0]], np.float32)
    packed = pack_observation(None, np.zeros(16, np.float32),
                              radars={"front": rows}, max_objects=0,
                              radar_capacity=3)
    front = packed["tokens"][1:4]
    assert packed["mask"][1:4].tolist() == [True, True, False]
    assert np.allclose(front[:2, 0], [0.02, 0.1])
    assert RADAR_LOCATIONS["left"][1] < 0 < RADAR_LOCATIONS["right"][1]
    assert RADAR_LOCATIONS["front"][0] > 0 > RADAR_LOCATIONS["rear"][0]
    empty = pack_observation(None, np.zeros(16, np.float32),
                             radars={"front": np.zeros((0, 4), np.float32)},
                             max_objects=0, radar_capacity=2)
    assert empty["radar_valid"].tolist() == [True, False, False, False]
    assert not empty["mask"][1:3].any()


def test_frame_buffer_rejects_stale_future_leakage_and_old_epoch_callbacks():
    buf = SensorFrameBuffer(capacity=2)
    assert buf.add("rgb", 10, 1.0, np.ones((1, 1, 3), np.uint8))
    assert buf.add("front", 10, 1.0, np.ones((1, 4), np.float32))
    assert not buf.add("rgb", 9, 0.9, "stale")
    bundle = buf.bundle(10)
    assert bundle.frame == 10 and bundle.valid["rgb"]
    assert bundle.valid["front"] and not bundle.valid["rear"]
    # Frame 11 data may not leak backward into frame 10.
    buf.add("rgb", 11, 1.1, "future")
    assert buf.bundle(10).rgb.shape == (1, 1, 3)
    old_epoch = buf.epoch
    assert buf.reset() == old_epoch + 1
    assert not buf.add("rgb", 10, 1.0, "late")
    assert buf.add("rgb", 12, 1.2, "new")
    assert buf.bundle(10).rgb is None


def test_mock_provider_receives_bundle_and_returns_no_oracle_objects():
    bundle = FrameBundle(1, 0.1)
    prediction = MockPerceptionProvider().predict(bundle)
    assert prediction.objects.shape == (0, 16)
    assert prediction.object_valid.shape == (0,)
    assert prediction.driving == {}
    prediction.validate_for(bundle)


def test_native_camera_contract_is_1280_by_720():
    cfg = CameraConfig()
    assert (cfg.width, cfg.height) == (1280, 720)
    assert cfg.sensor_tick > 0 and 1 <= cfg.fov < 179
    with pytest.raises(ValueError, match="1280x720"):
        CameraConfig(width=640)
    with pytest.raises(ValueError, match="FOV"):
        CameraConfig(fov=float("nan"))
    with pytest.raises(ValueError, match="sensor_tick"):
        RadarConfig(sensor_tick=0)


def test_frame_bundle_marks_timestamp_mismatches_invalid_and_keeps_calibration():
    buf = SensorFrameBuffer()
    buf.set_calibration("rgb", {"resolution_px": (1280, 720), "fov_deg": 90})
    buf.set_calibration("front", {"range_m": 40})
    rgb = np.zeros((1, 1, 3), np.uint8)
    radar = np.ones((1, 4), np.float32)
    buf.add("rgb", 5, 1.0, rgb)
    buf.add("front", 5, 1.01, radar)
    bundle = buf.bundle(5)
    assert bundle.timestamp == 1.0
    assert bundle.valid["rgb"] and not bundle.valid["front"]
    assert "front" not in bundle.radars
    assert bundle.calibration["sensors"]["rgb"]["resolution_px"] == (1280, 720)
    assert bundle.calibration["sensors"]["front"]["timestamp"] == 1.01


def test_provider_metadata_masks_bad_objects_and_checks_frame_time():
    objects = np.zeros((3, 16), np.float32)
    prediction = PerceptionPrediction(
        objects=objects, object_valid=np.array([True, True, True]),
        uncertainty=np.array([0.1, np.nan, 0.2]),
        depth_m=np.array([10.0, 10.0, np.nan]), frame=5, timestamp=1.0,
        driving={"speed_limit_mps": 8.0, "crosswalk": True})
    assert prediction.object_valid.tolist() == [True, True, True]
    assert prediction.uncertainty_valid.tolist() == [True, False, True]
    assert prediction.depth_valid.tolist() == [True, True, False]
    prediction.validate_for(FrameBundle(frame=5, timestamp=1.0))
    with pytest.raises(ValueError, match="frame"):
        prediction.validate_for(FrameBundle(frame=6, timestamp=1.0))
    assert prediction.driving["crosswalk"] is True
    with pytest.raises(ValueError, match="include frame and timestamp"):
        PerceptionPrediction(objects=np.zeros((0, 16), np.float32)).validate_for(
            FrameBundle(frame=5, timestamp=1.0))


class _FakeCarla:
    class Location:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Rotation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Transform:
        def __init__(self, location, rotation):
            self.location, self.rotation = location, rotation


class _FakeSensor:
    def __init__(self, sensor_id, *, skip_callback=False):
        self.sensor_id = sensor_id
        self.skip_callback = skip_callback
        self.destroyed = False
        self.stopped = False

    def listen(self, callback):
        if self.skip_callback:
            return
        if self.sensor_id in ("sensor.camera.rgb", "sensor.camera.depth"):
            image = SimpleNamespace(height=720, width=1280, frame=5, timestamp=1.0,
                                    raw_data=bytes(1280 * 720 * 4))
            callback(image)
        elif self.sensor_id == "sensor.other.radar":
            class RadarData:
                frame, timestamp = 5, 1.0

                def __iter__(self):
                    return iter(())
            callback(RadarData())
        elif self.sensor_id == "sensor.other.collision":
            actor = SimpleNamespace(id=42, type_id="walker.pedestrian.0001")
            callback(SimpleNamespace(frame=4, timestamp=0.9, other_actor=actor))

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True


class _FakeBlueprint:
    def __init__(self, sensor_id):
        self.id = sensor_id
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeWorld:
    def __init__(self, *, skip_sensor=None):
        self.skip_sensor = skip_sensor
        self.sensors = []

    def get_blueprint_library(self):
        return SimpleNamespace(find=_FakeBlueprint)

    def spawn_actor(self, blueprint, transform, attach_to):
        del transform, attach_to
        sensor = _FakeSensor(blueprint.id, skip_callback=blueprint.id == self.skip_sensor)
        self.sensors.append(sensor)
        return sensor


def _make_fake_rig(monkeypatch, *, skip_sensor=None):
    monkeypatch.setitem(sys.modules, "carla", _FakeCarla)
    from dreamerv3_attention.sensors import SensorRig
    world = _FakeWorld(skip_sensor=skip_sensor)
    rig = SensorRig(world, vehicle=object())
    return rig, world


def test_sensor_rig_waits_for_all_continuous_sensors_and_preserves_collision_events(monkeypatch):
    rig, world = _make_fake_rig(monkeypatch)
    bundle = rig.get(5, timeout=0.01)
    assert all(bundle.valid[name] for name in ("rgb", "depth", "front", "left", "right", "rear"))
    assert bundle.valid["collision"]
    assert len(bundle.events) == 1
    assert bundle.events[0]["frame"] == 4
    assert bundle.events[0]["other_actor_type"].startswith("walker")

    # Late callbacks no later than current frame remain pending until consumed.
    rig.buffer.add_collision({"frame": 3, "timestamp": 0.8,
                              "other_actor_id": 7, "other_actor_type": "vehicle.test"})
    pending = rig.collision_events(5)
    assert len(pending) == 2
    assert len(rig.consume_collision(5)) == 2
    assert rig.collision_events(5) == ()

    # Future collisions are retained but never surfaced before their frame.
    rig.buffer.add_collision({"frame": 6, "timestamp": 1.1,
                              "other_actor_id": 8, "other_actor_type": "vehicle.test"})
    assert rig.collision_events(5) == ()
    assert rig.consume_collision(5) == ()
    assert len(rig.consume_collision(6)) == 1

    rig.close()
    assert all(sensor.stopped and sensor.destroyed for sensor in world.sensors)
    with pytest.raises(RuntimeError, match="closed"):
        rig.get(5)


def test_sensor_rig_marks_a_skipped_radar_frame_invalid_after_bounded_wait(monkeypatch):
    rig, _ = _make_fake_rig(monkeypatch, skip_sensor="sensor.other.radar")
    started = time.monotonic()
    bundle = rig.get(5, timeout=0.01)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5
    assert bundle.valid["rgb"] and bundle.valid["depth"]
    assert not bundle.valid["front"]
    assert "front" not in bundle.radars
    rig.close()


def test_late_collision_before_future_collision_does_not_consume_future_event():
    buf = SensorFrameBuffer()
    buf.add_collision({"frame": 10, "timestamp": 1.0, "other_actor_type": "vehicle.future"})
    buf.add_collision({"frame": 3, "timestamp": 0.3, "other_actor_type": "walker.late"})
    assert [event["other_actor_type"] for event in buf.consume_collision(5)] == ["walker.late"]
    assert [event["other_actor_type"] for event in buf.consume_collision(10)] == ["vehicle.future"]

"""Camera color conversion and viewer isolation from simulation ownership."""

import struct
import sys
import json
from types import SimpleNamespace
import zlib

from world_model_rl.config import Config
from world_model_rl.viewer import CameraFeed, image_png
from world_model_rl.telemetry import EpisodeTelemetry, read_telemetry
from world_model_rl.schema import Observation, OBJECT_OFFSET


def test_camera_png_has_rgb_pixels_and_valid_chunks():
    image = SimpleNamespace(width=2, height=1, raw_data=bytes([1, 2, 3, 255, 11, 12, 13, 255]))
    png = image_png(image)
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    position, chunks = 8, {}
    while position < len(png):
        size = struct.unpack('!I', png[position:position + 4])[0]
        kind = png[position + 4:position + 8]
        data = png[position + 8:position + 8 + size]
        crc = struct.unpack('!I', png[position + 8 + size:position + 12 + size])[0]
        assert crc == zlib.crc32(kind + data)
        chunks[kind] = data
        position += 12 + size
    assert struct.unpack('!2I5B', chunks[b'IHDR']) == (2, 1, 8, 2, 0, 0, 0)
    assert zlib.decompress(chunks[b'IDAT']) == bytes([0, 3, 2, 1, 13, 12, 11])


def test_viewer_follows_episode_changes_and_only_destroys_its_cameras(monkeypatch):
    # These fake interfaces deliberately provide no world.tick, settings mutation,
    # vehicle control or vehicle destruction methods.
    feed = CameraFeed(Config())
    class Episodes:
        index = 0
        def is_set(self):
            return self.index == 3
        def wait(self, seconds):
            self.index += 1
    feed.stopped = Episodes()
    image = SimpleNamespace(width=1, height=1, raw_data=bytes([0, 0, 0, 255]), frame=9)
    class Camera:
        is_alive = True
        stopped = destroyed = False
        def listen(self, callback):
            self.callback = callback
            callback(image)
        def stop(self):
            self.stopped = True
        def destroy(self):
            self.destroyed = True
    cameras = []
    class Actors(list):
        def filter(self, pattern):
            return self
    class World:
        def get_actors(self):
            return Actors([] if feed.stopped.index == 1 else [SimpleNamespace(
                id=100 + feed.stopped.index, attributes={'role_name': 'world_model_rl'})])
        def get_blueprint_library(self):
            return SimpleNamespace(find=lambda name: SimpleNamespace(set_attribute=lambda *args: None))
        def spawn_actor(self, *args, **kwargs):
            assert kwargs['attach_to'].id in (100, 102)
            camera = Camera()
            cameras.append(camera)
            return camera
    fake_carla = SimpleNamespace(Client=lambda *args: SimpleNamespace(
        set_timeout=lambda seconds: None, get_world=World),
        Transform=lambda *args: args, Location=lambda **kwargs: kwargs,
        Rotation=lambda **kwargs: kwargs, AttachmentType=SimpleNamespace(Rigid=1))
    monkeypatch.setitem(sys.modules, 'carla', fake_carla)
    feed.follow()
    assert len(cameras) == 2
    assert all(camera.stopped and camera.destroyed for camera in cameras)
    assert feed.snapshot()[0] is None
    # A late image from a destroyed vehicle cannot reappear after cleanup.
    cameras[0].callback(image)
    assert feed.snapshot()[0] is None


def test_episode_telemetry_tracks_actual_observations_and_resets_returns(tmp_path):
    cfg = Config()
    env = SimpleNamespace(cfg=cfg, ego=SimpleNamespace(id=42))
    metadata = {'episode_id': 3, 'collector': 'autopilot',
                'observation_source': 'carla_ground_truth_with_geometric_visibility'}
    path = tmp_path / 'live.json'
    obs = Observation.empty()
    obs.continuous[:2] = [3, 4]
    obs.continuous[5:8] = [0, 0, 0]
    obs.continuous[12:14] = [6, 75]
    obs.present[6], obs.classes[6], obs.ids[6] = True, 2, 123
    obs.continuous[OBJECT_OFFSET + 6 * 7:OBJECT_OFFSET + 7 * 7] = [15, -2, 0, 0, .3, .2, .8]
    publisher = EpisodeTelemetry(env, metadata, path)
    publisher.publish(obs, {'frame': 10})
    initial = read_telemetry(path, 42)
    assert initial['version'] == 2
    assert initial['missing_fields'] == []
    assert initial['reward_last'] is None
    assert initial['reward_component_totals'] == {}
    assert initial['route_completion'] == 0
    assert initial['applied_action'] == [0, 0, 0]
    publisher.publish(obs, {'frame': 12, 'components': {'progress': 3, 'time': -.5}}, reward=2.5)
    publisher.publish(obs, {'frame': 14, 'components': {'progress': 1, 'collision': -2},
                            'route_completion': .25, 'applied_action': [-.25, .4, .1]}, reward=-1, active=False)
    data = read_telemetry(path, 42)
    assert data['speed_mps'] == 5
    assert data['episode_return'] == 1.5 and not data['active']
    assert data['reward_last'] == -1
    assert data['reward_components'] == {'progress': 1, 'collision': -2}
    assert data['reward_component_totals'] == {'progress': 4, 'time': -.5, 'collision': -2}
    assert sum(data['reward_component_totals'].values()) == data['episode_return']
    assert data['target_speed_mps'] == 6
    assert data['route_remaining_m'] == 75 and data['route_completion'] == .25
    assert data['applied_action'] == [-.25, .4, .1]
    assert data['attention'] is None
    assert data['objects'][0]['class'] == 'child'
    assert data['objects'][0]['id'] == 123
    assert data['objects'][0]['left_m'] == -2
    assert len(data['objects']) == 1
    assert read_telemetry(path, 43) is None
    assert read_telemetry(path, 42, now=data['updated_at'] + 6)['stale']
    EpisodeTelemetry(env, {**metadata, 'episode_id': 4}, path).publish(obs, {'frame': 16})
    reset = read_telemetry(path, 42)
    assert reset['episode_return'] == 0
    assert reset['reward_component_totals'] == {} and reset['reward_last'] is None
    path.write_text('{incomplete')
    assert read_telemetry(path, 42) is None


def test_old_driving_process_telemetry_identifies_unavailable_panels(tmp_path):
    import time
    from world_model_rl.telemetry import DRIVING_DETAIL_FIELDS

    path = tmp_path / 'old-process.json'
    legacy = {'version': 1, 'updated_at': time.time(), 'actor_id': 42,
              'speed_mps': 5, 'episode_return': 7.5, 'objects': [], 'attention': None}
    path.write_text(json.dumps(legacy))
    data = read_telemetry(path, 42)
    assert data['speed_mps'] == 5 and data['episode_return'] == 7.5
    assert set(data['missing_fields']) == set(DRIVING_DETAIL_FIELDS)
    assert 'target_speed_mps' not in data and 'route_completion' not in data
    legacy['version'] = 999
    path.write_text(json.dumps(legacy))
    assert read_telemetry(path, 42) is None


def test_attention_weights_match_encoder_and_mask_empty_slots():
    import numpy as np
    import torch
    from world_model_rl.model import AttentionEncoder, tensor_observation
    from world_model_rl.schema import CONTINUOUS, Normalizer

    cfg = Config()
    cfg.model.embedding, cfg.model.attention_heads = 8, 2
    model = AttentionEncoder(cfg.model).eval()
    obs = Observation.empty()
    obs.present[0] = True
    normalizer = Normalizer(np.zeros(CONTINUOUS, np.float32), np.ones(CONTINUOUS, np.float32))
    arrays = {key: np.stack([value, value])[None] for key, value in obs.arrays().items()}
    inputs = tensor_observation(arrays, normalizer, 'cpu')
    with torch.no_grad():
        expected = model(inputs)
        encoded, weights = model(inputs, return_attention=True)
    assert weights.shape == (1, 2, 2, 21)
    torch.testing.assert_close(expected, encoded, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 2, 2))
    assert (weights[..., 10:] == 0).all()
    assert (weights[..., 9] > 0).all()


def test_telemetry_write_failure_does_not_interrupt_driving(tmp_path):
    import pytest
    env = SimpleNamespace(cfg=Config(), ego=SimpleNamespace(id=42))
    publisher = EpisodeTelemetry(env, {'episode_id': 0, 'collector': 'autopilot'}, tmp_path / 'missing' / 'live.json')
    with pytest.warns(RuntimeWarning, match='Viewer telemetry unavailable'):
        publisher.publish(Observation.empty(), {'frame': 1}, reward=2)
    publisher.publish(Observation.empty(), {'frame': 2}, reward=3)
    assert publisher.total == 5


def test_status_endpoint_contains_telemetry_without_controlling_simulator():
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen
    from world_model_rl.viewer import handler_for

    feed = SimpleNamespace(snapshot=lambda: (None, 'Waiting'), telemetry=lambda: {'episode_return': 7.5})
    with ThreadingHTTPServer(('127.0.0.1', 0), handler_for(feed)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/status') as response:
                data = json.load(response)
            assert data == {'message': 'Waiting', 'has_frame': False, 'telemetry': {'episode_return': 7.5}}
        finally:
            server.shutdown()
            thread.join()

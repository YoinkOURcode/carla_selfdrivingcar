"""Best-effort local snapshots shared by the driving process and browser viewer."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np

from .schema import OBJECT_CLASSES, OBJECT_FIELDS, OBJECT_OFFSET, SLOTS


TOKEN_LABELS = ['Ego state', 'Route', 'Weather', 'Traffic controls',
                'Waypoint 5 m', 'Waypoint 10 m', 'Waypoint 20 m', 'Waypoint 30 m', 'Waypoint 40 m']
TELEMETRY_VERSION = 2
DRIVING_DETAIL_FIELDS = ('target_speed_mps', 'route_completion', 'route_remaining_m',
                         'applied_action', 'reward_last', 'reward_components', 'reward_component_totals')


def telemetry_path(cfg):
    endpoint = f'{cfg.environment.host}:{cfg.environment.port}'
    key = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f'world-model-rl-{os.getuid()}-{key}.json'


class EpisodeTelemetry:
    def __init__(self, env, metadata, path=None):
        self.path = telemetry_path(env.cfg) if path is None else Path(path)
        self.actor_id = getattr(getattr(env, 'ego', None), 'id', None)
        self.metadata = metadata
        self.total = 0.0
        self.component_totals = {}
        self.warned = False

    def publish(self, observation, info, reward=None, attention=None, active=True):
        # Test/custom environments without an ego actor do not publish live CARLA data.
        if self.actor_id is None:
            return
        if reward is not None:
            self.total += float(reward)
            for name, value in info.get('components', {}).items():
                self.component_totals[name] = self.component_totals.get(name, 0.0) + float(value)
        features = observation.continuous[OBJECT_OFFSET:].reshape(SLOTS, len(OBJECT_FIELDS))
        objects = []
        labels = TOKEN_LABELS.copy()
        for slot in range(SLOTS):
            name = OBJECT_CLASSES[int(observation.classes[slot])]
            present = bool(observation.present[slot])
            labels.append(f'{name.title()} #{int(observation.ids[slot])}' if present else f'Empty slot {slot + 1}')
            if present:
                objects.append({'slot': slot, 'id': int(observation.ids[slot]), 'class': name,
                                'forward_m': float(features[slot, 0]), 'left_m': float(features[slot, 1]),
                                'confidence': float(features[slot, 6])})
        payload = {'version': TELEMETRY_VERSION, 'updated_at': time.time(), 'actor_id': self.actor_id,
                   'episode_id': self.metadata['episode_id'], 'controller': self.metadata['collector'],
                   'source': self.metadata.get('observation_source', 'unknown'),
                   'frame': int(info['frame']), 'active': active,
                   'speed_mps': float(np.linalg.norm(observation.continuous[:2])),
                   'target_speed_mps': float(observation.continuous[12]),
                   'route_completion': float(info.get('route_completion', 0.0)),
                   'route_remaining_m': float(observation.continuous[13]),
                   'applied_action': np.asarray(info.get('applied_action', observation.continuous[5:8])).tolist(),
                   'reward_last': None if reward is None else float(reward),
                   'reward_components': {name: float(value) for name, value in info.get('components', {}).items()},
                   'reward_component_totals': self.component_totals.copy(),
                   'episode_return': self.total, 'objects': objects,
                   'attention': None if attention is None else {'labels': labels, 'heads': attention}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, prefix=self.path.name + '.',
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, allow_nan=False)
            os.replace(temporary, self.path)
        except (OSError, ValueError) as exc:
            if not self.warned:
                warnings.warn(f'Viewer telemetry unavailable: {exc}', RuntimeWarning)
                self.warned = True
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def read_telemetry(path, actor_id, now=None):
    """Do not mix the current vehicle with old episodes or silently show stale state."""
    try:
        data = json.loads(Path(path).read_text())
        age = max(0.0, (time.time() if now is None else now) - data['updated_at'])
        if data['version'] not in (1, TELEMETRY_VERSION) or data['actor_id'] != actor_id:
            return None
        data['missing_fields'] = [field for field in DRIVING_DETAIL_FIELDS if field not in data]
        data['age_seconds'] = age
        data['stale'] = age > 5
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None

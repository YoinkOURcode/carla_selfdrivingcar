"""Expert labels are separate from executed actions; all existing replay semantics hold."""

import numpy as np
import pytest
from types import SimpleNamespace

from world_model_rl.control import apply_action
from world_model_rl.model import RSSM
from world_model_rl.runner import drive_episode
from world_model_rl.sac import Demonstrations
from world_model_rl.schema import Observation, OBS_KEYS, EVENTS, Normalizer
from world_model_rl.tests.test_pipeline import cfg, make_episode


@pytest.mark.parametrize("expert_probability", [0., 1.])
def test_labels_are_preserved_independently_of_executed_policy(cfg, expert_probability, tmp_path, monkeypatch):
    from world_model_rl import telemetry
    path = tmp_path / 'telemetry.json'
    monkeypatch.setattr(telemetry, 'telemetry_path', lambda cfg: path)
    identity = next(i for i in range(10) if make_episode(cfg, i).metadata['split'] == 'train')
    source = make_episode(cfg, identity)
    expert = np.array([.2, .6, 0], np.float32)
    policy = np.array([-.3, .2, 0], np.float32)
    class Env:
        def __init__(self):
            self.cfg = cfg
            self.metadata = source.metadata.copy()
            self.ego = SimpleNamespace(id=42)
        def observation(self):
            obs = Observation(**{key: source.arrays[key][self.index].copy() for key in OBS_KEYS})
            obs.continuous[5:8] = self.previous
            return obs
        def reset(self, *args):
            self.index = 0
            self.previous = np.zeros(3, np.float32)
            return self.observation(), {'frame': 0, 'sim_time': 0}
        def expert_action(self):
            return expert.copy()
        def finish_episode(self):
            pass
        def step(self, requested):
            self.previous = apply_action(requested, self.previous, cfg.environment)
            self.index += 1
            ended = self.index == len(source)
            info = {'frame': self.index * cfg.environment.action_repeat,
                'sim_time': self.index * cfg.environment.decision_dt, 'applied_action': self.previous,
                'events': {**dict.fromkeys(EVENTS, 0), 'success': int(ended)}, 'components': {'progress': 1.},
                'terminal_reason': 'success' if ended else 'running', 'route_completion': 1.,
                'distance_metres': .2, 'lane_offset': 0., 'heading_error': 0.,
                'acceleration_mps2': 0., 'jerk_mps3': 0., 'steering_rate_per_s': 0.}
            return self.observation(), 1., ended, False, info
    class Policy:
        updates = 0
        def act(self, *args):
            return policy[None]
    model = RSSM(cfg.model)
    norm = Normalizer.fit([source])
    episode, _ = drive_episode(Env(), identity, model=model, normalizer=norm,
        sac=Policy(), device='cpu', expert_probability=expert_probability)
    episode.validate()
    live = telemetry.read_telemetry(path, 42)
    assert live['episode_return'] == float(episode.arrays['rewards'].sum())
    assert not live['active']
    assert live['frame'] == len(source) * cfg.environment.action_repeat
    assert len(live['attention']['heads']) == cfg.model.attention_heads
    assert live['attention']['labels'][9] == 'Vehicle #37'
    np.testing.assert_equal(episode.arrays['expert_actions'], np.tile(expert, (len(source), 1)))
    executed = expert if expert_probability else policy
    np.testing.assert_equal(episode.arrays['requested_actions'], np.tile(executed, (len(source), 1)))
    demos = Demonstrations([episode], model, norm, 'cpu', 1)
    np.testing.assert_equal(demos.arrays['actions'], episode.arrays['expert_actions'])


def test_labelled_episodes_require_explicit_valid_expert_actions(cfg):
    ep = make_episode(cfg)
    ep.metadata['collector'] = 'dagger'
    with pytest.raises(ValueError, match='missing expert_actions'):
        ep.validate()
    ep.arrays['expert_actions'] = np.full((len(ep), 3), np.nan, np.float32)
    with pytest.raises(ValueError, match='Invalid expert_actions'):
        ep.validate()

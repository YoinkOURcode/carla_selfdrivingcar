"""A learned event head sees only sorted, visible current geometry and controls."""

import copy

import numpy as np
import pytest
import torch

from world_model_rl.data import sample_sequences
from world_model_rl.model import RSSM
from world_model_rl.policy import current_policy_observation, policy_feature_size
from world_model_rl.sac import SAC, real_replay
from world_model_rl.schema import Normalizer, OBJECT_OFFSET
from world_model_rl.training import imagine, load_checkpoint, save_checkpoint, termination_examples
from world_model_rl.tests.test_pipeline import cfg, make_episode


def full_config(cfg):
    cfg.model.deterministic, cfg.model.stochastic = 128, 32
    cfg.training.policy_observation_skip = cfg.training.termination_observation_skip = True
    cfg.validate()
    return cfg


def test_visible_geometry_is_sorted_and_absent_poison_does_not_enter_head(cfg):
    cfg = full_config(cfg)
    model = RSSM(cfg.model, termination_observation_skip=True)
    episode = make_episode(cfg)
    obs = {key: torch.as_tensor(episode.arrays[key][:2]).clone()
           for key in ('continuous', 'present', 'controls', 'light')}
    obs['present'][:, 1] = True
    objects = obs['continuous'][:, OBJECT_OFFSET:].reshape(2, 12, 7)
    objects[:, 1] = torch.tensor([3, 1, 0, 0, 2, 1, 1])
    objects[:, 2:] = float('nan')
    state, applied = model.initial(2, 'cpu'), torch.zeros(2, 3)
    first = model.termination_inputs(state, applied=applied, observation=obs)
    permutation = torch.tensor([1, 0, *range(2, 12)])
    changed = {**obs, 'continuous': obs['continuous'].clone(), 'present': obs['present'][:, permutation]}
    changed['continuous'][:, OBJECT_OFFSET:] = objects[:, permutation].flatten(1)
    second = model.termination_inputs(state, applied=applied, observation=changed)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.termination_physical_size == 125


def test_training_features_align_current_observation_and_real_action(cfg):
    cfg = full_config(cfg)
    episode = make_episode(cfg, terminal=True)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model, termination_observation_skip=True)
    inputs, _ = termination_examples(model, [episode], norm, cfg, 'cpu')
    context = episode.arrays['continuous'][:-1, :20]
    np.testing.assert_allclose(inputs[:, 35:55], np.sign(context) * np.log1p(abs(context)), atol=1e-6)
    np.testing.assert_array_equal(inputs[:, -3:], episode.arrays['applied_actions'])
    batch = sample_sequences([episode], 2, 4, np.random.default_rng(3))
    loss, _ = model.sequence_loss(batch, norm, 'cpu')
    loss.backward()
    assert torch.isfinite(loss) and model.termination_head[0].weight.grad is not None


def test_imagination_propagates_predicted_geometry_without_future_observation(cfg, monkeypatch):
    cfg = full_config(cfg)
    episode = make_episode(cfg)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model, termination_observation_skip=True)
    for parameter in model.termination_head.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(-100)
    replay = real_replay([episode], model, norm, 'cpu', 1, policy_skip=True)
    sac = SAC(policy_feature_size(model, cfg), cfg, 'cpu')
    cfg.training.imagination_horizon = 3
    sources = []
    original = model.termination_inputs
    def record(state, context=None, applied=None, observation=None):
        sources.append({key: value.clone() for key, value in observation.items()})
        return original(state, context, applied, observation)
    monkeypatch.setattr(model, 'termination_inputs', record)
    def forbidden(*args, **kwargs):
        raise AssertionError('Future CARLA observations are forbidden')
    monkeypatch.setattr(model, 'observe_initial', forbidden)
    generated, _ = imagine(model, sac, replay, norm, cfg, 'cpu', np.random.default_rng(37))
    n = cfg.training.imagination_starts
    assert len(sources) == 3 and len(generated) == n * 3
    for step, actual in enumerate(sources):
        expected = current_policy_observation(torch.tensor(generated.arrays['features'][step*n:(step+1)*n]), cfg, norm)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])


def test_checkpoint_contract_keeps_full_observation_distinct_from_other_heads(cfg, tmp_path):
    cfg = full_config(cfg)
    model, norm = RSSM(cfg.model, termination_observation_skip=True), Normalizer.fit([make_episode(cfg)])
    path = tmp_path / 'model.pt'
    save_checkpoint(path, cfg, model, norm, torch.optim.Adam(model.parameters()), 1, 20)
    _, restored, _, _ = load_checkpoint(path, cfg, 'cpu')
    assert restored.termination_observation_skip
    old = copy.deepcopy(cfg);old.training.termination_observation_skip = False
    with pytest.raises(ValueError, match='termination input contract'):
        load_checkpoint(path, old, 'cpu')

"""Known current context can condition learned events without future observations."""

import copy

import numpy as np
import pytest
import torch

from world_model_rl.model import RSSM
from world_model_rl.policy import policy_feature_size
from world_model_rl.sac import SAC, real_replay
from world_model_rl.schema import Normalizer
from world_model_rl.training import imagine, load_checkpoint, save_checkpoint, termination_examples
from world_model_rl.tests.test_pipeline import cfg, make_episode


def context_config(cfg):
    cfg.model.deterministic, cfg.model.stochastic = 32, 8
    cfg.training.policy_observation_skip = True
    cfg.training.termination_context_skip = True
    cfg.validate()
    return cfg


def test_head_training_uses_current_physical_context_and_real_applied_action(cfg):
    cfg = context_config(cfg)
    episode = make_episode(cfg, terminal=True)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model, True)
    inputs, _ = termination_examples(model, [episode], norm, cfg, 'cpu')
    context = episode.arrays['continuous'][:-1, :20]
    np.testing.assert_allclose(inputs[:, -23:-3], np.sign(context) * np.log1p(abs(context)), atol=1e-6)
    np.testing.assert_array_equal(inputs[:, -3:], episode.arrays['applied_actions'])
    assert inputs[0, -10] != inputs[1, -10]  # Distinct current route distances, not one final observation.
    with pytest.raises(ValueError, match='required'):
        model.termination_logits(model.initial(1, 'cpu'))


def test_imagined_context_starts_from_replay_then_only_model_predictions(cfg, monkeypatch):
    cfg = context_config(cfg)
    episode = make_episode(cfg)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model, True)
    for parameter in model.termination_head.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(-100)
    replay = real_replay([episode], model, norm, 'cpu', 1, policy_skip=True)
    sac = SAC(policy_feature_size(model, cfg), cfg, 'cpu')
    cfg.training.imagination_horizon = 3
    expected = replay.sample(cfg.training.imagination_starts, np.random.default_rng(51))['features']
    start = cfg.model.deterministic + cfg.model.stochastic + 4
    contexts = []
    original = model.termination_inputs
    def record(state, context=None, applied=None):
        contexts.append((context.detach().clone(), applied.detach().clone()))
        return original(state, context, applied)
    monkeypatch.setattr(model, 'termination_inputs', record)
    def forbidden(*args, **kwargs):
        raise AssertionError('Imagination must not observe a future real observation')
    monkeypatch.setattr(model, 'observe_initial', forbidden)
    generated, _ = imagine(model, sac, replay, norm, cfg, 'cpu', np.random.default_rng(51))
    first = expected[:, start:start + 20] * norm.scale[:20] + norm.mean[:20]
    np.testing.assert_allclose(contexts[0][0], first, atol=1e-6)
    n = cfg.training.imagination_starts
    for step, (context, applied) in enumerate(contexts):
        current = generated.arrays['features'][step * n:(step + 1) * n]
        physical = current[:, start:start + 20] * norm.scale[:20] + norm.mean[:20]
        np.testing.assert_allclose(context, physical, atol=1e-6)
        # Recorded source context retains the previous applied controls, not future real ones.
        np.testing.assert_allclose(physical[:, 5:8], current[:, -3:], atol=1e-6)
    assert len(contexts) == 3 and len(generated) == 3 * n


def test_checkpoint_refuses_silent_termination_input_contract_change(cfg, tmp_path):
    cfg = context_config(cfg)
    model = RSSM(cfg.model, True)
    norm = Normalizer.fit([make_episode(cfg)])
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / 'model.pt'
    save_checkpoint(path, cfg, model, norm, optimizer, 1, 20)
    _, restored, _, _ = load_checkpoint(path, cfg, 'cpu')
    assert restored.termination_context_skip
    old = copy.deepcopy(cfg);old.training.termination_context_skip = False
    with pytest.raises(ValueError, match='termination input contract'):
        load_checkpoint(path, old, 'cpu')

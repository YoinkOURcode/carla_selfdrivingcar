"""Explicit reward targets and training-only traffic labels preserve raw recordings."""

import numpy as np
import pytest
import torch

from world_model_rl.data import sample_sequences
from world_model_rl.model import RSSM
from world_model_rl.rewards import learning_rewards
from world_model_rl.sac import SAC, real_replay
from world_model_rl.schema import EVENTS, Normalizer
from world_model_rl.teachers import geometry_actions
from world_model_rl.training import load_checkpoint, save_checkpoint, evaluate_model, seed_everything
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_reward_targets_match_sequence_replay_and_validation_without_rewriting_raw_data(cfg):
    ep = make_episode(cfg, terminal=True)
    ep.arrays['events'][0, EVENTS.index('red_light')] = 1
    ep.arrays['events'][-1, EVENTS.index('collision')] = 1
    raw = ep.arrays['rewards'].copy()
    components = ep.arrays['components'].copy()
    cfg.training.red_light_objective_multiplier = 10
    cfg.training.collision_objective_multiplier = 5
    expected = raw.copy(); expected[0] -= 180; expected[-1] -= 400
    np.testing.assert_allclose(learning_rewards(ep.arrays, cfg), expected)
    batch = sample_sequences([ep], 2, len(ep), np.random.default_rng(1), reward_cfg=cfg)
    np.testing.assert_allclose(batch['rewards'], np.stack([expected] * 2))
    model, norm = RSSM(cfg.model), Normalizer.fit([ep])
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    replay = real_replay([ep], model, norm, 'cpu', 1, reward_cfg=cfg)
    np.testing.assert_allclose(replay.arrays['rewards'], expected)
    cfg.training.validation_branches = 100
    report = evaluate_model(model, [ep], norm, cfg, 'cpu')
    assert report['horizons']['1']['reward_sum_mae'] == pytest.approx(np.abs(expected).mean())
    np.testing.assert_array_equal(ep.arrays['rewards'], raw)
    np.testing.assert_array_equal(ep.arrays['components'], components)


def test_model_with_a_different_learning_reward_contract_cannot_be_substituted(cfg, tmp_path):
    ep = make_episode(cfg)
    model, norm = RSSM(cfg.model), Normalizer.fit([ep])
    path = tmp_path / 'model.pt'
    save_checkpoint(path, cfg, model, norm, torch.optim.Adam(model.parameters()), 1, 1)
    cfg.training.red_light_objective_multiplier = 10
    with pytest.raises(ValueError, match='learning reward objective differs'):
        load_checkpoint(path, cfg, 'cpu')


def test_v4_labels_brake_consistently_near_red_and_release_on_green(cfg):
    ep = make_episode(cfg, steps=3)
    ep.arrays['continuous'][:, 0] = [2, 0, 0, 0]
    ep.arrays['continuous'][:, 14] = [15, 5, 5, 5]
    ep.arrays['controls'][:, 0] = True
    ep.arrays['light'][:] = [2, 2, 4, 4]
    labels = geometry_actions(ep.arrays, calibrated=True, hold_stops=True, early_light_hold=True)
    assert labels[0, 1] == labels[1, 1] == 0
    assert labels[0, 2] >= .6 - 1e-6 and labels[1, 2] >= .35 - 1e-6
    assert labels[2, 1] > 0 and labels[2, 2] == 0


def test_applied_control_cloning_penalizes_throttle_brake_cancellation(cfg):
    sac = SAC(cfg.model.deterministic + cfg.model.stochastic + 7, cfg, 'cpu')
    features = torch.zeros(1, cfg.model.deterministic + cfg.model.stochastic + 7)
    chosen = torch.tensor([[0., .15, .1]], requires_grad=True)
    labels = torch.tensor([[0., 0., .6]])
    base = sac.cloning_loss(chosen, labels, features)
    cfg.training.behavior_cloning_applied_weight = 2
    loss = sac.cloning_loss(chosen, labels, features)
    assert loss > base
    loss.backward()
    assert chosen.grad[0, 1] > 0 and chosen.grad[0, 2] < 0


def test_explicit_determinism_repeats_draws_and_enables_strict_checks(monkeypatch):
    prior = torch.are_deterministic_algorithms_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    try:
        seed_everything(17, deterministic=True)
        first = torch.rand(5)
        seed_everything(17, deterministic=True)
        torch.testing.assert_close(torch.rand(5), first)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(prior)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic

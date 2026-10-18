"""Termination calibration regressions using generated arrays only, never CARLA."""

import json

import numpy as np
import pytest
import torch

from world_model_rl.data import Dataset, sample_sequences, write_json
from world_model_rl.model import RSSM
from world_model_rl.sac import SAC, real_replay
from world_model_rl.schema import EVENTS, Normalizer
from world_model_rl.training import (ImaginationGateError, fit_termination_updates, imagine,
    evaluate_model, load_checkpoint, save_checkpoint, termination_examples, termination_metrics)
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_ending_sampler_covers_success_and_failure_but_not_truncation(cfg):
    success = make_episode(cfg, 0, terminal=True)
    success.arrays['events'][-1, EVENTS.index('success')] = 1
    failure = make_episode(cfg, 1, terminal=True)
    truncated = make_episode(cfg, 2)
    success.arrays['rewards'][-1] = 100
    failure.arrays['rewards'][-1] = -100
    truncated.arrays['rewards'][-1] = 777
    batch = sample_sequences([success, failure, truncated], 64, 3, np.random.default_rng(0), failure_fraction=1)
    assert batch['terminated'][:, -1].all()
    assert set(batch['rewards'][:, -1]) == {-100, 100}


def test_class_prior_offset_restores_natural_frequency_and_survives_checkpoint(cfg, tmp_path):
    model = RSSM(cfg.model)
    for parameter in model.termination_head.parameters():
        torch.nn.init.zeros_(parameter)
    prior = 0.01
    model.termination_logit_offset.fill_(np.log(prior / (1 - prior)))
    state = model.initial(4, 'cpu')
    torch.testing.assert_close(model.termination_logits(state).sigmoid(), torch.full((4,), prior))
    norm = Normalizer.fit([make_episode(cfg)])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, cfg, model, norm, optimizer, 1, 12)
    _, restored, _, _ = load_checkpoint(path, cfg, 'cpu')
    torch.testing.assert_close(restored.termination_logits(state), model.termination_logits(state))
    legacy = torch.load(path, map_location='cpu', weights_only=False)
    legacy['model'].pop('termination_logit_offset')
    torch.save(legacy, tmp_path / 'older_format2.pt')
    _, older, _, _ = load_checkpoint(tmp_path / 'older_format2.pt', cfg, 'cpu')
    assert float(older.termination_logit_offset) == 0


def test_readout_fit_uses_full_history_prior_means_and_changes_only_ending_parameters(cfg):
    episodes = [make_episode(cfg, 0, terminal=True), make_episode(cfg, 1)]
    norm, model = Normalizer.fit(episodes), RSSM(cfg.model)
    features, labels = termination_examples(model, episodes, norm, cfg, 'cpu')
    assert features.shape == (12, cfg.model.deterministic + cfg.model.stochastic)
    np.testing.assert_equal(labels, np.concatenate([ep.arrays['terminated'] for ep in episodes]))
    encoded = model.encode_episode(episodes[0], norm, 'cpu')
    h, z = cfg.model.deterministic, cfg.model.stochastic
    with torch.no_grad():
        source = torch.tensor(encoded[4:5])
        state, _, _ = model.transition((source[:, :h], source[:, h:h + z]),
            torch.tensor(episodes[0].arrays['applied_actions'][4:5]), sample=False,
            weather=source[:, h + z:h + z + 4])
    np.testing.assert_allclose(features[4:5], torch.cat(state, -1).numpy(), atol=1e-6)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    normalizer_before = norm.to_dict()
    stats = fit_termination_updates(model, episodes, norm, cfg, 4, 'cpu', np.random.default_rng(1))
    assert stats['positive_count'] == 1 and stats['negative_count'] == 11
    assert stats['natural_training_prevalence'] == pytest.approx(1 / 12)
    assert stats['sampled_training_prevalence'] == 0.5
    assert float(model.termination_logit_offset) == pytest.approx(np.log(1 / 11))
    assert norm.to_dict() == normalizer_before
    changed = []
    for key, value in model.state_dict().items():
        if not torch.equal(value, before[key]):
            changed.append(key)
            assert key.startswith('termination_head.') or key == 'termination_logit_offset'
    assert any(key.startswith('termination_head.') for key in changed)


def test_imagination_uses_same_mean_latent_ending_readout_as_validation(cfg, monkeypatch):
    episode = make_episode(cfg)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(-100)
    real = real_replay([episode], model, norm, 'cpu', 1)
    sac = SAC(model.feature_size, cfg, 'cpu')
    captured = []
    original = model.termination_logits
    def record(state):
        captured.append(state[1].detach().clone())
        return original(state)
    monkeypatch.setattr(model, 'termination_logits', record)
    generated, _ = imagine(model, sac, real, norm, cfg, 'cpu', np.random.default_rng(0))
    assert captured and all(torch.count_nonzero(value) == 0 for value in captured)
    h, z = cfg.model.deterministic, cfg.model.stochastic
    assert np.any(generated.arrays['next_features'][:, h:h + z] != 0)


def test_confusion_counts_keep_false_positives_visible():
    result = termination_metrics([0.01, 0.8, 0.9, 0.2], [True, False, True, False])
    assert result['true_positive'] == result['false_positive'] == result['false_negative'] == result['true_negative'] == 1
    assert result['precision'] == result['recall'] == 0.5


def test_ending_diagnostics_keep_truncations_negative_and_rank_all_episode_scores(cfg, monkeypatch):
    ending, truncated = make_episode(cfg, 1, terminal=True), make_episode(cfg, 2)
    ending.metadata['terminal_reason'] = 'collision'
    model, norm = RSSM(cfg.model), Normalizer.fit([ending, truncated])
    def scores(state):
        logits = torch.full((len(state[0]),), 2.0)
        logits[-1] = -2.0
        return logits
    monkeypatch.setattr(model, 'termination_logits', scores)
    fit_stats = {'training_metrics': {'true_positive': 3}}
    report = evaluate_model(model, [ending, truncated], norm, cfg, 'cpu', termination_fit=fit_stats)
    diagnostics = report['termination']['endings']
    assert len(diagnostics) == 1
    assert diagnostics[0] == {'episode_id': 1, 'transition': 5, 'reason': 'collision',
        'weather': 'ClearNoon', 'probability': pytest.approx(float(torch.tensor(-2.0).sigmoid())),
        'visible_objects_before': 1, 'nonterminal_scores_at_or_above': 11}
    assert report['termination']['false_negative'] == 1
    assert report['termination']['false_positive'] == 10
    assert report['termination']['false_positive_group_count'] == 2
    assert [group['count'] for group in report['termination']['false_positive_groups']] == [5, 5]
    assert report['termination']['false_positive_groups'][0]['first_transition'] == 0
    assert report['termination']['false_positive_groups'][0]['last_transition'] == 4
    assert report['termination']['false_positive_groups'][0]['decision_seconds'] == pytest.approx(0.5)
    assert [item['false_positive'] for item in report['termination']['by_episode']] == [5, 5]
    assert report['termination_fit'] == fit_stats


def test_generalization_failure_does_not_repeat_head_fit_or_start_training(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    cfg.environment.max_steps = 24
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(10):
        episode = make_episode(cfg, identity, steps=24, terminal=True)
        episode.metadata['terminal_reason'] = 'collision'
        dataset.save(episode)
    norm, model = Normalizer.fit(dataset.load('train')), RSSM(cfg.model)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    # Saved-fit provenance is a fixture: this test checks recovery orchestration,
    # without optimizing a model to manufacture a generalization gap.
    fitted = {'training_metrics': {'true_positive': 8, 'positive_count': 8,
        'precision': 1.0, 'recall': 1.0, 'brier': 0.0}}
    source = tmp_path / 'model.pt'
    save_checkpoint(source, cfg, model, norm, optimizer, 1, 12, termination_fit=fitted)
    source_bytes = source.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('A failed gate must not fit, update SAC, or connect to CARLA')
    monkeypatch.setattr(runner, 'CarlaDrivingEnv', forbidden)
    monkeypatch.setattr(runner, 'fit_updates', forbidden)
    monkeypatch.setattr(runner, 'fit_termination_updates', forbidden)
    monkeypatch.setattr(SAC, 'update', forbidden)
    output = tmp_path / 'policy'
    with pytest.raises(ImaginationGateError) as stopped:
        runner.train(cfg, dataset.path, output, model_path=source, policy_updates=2)
    message = str(stopped.value)
    assert '8/8 training endings' in message and 'train a new world model from scratch' in message
    assert 'Use fit-termination' not in message
    assert source.read_bytes() == source_bytes and not (output / 'agent.pt').exists()
    report = json.loads((output / 'validation.json').read_text())
    assert report['termination_fit'] == fitted
    assert len(report['termination']['endings']) == 1


def test_precision_failure_points_to_false_alarm_context_without_fitting(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(10):
        dataset.save(make_episode(cfg, identity, steps=24, terminal=True))
    norm, model = Normalizer.fit(dataset.load('train')), RSSM(cfg.model)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(10)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    source = tmp_path / 'model.pt'
    # Provenance is an explicit fixture, independently of the constant-score model.
    save_checkpoint(source, cfg, model, norm, optimizer, 1, 12,
        termination_fit={'training_metrics': {'true_positive': 8, 'positive_count': 8,
            'precision': 1.0, 'recall': 1.0, 'brier': 0.0}})
    source_bytes = source.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Precision failure must not train or connect to CARLA')
    monkeypatch.setattr(runner, 'CarlaDrivingEnv', forbidden)
    monkeypatch.setattr(runner, 'fit_updates', forbidden)
    monkeypatch.setattr(SAC, 'update', forbidden)
    output = tmp_path / 'policy'
    with pytest.raises(ImaginationGateError) as stopped:
        runner.train(cfg, dataset.path, output, model_path=source, policy_updates=2)
    assert 'Recall passes, but false alarms reduce precision' in str(stopped.value)
    assert 'termination.false_positive_groups' in str(stopped.value)
    assert 'collision partner' not in str(stopped.value)
    report = json.loads((output / 'validation.json').read_text())
    assert report['termination']['true_positive'] == 1
    assert report['termination']['false_positive'] == 23
    group = report['termination']['false_positive_groups'][0]
    assert group['count'] == 23 and group['last_transition'] == 22
    assert source.read_bytes() == source_bytes and not (output / 'agent.pt').exists()


def test_false_alarm_details_are_bounded_without_changing_total_counts(cfg, monkeypatch):
    episode = make_episode(cfg, steps=65, terminal=True)
    model, norm = RSSM(cfg.model), Normalizer.fit([episode])
    def alternating_scores(state):
        return torch.where(torch.arange(len(state[0])) % 2 == 0, 2.0, -2.0)
    monkeypatch.setattr(model, 'termination_logits', alternating_scores)
    report = evaluate_model(model, [episode], norm, cfg, 'cpu')['termination']
    assert report['true_positive'] == 1 and report['false_positive'] == 32
    assert report['false_positive_group_count'] == 32
    assert len(report['false_positive_groups']) == report['false_positive_groups_limit'] == 20
    assert report['by_episode'][0]['false_positive'] == 32


def test_head_only_recovery_preserves_original_checkpoint_and_dynamics(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(10):
        dataset.save(make_episode(cfg, identity, terminal=identity % 2 == 0))
    model = RSSM(cfg.model)
    norm = Normalizer.fit(dataset.load('train'))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    source, target = tmp_path / 'world_model.pt', tmp_path / 'termination_refit.pt'
    save_checkpoint(source, cfg, model, norm, optimizer, 5, 23400, kind='world_model')
    source_bytes = source.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Head-only recovery must not run dynamics fitting or connect to CARLA')
    monkeypatch.setattr(runner, 'fit_updates', forbidden)
    monkeypatch.setattr(runner, 'CarlaDrivingEnv', forbidden)
    report = runner.fit_termination(cfg, dataset.path, source, target, updates=2)
    assert report['available'] and source.read_bytes() == source_bytes
    payload, restored, restored_norm, _ = load_checkpoint(target, cfg, 'cpu')
    assert payload['model_updates'] == 23400 and payload['termination_updates'] == 2
    assert payload['model_version'] == 6 and 'sac' not in payload
    assert report['termination_fit'] == payload['termination_fit']
    assert restored_norm.to_dict() == norm.to_dict()
    for name, value in restored.state_dict().items():
        if not name.startswith('termination_head.') and name != 'termination_logit_offset':
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        runner.fit_termination(cfg, dataset.path, source, target, updates=2)


def test_quality_stop_has_clean_cli_exit_but_code_errors_keep_tracebacks(cfg, tmp_path, monkeypatch, capsys):
    from world_model_rl import runner
    from world_model_rl.__main__ import main
    path = tmp_path / 'config.json'
    write_json(path, cfg.to_dict())
    def quality_stop(*args):
        raise ImaginationGateError('held-out termination recall failed; checkpoint preserved')
    monkeypatch.setattr(runner, 'run', quality_stop)
    with pytest.raises(SystemExit) as stopped:
        main(['run', '--config', str(path), '--output', str(tmp_path / 'run'), '--policy-updates', '1'])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert 'checkpoint preserved' in error and 'Traceback' not in error
    def implementation_error(*args):
        raise RuntimeError('unrelated implementation error')
    monkeypatch.setattr(runner, 'run', implementation_error)
    with pytest.raises(RuntimeError, match='implementation error'):
        main(['run', '--config', str(path), '--output', str(tmp_path / 'run'), '--policy-updates', '1'])

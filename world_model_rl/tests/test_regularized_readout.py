"""Training-only event recovery and exact plain-MLP serialization invariants."""

import json

import numpy as np
import pytest
import torch

from world_model_rl.data import Dataset, write_json
from world_model_rl.deadline import TrainingClock
from world_model_rl.model import RSSM, mlp
from world_model_rl.refit_termination import install_readout, internal_partition
from world_model_rl.schema import Normalizer
from world_model_rl.training import load_checkpoint, save_checkpoint
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_smaller_standardized_readout_serializes_exactly_for_arbitrary_inputs():
    torch.manual_seed(37)
    source, destination = mlp(5, 1, 8), mlp(5, 1, 32)
    center, scale = torch.randn(5), torch.rand(5) + 0.1
    inputs = torch.randn(200, 5) * 100
    expected = source((inputs - center) / scale)
    install_readout(destination, source, center, scale)
    torch.testing.assert_close(destination(inputs), expected, rtol=1e-5, atol=1e-5)


def test_internal_partition_preserves_episode_boundaries_and_refuses_external_data(cfg):
    episodes = []
    for identity in range(20):
        episode = make_episode(cfg, identity, terminal=identity % 2 == 0)
        episode.metadata.update(split='train', terminal_reason='success' if identity % 2 == 0 else 'time_limit')
        episodes.append(episode)
    mask, indices = internal_partition(episodes, 51)
    assert len(mask) == sum(map(len, episodes)) and len(indices) == 4
    for index, chunk in enumerate(np.split(mask, np.cumsum([len(ep) for ep in episodes])[:-1])):
        assert np.all(chunk == (index in indices))
    episodes[0].metadata['split'] = 'validation'
    with pytest.raises(ValueError, match='training episodes only'):
        internal_partition(episodes, 51)


def test_recovery_keeps_body_fixed_and_commits_selection_before_external_validation(cfg, tmp_path, monkeypatch):
    from world_model_rl import refit_termination as recovery
    cfg.training.device = 'cuda'
    write_json(tmp_path / 'config.json', cfg.to_dict())
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(30):
        dataset.save(make_episode(cfg, identity, terminal=identity % 2 == 0))
    original, norm = RSSM(cfg.model), Normalizer.fit(dataset.load('train'))
    optimizer = torch.optim.Adam(original.parameters(), lr=cfg.model.learning_rate)
    save_checkpoint(tmp_path / 'world_model.pt', cfg, original, norm, optimizer, 1, 27)
    write_json(tmp_path / 'world_model.validation.json', {'available': False})
    clock = TrainingClock(tmp_path / 'training_clock.json');clock.start();clock.close()
    monkeypatch.setattr(recovery, 'CANDIDATES', ({'hidden': 8, 'weight_decay': .001, 'noise': .05},))
    monkeypatch.setattr(recovery, 'COUNTS', (2, 4))
    real_evaluate = recovery.evaluate_model
    def evaluate_after_selection(*args, **kwargs):
        assert (tmp_path / 'termination_refit/selected_before_external_validation.json').exists()
        return real_evaluate(*args, **kwargs)
    monkeypatch.setattr(recovery, 'evaluate_model', evaluate_after_selection)
    recovery.run(tmp_path, device_override='cpu')
    result, restored, restored_norm, _ = load_checkpoint(tmp_path / 'termination_refit/world_model_refit.pt', cfg, 'cpu')
    assert result['model_updates'] == 27 and result['termination_updates'] in (6, 8)
    assert result['config']['training']['device'] == 'cpu'
    assert json.loads((tmp_path / 'termination_refit/source_config.json').read_text())['training']['device'] == 'cuda'
    assert restored_norm.to_dict() == norm.to_dict()
    for name, value in original.state_dict().items():
        if not name.startswith('termination_head.') and name != 'termination_logit_offset':
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    plan = json.loads((tmp_path / 'termination_refit/plan.json').read_text())
    assert set(plan['fit_episode_ids'] + plan['holdout_episode_ids']) == {ep.metadata['episode_id'] for ep in dataset.load('train')}
    with pytest.raises(FileExistsError):
        recovery.run(tmp_path)

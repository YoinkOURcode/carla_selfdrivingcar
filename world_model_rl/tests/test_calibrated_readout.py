"""The deployed ensemble is exact and calibration never reads held-out labels."""

import shutil

import numpy as np
import pytest
import torch

from world_model_rl.calibrate_termination import episode_folds, install_ensemble
from world_model_rl.data import Dataset, write_json
from world_model_rl.deadline import TrainingClock
from world_model_rl.model import RSSM, mlp
from world_model_rl.schema import Normalizer
from world_model_rl.training import load_checkpoint, save_checkpoint
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_ensemble_folding_preserves_arbitrary_neural_predictions():
    torch.manual_seed(87)
    members = [(mlp(7, 1, 8), torch.randn(7), torch.rand(7) + .1) for _ in range(5)]
    destination = mlp(7, 1, 64)
    inputs = torch.randn(500, 7) * 100
    expected = torch.stack([head((inputs - center) / scale) for head, center, scale in members]).mean(0)
    install_ensemble(destination, members)
    torch.testing.assert_close(destination(inputs), expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match='exceeds'):
        install_ensemble(mlp(7, 1, 32), members)


def test_cross_folds_have_one_assignment_per_whole_training_episode(cfg):
    episodes = [make_episode(cfg, i) for i in range(25)]
    for episode in episodes:
        episode.metadata['split'] = 'train'
    masks, assigned = episode_folds(episodes, 5, 43)
    assert set(assigned) == set(range(5))
    for i, chunk in enumerate(np.split(masks, np.cumsum([len(ep) for ep in episodes])[:-1])):
        assert np.all(chunk == assigned[i])
    episodes[-1].metadata['split'] = 'test'
    with pytest.raises(ValueError, match='training episodes only'):
        episode_folds(episodes, 5, 43)


def test_calibration_committed_before_external_gate_and_preserves_frozen_body(cfg, tmp_path, monkeypatch):
    from world_model_rl import calibrate_termination as recovery
    cfg.training.device = 'cpu'
    write_json(tmp_path / 'config.json', cfg.to_dict())
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(30):
        dataset.save(make_episode(cfg, identity, terminal=identity % 2 == 0))
    original, norm = RSSM(cfg.model), Normalizer.fit(dataset.load('train'))
    optimizer = torch.optim.Adam(original.parameters(), lr=cfg.model.learning_rate)
    save_checkpoint(tmp_path / 'world_model.pt', cfg, original, norm, optimizer, 1, 27)
    old = tmp_path / 'termination_refit';old.mkdir()
    shutil.copyfile(tmp_path / 'world_model.pt', old / 'baseline.pt')
    write_json(old / 'selected_before_external_validation.json',
               {'spec': {'hidden': 2, 'noise': .05, 'weight_decay': .0001}, 'updates': 2})
    clock = TrainingClock(tmp_path / 'training_clock.json');clock.start();clock.close()
    real_evaluate = recovery.evaluate_model
    def evaluate_after_calibration(model, episodes, *args, **kwargs):
        assert (tmp_path / 'termination_calibration/calibration_before_external_validation.json').exists()
        assert all(ep.metadata['split'] == 'validation' for ep in episodes)
        return real_evaluate(model, episodes, *args, **kwargs)
    monkeypatch.setattr(recovery, 'evaluate_model', evaluate_after_calibration)
    recovery.run(tmp_path)
    result, restored, restored_norm, _ = load_checkpoint(tmp_path / 'termination_calibration/world_model_calibrated.pt', cfg, 'cpu')
    assert result['model_updates'] == 27 and result['termination_updates'] == 1010
    assert restored_norm.to_dict() == norm.to_dict()
    for name, value in original.state_dict().items():
        if not name.startswith('termination_head.') and name != 'termination_logit_offset':
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        recovery.run(tmp_path)

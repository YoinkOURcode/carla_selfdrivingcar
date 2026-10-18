"""Missing task endings must be detected before expensive fitting."""

import json
from pathlib import Path
import subprocess

import pytest

from world_model_rl import runner
from world_model_rl.__main__ import main
from world_model_rl.data import Dataset, write_json
from world_model_rl.tests.test_pipeline import cfg, make_episode
from world_model_rl.training import imagination_gate


def dataset_without_validation_endings(cfg, tmp_path):
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(10):
        episode = make_episode(cfg, identity, steps=24, terminal=True)
        if episode.metadata['split'] == 'validation':
            episode.arrays['terminated'][-1] = False
            episode.arrays['truncated'][-1] = True
        dataset.save(episode)
    return dataset


def test_missing_validation_endings_stop_before_model_allocation(cfg, tmp_path, monkeypatch):
    dataset = dataset_without_validation_endings(cfg, tmp_path)
    originals = {p: p.read_bytes() for p in dataset.path.glob('episode_*')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing label coverage must stop before device selection, model allocation or gradient updates')
    monkeypatch.setattr(runner, 'choose_device', forbidden)
    monkeypatch.setattr(runner, 'RSSM', forbidden)
    monkeypatch.setattr(runner, 'fit_updates', forbidden)
    target = tmp_path / 'new_model.pt'
    with pytest.raises(runner.DataCoverageError, match='validation has 0 task endings'):
        runner.fit(cfg, dataset.path, target, updates=10000)
    report = json.loads(target.with_suffix('.coverage.json').read_text())
    assert report['validation']['positive_count'] == 0
    assert report['validation']['negative_count'] == 24
    assert report['validation']['truncated_count'] == 1
    assert not report['ready'] and not target.exists()
    assert not (tmp_path / 'world_model_training.jsonl').exists()
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_coverage_uses_fixed_validation_suite_without_selecting_favorable_endings(cfg):
    episodes = [make_episode(cfg, 8), make_episode(cfg, 14, terminal=True)]
    cfg.training.validation_episodes = 1
    selected = runner.validation_suite(episodes, cfg)
    report = runner.fitting_coverage(cfg, [make_episode(cfg, 0, terminal=True)], selected)
    assert report['validation']['episode_ids'] == [8]
    assert report['validation']['positive_count'] == 0
    assert not report['ready']


def test_coverage_requires_training_endings_even_if_validation_has_them(cfg):
    report = runner.fitting_coverage(cfg, [make_episode(cfg)], [make_episode(cfg, 8, steps=24, terminal=True)])
    assert not report['ready']
    assert any('train has 0 task endings' in reason for reason in report['reasons'])


def test_validate_for_training_returns_clean_failure_and_preserves_splits(cfg, tmp_path, capsys):
    dataset = dataset_without_validation_endings(cfg, tmp_path)
    config = tmp_path / 'config.json'
    write_json(config, cfg.to_dict())
    splits = (dataset.path / 'splits.json').read_bytes()
    with pytest.raises(SystemExit) as stopped:
        main(['validate', '--for-training', '--config', str(config), '--data', str(dataset.path)])
    assert stopped.value.code == 2
    output = capsys.readouterr()
    assert 'validation has 0 task endings' in output.err and 'Traceback' not in output.err
    assert (dataset.path / 'splits.json').read_bytes() == splits


def test_zero_endings_are_reported_as_missing_support_not_failed_recall(cfg):
    report = {'available': True, 'horizons': {'1': {'normalized_rmse': 0, 'reward_sum_mae': 0,
        'heading_angle_mae_rad': 0, 'context_mae': {'forward_speed_mps': 0, 'lane_offset_left_m': 0}}},
        'termination': {'positive_count': 0, 'negative_count': 3600, 'brier': 0,
                        'recall': None, 'precision': None}}
    passed, reason = imagination_gate(report, cfg)
    assert not passed and reason == 'insufficient terminal/nonterminal labels'
    report['termination'].update(positive_count=1, recall=0)
    passed, reason = imagination_gate(report, cfg)
    assert not passed and 'termination recall tolerance failed' in reason
    assert 'termination precision tolerance failed' in reason


@pytest.mark.parametrize('validation_code', [0, 2])
def test_recovery_script_checks_coverage_before_fresh_fitting(tmp_path, validation_code):
    project = tmp_path / 'project with spaces'
    package = project / 'world_model_rl'
    package.mkdir(parents=True)
    script = package / 'recover_from_scratch.sh'
    script.write_text((Path(__file__).resolve().parents[1] / script.name).read_text())
    source = project / 'runs' / 'source'
    (source / 'data').mkdir(parents=True)
    (source / 'data' / 'dataset.json').write_text('{}')
    (source / 'config.json').write_text('{}')
    (source / 'world_model.pt').write_bytes(b'keep the original checkpoint')
    interpreter = project / 'venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n' +
                           f'if [[ "$4" == validate ]]; then exit {validation_code}; fi\nexit 0\n')
    interpreter.chmod(0o755)
    result = subprocess.run(['bash', str(script), 'runs/source', '20', '7', 'runs/new recovery'],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == validation_code
    log = (project / 'runs' / 'new recovery.log').read_text()
    assert log.index('\ncollect\n') < log.index('\nvalidate\n') and '--for-training' in log
    if validation_code:
        assert '\nfit\n' not in log and '\ntrain\n' not in log
    else:
        assert log.index('\nvalidate\n') < log.index('\nfit\n') < log.index('\ntrain\n')
        assert '--model\nruns/new recovery/world_model.pt\n' in log
    assert '--resume' not in log and '--warm-start' not in log
    assert (source / 'world_model.pt').read_bytes() == b'keep the original checkpoint'

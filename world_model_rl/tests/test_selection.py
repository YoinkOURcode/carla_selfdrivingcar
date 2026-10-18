"""Fixed-time candidate selection never uses final outcomes or partial reports."""

import pytest

from world_model_rl.config import Config
from world_model_rl.evaluation import evaluation_plan, evaluation_metrics
from world_model_rl.select_checkpoint import scheduled_candidates, selection_score, verified_result, file_hash, finish
from world_model_rl.tests.test_evaluation import results
from world_model_rl.tests.test_pipeline import cfg


@pytest.mark.parametrize('config_dir', ['.', 'config'])
def test_post_training_loads_config_and_both_frozen_plans(tmp_path, monkeypatch, config_dir):
    from world_model_rl import select_checkpoint
    from world_model_rl.data import write_json

    directory = tmp_path / config_dir
    cfg = Config.load()
    write_json(directory / 'config.json', cfg.to_dict())
    for kind, episodes in [('development', 10), ('final', 50)]:
        write_json(directory / f'{kind}_plan.json', evaluation_plan(cfg, episodes))
    write_json(directory / 'selection_plan.json', {
        'training_seconds': 18000, 'checkpoint_seconds': 300,
        'ranking_descending': ['arrivals_without_recorded_violations', 'successes',
            'negative_collision_episodes', 'negative_red_light_violations',
            'mean_route_completion', 'negative_archive_index'],
        **{f'{kind}_plan_sha256': file_hash(directory / f'{kind}_plan.json')
           for kind in ('development', 'final')},
    })
    write_json(tmp_path / 'training_clock.json', {
        'format': 1, 'seconds': 18000, 'checkpoint_seconds': 300,
        'started_unix': 0, 'deadline_unix': 18000, 'checkpoints': [],
    })
    class InputsVerified(Exception):
        pass
    def stop_before_checkpoint_changes(*args):
        raise InputsVerified
    monkeypatch.setattr(select_checkpoint, 'save_deadline_world_model', stop_before_checkpoint_changes)
    with pytest.raises(InputsVerified):
        finish(tmp_path)
    assert not (tmp_path / 'agent.pt').exists()


def test_post_training_workflow_cannot_start_a_training_clock(tmp_path):
    with pytest.raises(RuntimeError, match='has not started'):
        finish(tmp_path)
    assert not (tmp_path / 'training_clock.json').exists()


def test_schedule_requires_real_sac_updates_and_includes_final_stop():
    records = [
        {'source': 'agent.pt', 'stage': 'initial_policy', 'saved_unix': 101, 'warmup_updates': 0},
        {'source': 'agent.pt', 'stage': 'warmup_complete', 'saved_unix': 110, 'warmup_updates': 50},
        {'source': 'agent.pt', 'stage': 'first_sac', 'saved_unix': 111, 'policy_updates': 1, 'warmup_updates': 50},
        {'source': 'world_model.pt', 'stage': 'validated_model', 'saved_unix': 120},
        {'source': 'agent.pt', 'stage': 'sac', 'saved_unix': 129, 'policy_updates': 10},
        {'source': 'agent.pt', 'stage': 'sac', 'saved_unix': 155, 'policy_updates': 20},
        {'source': 'agent.pt', 'stage': 'deadline', 'saved_unix': 161, 'policy_updates': 21}]
    clock = {'started_unix': 100, 'seconds': 60, 'checkpoints': records}
    plan = {'candidate_schedule_seconds': [30, 60], 'last_cutoff_uses_final_deadline_snapshot': True}
    assert [c['archive_index'] for c in scheduled_candidates(clock, plan)] == [2, 4, 6]


def test_selection_counts_violation_free_arrivals_and_refuses_partial_evidence():
    cfg = Config(); plan = evaluation_plan(cfg, 10)
    clean = results(plan, successes=5, collisions=0)
    more = results(plan, successes=7, collisions=0)
    for item in more:item['event_decision_counts']['red_light'] = 1
    a, b = evaluation_metrics(clean, plan, cfg), evaluation_metrics(more, plan, cfg)
    assert a['arrivals_without_recorded_violations'] == 5
    assert b['arrivals_without_recorded_violations'] == 0
    assert selection_score(a, 2) > selection_score(b, 1)
    with pytest.raises(ValueError, match='Incomplete'):
        selection_score(evaluation_metrics(clean[:-1], plan, cfg), 0)


def test_existing_result_requires_full_plan_and_exact_checkpoint_provenance(tmp_path):
    cfg = Config(); plan = evaluation_plan(cfg, 10)
    path = tmp_path / 'candidate.pt';path.write_bytes(b'fixed archive')
    report = {'plan': plan, 'config': cfg.to_dict(), 'checkpoint_sha256': file_hash(path),
              'episodes': results(plan, successes=8, collisions=1)}
    assert verified_result(report, plan, cfg, path)['successes'] == 8
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        verified_result(report, plan, cfg, path)


def test_deadline_save_preserves_frozen_model_tensors_and_archives_them(cfg, tmp_path, monkeypatch):
    import torch
    from world_model_rl.tests.test_fixes import build_ready_model
    from world_model_rl.tests.test_deadline import controlled_clock
    from world_model_rl.training import load_checkpoint, save_checkpoint
    from world_model_rl.select_checkpoint import save_deadline_world_model
    _, path = build_ready_model(cfg, tmp_path)
    _, model, norm, optimizer = load_checkpoint(path, cfg, 'cpu')
    save_checkpoint(tmp_path/'agent.pt', cfg, model, norm, optimizer, 1, 1, kind='agent')
    expected = {k:v.clone() for k,v in model.state_dict().items()}
    clock, now = controlled_clock(tmp_path, monkeypatch)
    clock.start();now[0] += 11
    save_deadline_world_model(tmp_path, clock)
    payload, restored, _, _ = load_checkpoint(path, cfg, 'cpu')
    assert payload['deadline_snapshot'] and payload['training_clock']['deadline_reached']
    for key,value in restored.state_dict().items():torch.testing.assert_close(value,expected[key])
    assert clock.value['checkpoints'][-1]['stage']=='deadline_world_model'
    assert clock.value['checkpoints'][-1]['sha256']==file_hash(path)
    clock.close()

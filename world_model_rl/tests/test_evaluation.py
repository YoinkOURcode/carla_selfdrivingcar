"""Acceptance uses complete fixed scenarios and collision episodes, not event totals."""

import pytest

from world_model_rl.config import Config
from world_model_rl.evaluation import evaluation_plan, validate_plan, evaluation_metrics


def results(plan, successes=40, collisions=5):
    return [{**item, "success": i < successes,
             "terminal_reason": "success" if i < successes else "collision" if i < successes + collisions else "time_limit",
             "event_decision_counts": {"collision": 3 if successes <= i < successes + collisions else 0},
             "return": 0., "route_completion": 1. if i < successes else 0.5}
            for i, item in enumerate(plan["scenarios"])]


def test_fixed_balanced_suite_and_exact_acceptance_boundary():
    cfg = Config()
    plan = evaluation_plan(cfg)
    report = evaluation_metrics(results(plan), plan, cfg)
    assert report["prototype_acceptance"] == {"eligible": True, "required_success_rate": .8,
        "maximum_collision_episode_rate": .1, "passed": True}
    assert report["collision_episodes"] == 5  # Fifteen events still mean five collision episodes.
    assert all(item["episodes"] == 10 for item in report["per_weather"].values())
    assert evaluation_plan(cfg) == plan
    assert not evaluation_metrics(results(plan, 39), plan, cfg)["prototype_acceptance"]["passed"]
    assert not evaluation_metrics(results(plan, 40, 6), plan, cfg)["prototype_acceptance"]["passed"]


def test_partial_or_small_suite_cannot_establish_acceptance():
    cfg = Config()
    plan = evaluation_plan(cfg)
    report = evaluation_metrics(results(plan)[:49], plan, cfg)
    assert not report["complete"] and not report["prototype_acceptance"]["eligible"]
    small = evaluation_plan(cfg, 5)
    assert not evaluation_metrics(results(small, 5, 0), small, cfg)["prototype_acceptance"]["passed"]


def test_red_light_metrics_include_successful_arrivals_and_each_weather():
    cfg = Config()
    plan = evaluation_plan(cfg)
    driven = results(plan)
    driven[0]['event_decision_counts']['red_light'] = 2
    driven[5]['event_decision_counts']['red_light'] = 1
    driven[1]['event_decision_counts']['red_light'] = 1
    report = evaluation_metrics(driven, plan, cfg)
    assert report['red_light_violations'] == 4
    assert report['red_light_episodes'] == 3
    assert report['red_light_episode_rate'] == .06
    assert report['per_weather']['ClearNoon']['red_light_violations'] == 3
    assert report['per_weather']['ClearNoon']['red_light_episode_rate'] == .2
    assert report['per_weather']['FogNight']['red_light_episodes'] == 0


def test_plan_and_driven_results_must_match():
    cfg = Config()
    plan = evaluation_plan(cfg)
    driven = results(plan)
    driven[-1]["seed"] = driven[0]["seed"]
    with pytest.raises(ValueError, match="fixed scenario order"):
        evaluation_metrics(driven, plan, cfg)
    plan["scenarios"][0]["weather"] = "FogNight"
    with pytest.raises(ValueError, match="weather"):
        validate_plan(plan, cfg)


def test_red_crossing_totals_preserve_multiple_crossings_in_one_decision():
    cfg = Config(); plan = evaluation_plan(cfg)
    driven = results(plan)
    driven[0]['event_decision_counts']['red_light'] = 1
    driven[0]['event_totals'] = {'red_light': 3}
    report = evaluation_metrics(driven, plan, cfg)
    assert report['red_light_violations'] == 3
    assert report['red_light_episodes'] == 1 and report['red_light_episode_rate'] == .02
    assert report['arrivals_without_recorded_violations'] == 39
    assert report['per_weather']['ClearNoon']['red_light_violations'] == 3
    assert report['prototype_acceptance']['passed']  # Arrival/collision gates are unchanged.


def test_evaluation_traces_cannot_be_saved_or_loaded_as_training(tmp_path):
    import json
    from world_model_rl.data import Dataset
    from world_model_rl.evaluation import save_evaluation_trace
    from world_model_rl.tests.test_pipeline import make_episode

    cfg = Config()
    ep = make_episode(cfg)
    traces = tmp_path / 'traces'
    save_evaluation_trace(ep, traces)
    meta = json.loads((traces / 'episode_000000.json').read_text())
    assert meta['split'] == 'evaluation' and meta['usage'] == 'evaluation_only_never_train'
    train = Dataset(tmp_path / 'training', cfg, create=True)
    with pytest.raises(ValueError, match='never be saved'):
        train.save(ep)
    # Even creating a compatible manifest around trace files cannot admit them to replay.
    wrapped = Dataset(traces, cfg, create=True)
    with pytest.raises(ValueError, match='never enter training replay'):
        wrapped.load()
    with pytest.raises(FileExistsError):
        save_evaluation_trace(ep, traces)

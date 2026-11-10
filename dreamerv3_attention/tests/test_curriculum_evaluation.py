"""Curriculum evaluation and viewer drives use only their selected scenario."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import carla
import pytest

from dreamerv3_attention import __main__ as cli
from dreamerv3_attention import environment, training
from dreamerv3_attention.config import Config
from dreamerv3_attention.evaluation import collection_scenarios, evaluate, scenario_suite
from dreamerv3_attention.tests.test_environment import spawn_test_environment
from dreamerv3_attention.tests.test_viewer import arguments
from dreamerv3_attention.viewer import Viewer


ROOT = Path("runs/experiment_26/curriculum")
EVALUATIONS = sorted(path.stem for path in ROOT.glob("evaluate_*.json"))


def spawned_kinds(scenario):
    env = spawn_test_environment(7, lambda bp, _pose: NS(type_id=bp.id))
    env.scenario = scenario
    env._spawn_hazards()
    return [actor.type_id for actor in env.actors]


class Agent:
    def initial(self):
        return None

    def act(self, observation, carry, deterministic=True):
        return [0., 0.], None


def evaluation_factory(records):
    class Environment:
        def __init__(self, cfg, scenario):
            records.append(scenario)

        def reset(self):
            return {}

        def step(self, action):
            return {}, 0., True, False, {"success": True, "applied_action": action}

        def close(self):
            pass

    return Environment


@pytest.mark.parametrize("name", EVALUATIONS)
@pytest.mark.parametrize("episodes", [None, 1])
def test_named_evaluation_and_inference_run_exactly_the_selected_case(name, episodes):
    cfg = Config.load(ROOT / (name + ".json"))
    records = []
    result = evaluate(cfg, Agent(), evaluation_factory(records), episodes=episodes)
    assert [record["name"] for record in records] == [cfg.scenario]
    assert [episode["name"] for episode in result["episodes"]] == [cfg.scenario]
    assert result["scope"] == "fixed_training_sanity"
    assert not records[0]["background_traffic"]
    kinds = spawned_kinds(records[0])
    if name == "evaluate_pedestrians":
        assert len(kinds) == 1 and kinds[0].startswith("walker.pedestrian.")
    elif name == "evaluate_pedestrians_vehicles":
        assert len(kinds) == 2
        assert sum(kind.startswith("vehicle.") for kind in kinds) == 1
        assert sum(kind.startswith("walker.pedestrian.") for kind in kinds) == 1
    else:
        assert kinds == []


def test_explicit_evaluation_list_does_not_expand_to_later_pedestrian_cases():
    cfg = Config.load(ROOT / "stage3_lights.json")
    records = []
    result = evaluate(cfg, Agent(), evaluation_factory(records))
    assert [record["name"] for record in records] == cfg.scenarios
    assert all(not record["actors"] and not record["background_traffic"] for record in records)
    assert len(result["episodes"]) == 4


def test_default_held_out_evaluation_keeps_the_full_suite():
    cfg = Config(split="held_out")
    records = []
    result = evaluate(cfg, Agent(), evaluation_factory(records))
    assert [record["name"] for record in records] == [record["name"] for record in scenario_suite(cfg)]
    assert len(records) > 1 and result["scope"] == "split_evaluation"


def test_wrong_split_named_case_errors_instead_of_running_an_empty_suite():
    cfg = replace(Config.load(ROOT / "evaluate_lights.json"), split="held_out")
    with pytest.raises(ValueError, match="absent from selected split"):
        evaluate(cfg, Agent(), lambda *_: pytest.fail("must validate before environment creation"))


def test_empty_default_evaluation_suite_errors(tmp_path):
    catalog = tmp_path / "empty.json"
    catalog.write_text("[]")
    with pytest.raises(ValueError, match="No evaluation scenarios"):
        evaluate(Config(scenario_file=str(catalog), split="held_out"), Agent())


@pytest.mark.parametrize("name", EVALUATIONS)
def test_viewer_passes_the_resolved_case_and_spawns_no_unrequested_npcs(monkeypatch, tmp_path, name):
    cfg = Config.load(ROOT / (name + ".json"))
    selected = collection_scenarios(cfg)[0]
    captured = []
    monkeypatch.setattr(training, "workload_lock", lambda *_a, **_kw: nullcontext())
    monkeypatch.setattr(training, "register_run", lambda *_a: None)
    monkeypatch.setattr(training, "seed_all", lambda *_a: None)
    monkeypatch.setattr(cli, "server", lambda *_a: nullcontext())
    monkeypatch.setattr(training.Trainer, "restore", lambda *_a, **_kw: NS(cfg=cfg, agent=object()))

    class Environment:
        def __init__(self, run_cfg, scenario=None):
            assert scenario is not None
            assert scenario == selected
            assert (run_cfg.route_start, run_cfg.route_end) == (scenario["route_start"], scenario["route_end"])
            assert run_cfg.seed == scenario["seed"]
            captured.append(spawned_kinds(scenario))
            self.ego = NS(is_alive=True, apply_control=lambda *_: None)
            self.carla = carla

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(environment, "CarlaEnv", Environment)
    viewer = Viewer(cfg, arguments(output=str(tmp_path / "viewer")))
    monkeypatch.setattr(viewer, "_episode", lambda *_a: viewer.update(status="arrived"))
    viewer.drive(2, 26)  # Picker inputs must not replace the named curriculum task.
    assert viewer.read()["status"] == "arrived"
    assert viewer.read()["scenario"] == cfg.scenario
    if name == "evaluate_pedestrians":
        assert len(captured[0]) == 1 and captured[0][0].startswith("walker.pedestrian.")
    elif name == "evaluate_pedestrians_vehicles":
        assert len(captured[0]) == 2
        assert sum(kind.startswith("vehicle.") for kind in captured[0]) == 1
        assert sum(kind.startswith("walker.pedestrian.") for kind in captured[0]) == 1
    else:
        assert captured == [[]]


def test_viewer_rejects_ambiguous_or_wrong_split_before_starting_carla(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "workload_lock", lambda *_a, **_kw: pytest.fail("must validate first"))
    for cfg in (Config.load(ROOT / "stage4_pedestrians.json"),
                replace(Config.load(ROOT / "evaluate_lights.json"), split="held_out")):
        viewer = Viewer(cfg, arguments(output=str(tmp_path / "unused")))
        viewer.drive(0, 10)
        assert viewer.read()["status"] == "error"
        assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("command", ["evaluate", "infer"])
def test_cli_named_training_case_needs_no_extra_selection_flag(monkeypatch, tmp_path, command):
    cfg = replace(Config.load(ROOT / "evaluate_lights.json"), device="cpu")
    path = tmp_path / "config.json"
    cfg.save(path)
    monkeypatch.setattr(cli, "workload_lock", lambda *_a, **_kw: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *_a: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *_a: None)
    monkeypatch.setattr(cli.Trainer, "restore", lambda *_a, **_kw: NS(cfg=cfg, agent=Agent()))

    def evaluator(run_cfg, agent, factory, **kwargs):
        assert run_cfg.scenario == cfg.scenario
        assert kwargs["fixed_scenario"] is True
        return {"episodes": [], "summary": {"episode_count": 0}}

    monkeypatch.setattr(cli, "evaluate", evaluator)
    cli.main([command, "--config", str(path), "--checkpoint", "source.pt", "--budget-seconds", "10",
              "--output", str(tmp_path / "run")])


def test_stage4_has_vehicles_only_in_the_mixed_training_case():
    records = collection_scenarios(Config.load(ROOT / "stage4_pedestrians.json"))
    assert len(records) == 6
    for record in records:
        kinds = spawned_kinds(record)
        assert not record["background_traffic"]
        mixed = record["name"] == "train_pedestrians_vehicles"
        pedestrian = record["name"] == "train_pedestrian_awareness" or mixed
        assert sum(kind.startswith("vehicle.") for kind in kinds) == int(mixed)
        assert sum(kind.startswith("walker.pedestrian.") for kind in kinds) == int(pedestrian)
        assert len(kinds) == int(mixed) + int(pedestrian)

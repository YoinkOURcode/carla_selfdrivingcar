"""Consolidation, scenario transfer, and continued world-model fitting."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
import torch

from world_model_rl import runner
from world_model_rl.__main__ import main
from world_model_rl.config import Config, require_compatible
from world_model_rl.data import Dataset, merge_datasets, split_for
from world_model_rl.tests.test_pipeline import cfg, make_episode
from world_model_rl.tests.test_fixes import build_ready_model
from world_model_rl.training import evaluate_model, imagination_gate, load_checkpoint, validation_suite


def test_scenario_changes_require_opt_in_and_keep_physics_reward_ego_contract(cfg):
    saved = cfg.to_dict()
    cfg.environment.town = "Town02"
    cfg.environment.weather = ["WetNoon"]
    cfg.environment.vehicles = 30
    with pytest.raises(ValueError, match="environment differs"):
        require_compatible(saved, cfg)
    cfg.training.allow_scenario_variation = True
    require_compatible(saved, cfg)
    for section, name, value in (("environment", "ego_model", "vehicle.audi.tt"),
            ("environment", "fixed_delta_seconds", .1), ("environment", "action_repeat", 3),
            ("environment", "observation_provider", "different"),
            ("environment", "max_steps", 99), ("environment", "steering_rate", 1.0),
            ("reward", "red_light", 999), ("model", "hidden", 64)):
        changed = copy.deepcopy(cfg)
        setattr(getattr(changed, section), name, value)
        with pytest.raises(ValueError, match=f"{section} differs"):
            require_compatible(saved, changed)


def test_merge_is_idempotent_preserves_splits_and_does_not_modify_sources(cfg, tmp_path):
    first = Dataset(tmp_path / "first", cfg, create=True)
    second = Dataset(tmp_path / "second", cfg, create=True)
    for identity in range(10):
        first.save(make_episode(cfg, identity))
        ep = make_episode(cfg, identity)
        if identity % 2:
            ep.arrays["continuous"][:, 0] += 1  # Same scene seed, distinct rollout.
        second.save(ep)
    original = {path: path.read_bytes() for folder in (first.path, second.path) for path in folder.iterdir()}
    target = tmp_path / "merged"
    report = merge_datasets(cfg, target, [first.path, second.path])
    assert report["episodes"] == 15
    assert report["sources"][1]["duplicates"] == 5
    loaded = Dataset(target, cfg).load()
    for ep in loaded:
        provenance = ep.metadata["imported_from"]
        assert ep.metadata["split"] == split_for(provenance["episode_id"], cfg.training.seed)
        assert ep.metadata["split"] == split_for(ep.metadata["episode_id"], cfg.training.seed)
    assert merge_datasets(cfg, target, [first.path, second.path])["imported"] == 0
    assert all(path.read_bytes() == contents for path, contents in original.items())


def test_merge_preflights_incompatible_sources_before_writing(cfg, tmp_path):
    good = Dataset(tmp_path / "good", cfg, create=True)
    good.save(make_episode(cfg))
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg.reward.red_light += 1
    bad = Dataset(tmp_path / "bad", bad_cfg, create=True)
    bad.save(make_episode(bad_cfg))
    with pytest.raises(ValueError, match="reward differs"):
        merge_datasets(cfg, tmp_path / "destination", [good.path, bad.path])
    assert not (tmp_path / "destination").exists()


def test_collect_cli_appends_new_scenario_with_explicit_saved_config(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config().to_dict()))
    received = []
    monkeypatch.setattr(runner, "collect", lambda *args: received.append(args))
    main(["collect", "--config", str(config), "--data", str(tmp_path / "data"),
          "--allow-scenario-variation", "--town", "Town02", "--weather", "WetNoon", "ClearNight",
          "--vehicles", "30", "--pedestrians", "10", "--episodes", "20"])
    cfg = received[0][0]
    assert cfg.training.allow_scenario_variation
    assert cfg.environment.town == "Town02" and cfg.environment.weather == ["WetNoon", "ClearNight"]
    assert cfg.environment.vehicles == 30 and cfg.environment.pedestrians == 10
    assert received[0][2] == 20


def test_mixed_scenario_dataset_load_and_validation_include_new_domains(cfg, tmp_path):
    dataset = Dataset(tmp_path / "data", cfg, create=True)
    for identity in range(10):
        dataset.save(make_episode(cfg, identity))
    transferred = copy.deepcopy(cfg)
    transferred.training.allow_scenario_variation = True
    transferred.environment.town = "Town02"
    mixed = Dataset(dataset.path, transferred)
    for identity in range(10, 20):
        mixed.save(make_episode(transferred, identity))
    validation = mixed.load("validation")
    suite = validation_suite(validation, transferred, [validation[0].metadata["episode_id"]])
    assert len(suite) == 2
    assert {ep.metadata["config"]["environment"]["town"] for ep in suite} == {"Town01", "Town02"}
    # Old-town holdouts cannot certify a new-town training set.
    report = runner.fitting_coverage(transferred, mixed.load("train"), suite[:1])
    assert any("no validation episodes" in reason for reason in report["reasons"])


def test_warm_start_retains_normalizer_optimizer_and_counts_without_overwriting(cfg, tmp_path):
    cfg.training.gate_min_terminal_negative = 1
    dataset, source = build_ready_model(cfg, tmp_path, endings=True)
    original_bytes = source.read_bytes()
    before, _, normalizer, _ = load_checkpoint(source, cfg, "cpu")
    original_train_count = len(dataset.load("train"))
    cfg.training.allow_scenario_variation = True
    new_scene = copy.deepcopy(cfg)
    new_scene.environment.town = "Town02"
    mixed = Dataset(dataset.path, new_scene)
    for identity in range(10, 20):
        mixed.save(make_episode(new_scene, identity, terminal=identity % 2 == 0))
    output = tmp_path / "adapted.pt"
    runner.fit(cfg, dataset.path, output, updates=1, warm_start=source)
    after, _, new_normalizer, optimizer = load_checkpoint(output, cfg, "cpu")
    assert new_normalizer.to_dict() == normalizer.to_dict()
    assert after["model_version"] == before["model_version"] + 1
    assert after["model_updates"] == before["model_updates"] + 1
    assert after["normalizer_frozen"] and after["source_checkpoint"] == str(source)
    assert len(after["validation"]["by_scenario"]) == 2
    assert len(after["training_episode_ids"]) > original_train_count
    assert optimizer.state_dict()["state"]
    assert any(not torch.equal(before["model"][key], value) for key, value in after["model"].items())
    assert source.read_bytes() == original_bytes and "sac" not in after


def test_new_scenario_validation_is_reported_and_independently_gated(cfg, tmp_path):
    dataset, source = build_ready_model(cfg, tmp_path)
    cfg.training.allow_scenario_variation = True
    other = copy.deepcopy(cfg)
    other.environment.town = "Town02"
    episodes = [make_episode(cfg, 8), make_episode(other, 18)]
    _, model, norm, _ = load_checkpoint(source, cfg, "cpu")
    report = evaluate_model(model, episodes, norm, cfg, "cpu")
    assert len(report["by_scenario"]) == 2
    assert sum(item["episodes"] for item in report["by_scenario"].values()) == 2
    # The aggregate can pass while a new-domain metric fails.
    report["by_scenario"] = {"Town02": {"available": False, "reason": "missing held-out endings"}}
    passed, reason = imagination_gate(report, cfg)
    assert not passed and "scenario Town02: missing held-out endings" in reason


@pytest.mark.parametrize("code", [0, 2])
def test_collected_launcher_uses_integrated_replay_and_propagates_failure(tmp_path, code):
    project = tmp_path / "project with spaces"
    package = project / "world_model_rl"
    package.mkdir(parents=True)
    script = package / "train_collected.sh"
    script.write_text((Path(__file__).resolve().parents[1] / script.name).read_text())
    config = package / "configs" / "collected.json"
    config.parent.mkdir()
    config.write_text("{}")
    workspace = project / "runs" / "world_model_rl"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "dataset.json").write_text("{}")
    (workspace / "world_model.pt").write_bytes(b"model")
    interpreter = project / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@"\nexit {code}\n')
    interpreter.chmod(0o755)
    result = subprocess.run(["bash", str(script), "runs/new policy", "7"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == code
    log = (project / "runs" / "new policy.log").read_text()
    assert "train\n" in log and "--data\nruns/world_model_rl/data\n" in log
    assert "--model\nruns/world_model_rl/world_model.pt\n" in log
    again = subprocess.run(["bash", str(script), "runs/new policy", "7"],
                           cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 2
    assert (project / "runs" / "new policy.log").read_text() == log

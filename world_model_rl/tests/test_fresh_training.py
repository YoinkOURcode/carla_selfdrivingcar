"""Fresh experiment isolation, supported CLI, and launcher failure propagation."""

from pathlib import Path
import subprocess

import pytest

from world_model_rl import runner
from world_model_rl.__main__ import main
from world_model_rl.tests.test_pipeline import cfg
from world_model_rl.tests.test_fixes import build_ready_model
from world_model_rl.training import load_checkpoint, save_checkpoint


@pytest.mark.parametrize("artifact", ["data/dataset.json", "world_model.pt", "agent.pt", "config.json"])
def test_fresh_run_rejects_existing_artifacts_before_collection(cfg, tmp_path, monkeypatch, artifact):
    output = tmp_path / "experiment"
    path = output / artifact
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve this experiment")
    def forbidden(*args, **kwargs):
        raise AssertionError("An existing experiment must be rejected before collection or fitting")
    monkeypatch.setattr(runner, "collect", forbidden)
    monkeypatch.setattr(runner, "fit", forbidden)
    with pytest.raises(FileExistsError, match="new or empty"):
        runner.run(cfg, output, policy_updates=1)
    assert path.read_bytes() == b"preserve this experiment"


@pytest.mark.parametrize("arguments", [["migrate"], ["fit", "--checkpoint", "unused.pt", "--resume", "old.pt"]])
def test_removed_entry_points_are_rejected(arguments):
    with pytest.raises(SystemExit) as stopped:
        main(arguments)
    assert stopped.value.code == 2


def test_policy_cannot_use_an_agent_as_its_initial_world_model(cfg, tmp_path):
    dataset, source = build_ready_model(cfg, tmp_path)
    _, model, norm, optimizer = load_checkpoint(source, cfg, "cpu")
    agent = tmp_path / "old_agent.pt"
    save_checkpoint(agent, cfg, model, norm, optimizer, 2, 100, kind="agent")
    with pytest.raises(ValueError, match="--model requires a world-model checkpoint"):
        runner.train(cfg, dataset.path, tmp_path / "new_policy", model_path=agent, policy_updates=1)
    assert not (tmp_path / "new_policy" / "agent.pt").exists()


def test_fresh_fit_initializes_model_normalizer_and_counters(cfg, tmp_path, monkeypatch):
    cfg.training.gate_min_terminal_negative = 1  # Six-step numeric episodes.
    dataset, source = build_ready_model(cfg, tmp_path, endings=True)
    source_bytes = source.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("A fresh world-model fit must not load a checkpoint or connect to CARLA")
    monkeypatch.setattr(runner, "load_checkpoint", forbidden)
    monkeypatch.setattr(runner, "CarlaDrivingEnv", forbidden)
    target = tmp_path / "new_model.pt"
    runner.fit(cfg, dataset.path, target, updates=1)
    payload, model, norm, _ = load_checkpoint(target, cfg, "cpu")
    assert payload["kind"] == "world_model"
    assert payload["model_version"] == 1 and payload["model_updates"] == 1
    assert payload["termination_updates"] == cfg.training.termination_fit_updates
    assert "sac" not in payload
    assert model.encoder.ego.weight.count_nonzero() > 0
    assert source.read_bytes() == source_bytes
    assert norm.to_dict() == runner.Normalizer.fit(dataset.load("train")).to_dict()


@pytest.mark.parametrize("code", [0, 2, 134])
def test_launcher_streams_logs_and_preserves_training_exit_status(tmp_path, code):
    # A fake interpreter verifies shell behavior without simulator or neural training.
    project = tmp_path / "project with spaces"
    package = project / "world_model_rl"
    package.mkdir(parents=True)
    script = package / "train_from_scratch.sh"
    script.write_text((Path(__file__).resolve().parents[1] / script.name).read_text())
    (package / "config.json").write_text("{}")
    interpreter = project / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@"\nexit {code}\n')
    interpreter.chmod(0o755)
    result = subprocess.run(["bash", str(script), "runs/new experiment", "7"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == code
    log = (project / "runs" / "new experiment.log").read_text()
    assert "--policy-updates\n7\n" in log and "--output\nruns/new experiment\n" in log
    assert "--resume" not in log
    # A second invocation refuses the existing log rather than overwriting evidence.
    again = subprocess.run(["bash", str(script), "runs/new experiment", "7"],
                           cwd=tmp_path, capture_output=True, text=True)
    assert again.returncode == 2
    assert (project / "runs" / "new experiment.log").read_text() == log

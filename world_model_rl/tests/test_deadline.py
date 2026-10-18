"""Optimizer-boundary deadlines and immutable candidates; no live driving."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from world_model_rl import deadline, runner
from world_model_rl.deadline import TrainingClock
from world_model_rl.tests.test_fixes import build_ready_model
from world_model_rl.tests.test_pipeline import cfg
from world_model_rl.training import load_checkpoint


def controlled_clock(tmp_path, monkeypatch, seconds=10, checkpoint_seconds=2):
    now = [1000.0]
    monkeypatch.setattr(deadline.time, "time", lambda: now[0])
    return TrainingClock(tmp_path / "clock.json", seconds, checkpoint_seconds), now


def test_shared_clock_keeps_deadline_archives_and_excludes_concurrent_training(tmp_path, monkeypatch):
    clock, now = controlled_clock(tmp_path, monkeypatch)
    assert not clock.path.exists() and not clock.expired
    clock.start()
    started = clock.summary()["deadline_unix"]
    source = tmp_path / "agent.pt"
    source.write_bytes(b"first candidate")
    clock.record_checkpoint(source, "warmup", {"warmup_updates": 4}, force=True)
    first = clock.value["checkpoints"][0]
    source.write_bytes(b"later candidate")
    assert Path(first["path"]).read_bytes() == b"first candidate"
    assert first["sha256"] == hashlib.sha256(b"first candidate").hexdigest()
    contender = TrainingClock(clock.path, 10, 2)
    with pytest.raises(RuntimeError, match="Another training command"):
        contender.start()
    clock.close()
    now[0] += 11
    next_stage = TrainingClock(clock.path, 10, 2)
    next_stage.start()
    assert next_stage.expired and next_stage.summary()["deadline_unix"] == started
    next_stage.record_checkpoint(source, "deadline", {"policy_updates": 2})
    assert len(next_stage.value["checkpoints"]) == 2
    next_stage.close()
    with pytest.raises(ValueError, match="cannot be reset"):
        TrainingClock(clock.path, 20, 2)


def test_fresh_model_saves_when_deadline_expires_mid_fitting(cfg, tmp_path, monkeypatch):
    cfg.training.gate_min_terminal_negative = 1
    dataset, original = build_ready_model(cfg, tmp_path, endings=True)
    original_bytes = original.read_bytes()
    clock, now = controlled_clock(tmp_path, monkeypatch)
    step = torch.optim.Adam.step
    def advance_after_complete_update(opt, *args, **kwargs):
        result = step(opt, *args, **kwargs)
        now[0] += 6
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", advance_after_complete_update)
    monkeypatch.setattr(runner, "CarlaDrivingEnv", lambda *a, **k: pytest.fail("Offline fitting connected to CARLA"))
    target = tmp_path / "fresh_model.pt"
    runner.fit(cfg, dataset.path, target, updates=50, clock=clock)
    payload, _, norm, _ = load_checkpoint(target, cfg, "cpu")
    assert payload["model_updates"] == 2 and payload["termination_updates"] == 0
    assert payload["fit_stage"] == "deadline" and payload["training_clock"]["deadline_reached"]
    assert payload["training_episode_ids"] == [ep.metadata["episode_id"] for ep in dataset.load("train")]
    assert norm.to_dict() == runner.Normalizer.fit(dataset.load("train")).to_dict()
    assert original.read_bytes() == original_bytes
    assert any(r["deadline_reached"] for r in clock.value["checkpoints"])
    clock.close()


def test_warmup_deadline_preserves_partial_actor_and_resumes_remaining_updates(cfg, tmp_path, monkeypatch):
    cfg.training.collection_episodes = cfg.training.refresh_updates = 0
    cfg.training.behavior_cloning_updates = 8
    cfg.training.behavior_cloning_teacher = "geometry_v3"
    dataset, model_path = build_ready_model(cfg, tmp_path)
    clock, now = controlled_clock(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "imagination_gate", lambda *a: (True, "numeric fixture only"))
    monkeypatch.setattr(runner, "CarlaDrivingEnv", lambda *a, **k: pytest.fail("Offline warm-up connected to CARLA"))
    clone = runner.SAC.clone_expert
    calls = []
    def advance_after_warmup(sac, batch):
        result = clone(sac, batch)
        calls.append(sac.updates)
        now[0] += 6
        return result
    monkeypatch.setattr(runner.SAC, "clone_expert", advance_after_warmup)
    output = tmp_path / "policy"
    result = runner.train(cfg, dataset.path, output, model_path=model_path, policy_updates=2, clock=clock)
    payload = torch.load(output / "agent.pt", map_location="cpu", weights_only=False)
    assert result["deadline_reached"] and result["policy_updates"] == 0
    assert payload["policy_initialization"]["updates"] == 2 and calls == [0, 0]
    clock.close()
    # An expired clock is never extended by another invocation.
    result = runner.train(cfg, dataset.path, output, resume=output / "agent.pt", policy_updates=2, clock=clock)
    assert result["deadline_reached"] and calls == [0, 0]
    clock.close()
    # Explicit untimed recovery exercises partial-warmup counters, not a fresh actor.
    result = runner.train(cfg, dataset.path, output, resume=output / "agent.pt", policy_updates=2)
    assert calls == [0] * 8 and result["policy_updates"] == 2
    assert torch.load(output / "agent.pt", weights_only=False)["policy_initialization"]["updates"] == 8
    assert set(payload["policy_initialization"]["episode_ids"]) == {
        ep.metadata["episode_id"] for ep in dataset.load("train")}


def test_deadline_during_readout_fit_saves_its_optimizer_and_actual_counts(cfg, tmp_path, monkeypatch):
    cfg.training.gate_min_terminal_negative = 1
    cfg.training.termination_fit_updates = 50
    dataset, _ = build_ready_model(cfg, tmp_path, endings=True)
    clock, now = controlled_clock(tmp_path, monkeypatch)
    step = torch.optim.Adam.step
    def advance_after_step(opt, *args, **kwargs):
        result = step(opt, *args, **kwargs)
        now[0] += 6
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", advance_after_step)
    target = tmp_path / "fresh_model.pt"
    runner.fit(cfg, dataset.path, target, updates=1, clock=clock)
    payload, _, _, _ = load_checkpoint(target, cfg, "cpu")
    assert payload["model_updates"] == 1 and payload["termination_updates"] == 1
    assert payload["termination_optimizer"]["state"]
    assert payload["termination_fit"]["updates"] == 1
    assert payload["training_clock"]["deadline_reached"]
    clock.close()


def test_sac_deadline_saves_completed_updates_and_separate_stop_log(cfg, tmp_path, monkeypatch):
    cfg.training.collection_episodes = cfg.training.refresh_updates = 0
    dataset, model_path = build_ready_model(cfg, tmp_path)
    clock, now = controlled_clock(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "imagination_gate", lambda *a: (True, "numeric fixture only"))
    update = runner.SAC.update
    calls = []
    def advance_after_update(sac, *args, **kwargs):
        result = update(sac, *args, **kwargs)
        calls.append(sac.updates)
        now[0] += 6
        return result
    monkeypatch.setattr(runner.SAC, "update", advance_after_update)
    output = tmp_path / "policy"
    result = runner.train(cfg, dataset.path, output, model_path=model_path, policy_updates=50, clock=clock)
    payload = torch.load(output / "agent.pt", weights_only=False)
    assert calls == [1, 2] and result["policy_updates"] == 2 and result["deadline_reached"]
    assert payload["sac"]["updates"] == 2
    assert payload["sac"]["q_optimizer"]["state"] and payload["rng"]
    stop = json.loads((output / "training.jsonl").read_text().splitlines()[-1])
    assert stop["stage"] == "deadline" and stop["policy_updates"] == 2 and stop["warmup_updates"] == 0
    clock.close()


def test_deadline_during_imagination_does_not_allow_a_sac_update(cfg, tmp_path, monkeypatch):
    cfg.training.collection_episodes = cfg.training.refresh_updates = 0
    dataset, model_path = build_ready_model(cfg, tmp_path)
    clock, now = controlled_clock(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "imagination_gate", lambda *a: (True, "numeric fixture only"))
    imagine = runner.imagine
    def late_imagination(*args, **kwargs):
        result = imagine(*args, **kwargs)
        now[0] += 11
        return result
    monkeypatch.setattr(runner, "imagine", late_imagination)
    monkeypatch.setattr(runner.SAC, "update", lambda *a: pytest.fail("An update ran after the deadline"))
    result = runner.train(cfg, dataset.path, tmp_path / "policy", model_path=model_path, policy_updates=8, clock=clock)
    assert result["deadline_reached"] and result["policy_updates"] == 0
    assert clock.value["checkpoints"][-1]["stage"] == "deadline"
    clock.close()


def test_timed_training_rejects_live_collection_before_starting_clock(cfg, tmp_path):
    clock = TrainingClock(tmp_path / "clock.json")
    with pytest.raises(ValueError, match="requires offline replay"):
        runner.train(cfg, "unused", tmp_path / "policy", model_path="unused", clock=clock)
    assert not clock.path.exists() and not (tmp_path / "policy").exists()

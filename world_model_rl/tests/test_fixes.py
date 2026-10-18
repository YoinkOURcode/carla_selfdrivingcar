"""Regression tests for comfort rewards and offline policy blocks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from world_model_rl.data import Dataset, sample_sequences
from world_model_rl.environment import CarlaDrivingEnv
from world_model_rl.model import RSSM, tensor_observation
from world_model_rl.observations import legitimate_wait
from world_model_rl.rewards import ComfortCost
from world_model_rl.sac import HistoryCache
from world_model_rl.schema import Normalizer
from world_model_rl.training import save_checkpoint, validation_suite
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_comfort_is_filtered_bounded_and_does_not_charge_collision_impulses(cfg):
    calc = ComfortCost(cfg)
    before, applied = np.zeros(3), np.array([1.0, 0.5, 0])
    dt = cfg.environment.decision_dt
    costs = calc.step(0, 2, applied, before, dt)
    expected = 2 * (1 - np.exp(-dt / cfg.reward.comfort_filter_seconds))
    assert calc.acceleration == pytest.approx(expected)
    assert calc.jerk == pytest.approx(expected / dt)
    for acceleration in [1000, -1000] * 100:
        costs = calc.step(0, acceleration, applied, before, dt)
        assert -0.25 * dt - 1e-10 <= sum(costs.values()) <= 0
    assert sum(calc.step(0, 1e6, applied, before, dt, collision=True).values()) == 0
    costs = calc.step(0, 0, before, before, dt)
    assert sum(costs.values()) == 0  # Impact does not contaminate subsequent filtering.


def test_waiting_exemption_requires_a_relevant_control_ahead():
    assert legitimate_wait(True, "red", 5, False, 0, False)
    assert legitimate_wait(True, "yellow", 0, False, 0, False)
    assert not legitimate_wait(True, "red", -1, False, 0, False)
    assert not legitimate_wait(True, "red", 26, False, 0, False)
    assert not legitimate_wait(True, "green", 5, False, 0, False)
    assert legitimate_wait(False, "none", 0, True, 2, False)
    assert not legitimate_wait(False, "none", 0, True, 2, True)


def test_traffic_crossings_and_stop_dwell_keep_physics_tick_resolution(cfg):
    env = CarlaDrivingEnv.__new__(CarlaDrivingEnv)
    env.cfg = cfg
    env.carla = SimpleNamespace(TrafficLightState=SimpleNamespace(Red="red"))
    light = {"kind": "light", "centre": np.zeros(2), "forward": np.array([1, 0]),
             "half_width": 2, "actor": SimpleNamespace(get_state=lambda: "red")}
    stop = {"kind": "stop", "centre": np.zeros(2), "forward": np.array([1, 0]),
            "half_width": 2, "dwell": 0.0, "satisfied": False}
    env.controls = [light, stop]
    events = env._traffic_events(np.array([-1, 0]), np.array([1, 0]), 2)
    assert events == {"red_light": 1, "stop_sign": 1}
    for _ in range(20):
        env._traffic_events(np.array([-1, 0]), np.array([-1, 0]), 0)
    assert stop["satisfied"] and stop["dwell"] == pytest.approx(1)
    assert env._traffic_events(np.array([-1, 0]), np.array([1, 0]), 2)["stop_sign"] == 0


def test_failure_tail_sampling_reaches_endings_without_changing_validation_distribution(cfg):
    normal, failure = make_episode(cfg), make_episode(cfg, 1, terminal=True)
    batch = sample_sequences([normal, failure], 32, 3, np.random.default_rng(4), failure_fraction=1)
    assert batch["terminated"][:, -1].all()
    episodes = [make_episode(cfg, identity) for identity in range(25)]
    cfg.training.validation_episodes = 3
    suite = validation_suite(episodes, cfg)
    assert [ep.metadata["episode_id"] for ep in suite] == [0, 1, 2]
    assert [ep.metadata["episode_id"] for ep in validation_suite(episodes + [failure], cfg, [0, 2])] == [0, 2]


def test_batched_encoding_matches_online_history_and_cache_invalidates_by_version(cfg, monkeypatch):
    episodes = [make_episode(cfg), make_episode(cfg, 1, steps=2)]
    norm, model = Normalizer.fit(episodes), RSSM(cfg.model)
    results = model.encode_episodes(episodes, norm, "cpu")
    for episode, result in zip(episodes, results):
        obs = tensor_observation(episode.arrays, norm, "cpu")
        at = lambda t: {key: value[t:t + 1] for key, value in obs.items()}
        state = model.observe_initial(at(0))
        weather = obs["continuous"][0:1, 16:20]
        expected = [model.features(state, torch.tensor(episode.arrays["continuous"][0:1, 5:8]), weather)]
        for t, action in enumerate(episode.arrays["applied_actions"]):
            action = torch.tensor(action[None])
            state, _, _ = model.transition(state, action, at(t + 1), sample=False, weather=weather)
            expected.append(model.features(state, action, weather))
        np.testing.assert_allclose(result, torch.cat(expected).detach().numpy(), atol=1e-6)
    cache = HistoryCache()
    calls = []
    original = model.encode_episodes
    def recorded(items, *args):
        calls.append(len(items))
        return original(items, *args)
    monkeypatch.setattr(model, "encode_episodes", recorded)
    cache.encode(episodes, model, norm, "cpu", 1)
    cache.encode(episodes, model, norm, "cpu", 1)
    cache.encode(episodes, model, norm, "cpu", 2)
    assert calls == [2, 0, 2]
    batched = tensor_observation({key: np.stack([episode.arrays[key][:3] for episode in episodes])
                                 for key in ("continuous", "present", "classes", "controls", "light")}, norm, "cpu")
    encoded = model.encoder(batched)
    torch.testing.assert_close(encoded[:, 1], model.encoder({key: value[:, 1] for key, value in batched.items()}))


def build_ready_model(cfg, tmp_path, endings=False):
    dataset = Dataset(tmp_path / "data", cfg, create=True)
    for identity in range(10):
        dataset.save(make_episode(cfg, identity, terminal=endings and identity % 2 == 0))
    model = RSSM(cfg.model)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    model.termination_head[-1].bias.data.fill_(-100)
    norm = Normalizer.fit(dataset.load("train"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    path = tmp_path / "world_model.pt"
    save_checkpoint(path, cfg, model, norm, optimizer, 1, 1, kind="world_model")
    return dataset, path


def test_failed_gate_never_constructs_carla_or_saves_a_sac_checkpoint(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    dataset, path = build_ready_model(cfg, tmp_path)
    def forbidden(*args):
        raise AssertionError("Failed model gate must not connect to CARLA")
    monkeypatch.setattr(runner, "CarlaDrivingEnv", forbidden)
    with pytest.raises(RuntimeError, match="gate blocked SAC"):
        runner.train(cfg, dataset.path, tmp_path / "policy", model_path=path, policy_updates=2)
    assert not (tmp_path / "policy" / "agent.pt").exists()
    assert (tmp_path / "policy" / "validation.json").exists()


def test_offline_policy_budget_and_resume_do_not_construct_carla(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    cfg.training.policy_block_updates = 3
    cfg.training.collection_episodes = 0
    dataset, path = build_ready_model(cfg, tmp_path)
    def forbidden(*args):
        raise AssertionError("Replay-only optimizer blocks must not connect to CARLA")
    monkeypatch.setattr(runner, "CarlaDrivingEnv", forbidden)
    monkeypatch.setattr(runner, "imagination_gate", lambda *args: (True, "explicit fixture pass"))
    output = tmp_path / "policy"
    result = runner.train(cfg, dataset.path, output, model_path=path, policy_updates=5)
    assert result["additional_policy_updates"] == 5 and result["training_episodes"] == 0
    result = runner.train(cfg, dataset.path, output, resume=output / "agent.pt", policy_updates=2)
    assert result["policy_updates"] == 7 and result["additional_collection_episodes"] == 0
    logs = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert all(item["synthetic_fraction"] == 0.25 for item in logs)
    assert sum(item["imagination"]["accepted"] for item in logs) > 0

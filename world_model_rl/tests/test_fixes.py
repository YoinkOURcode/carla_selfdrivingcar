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
from world_model_rl.rewards import ComfortCost, RewardCalculator
from world_model_rl.sac import HistoryCache
from world_model_rl.schema import EVENTS, Normalizer, Observation
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
    assert legitimate_wait(True, "red", 2.5, 0)
    assert legitimate_wait(True, "yellow", 0, 0)
    assert legitimate_wait(True, "red", 5, 0.1)
    assert not legitimate_wait(True, "red", -1, 0)
    assert not legitimate_wait(True, "red", 5.1, 0)
    assert not legitimate_wait(True, "red", 25, 0)
    assert not legitimate_wait(True, "green", 2.5, 0)
    assert not legitimate_wait(True, "unknown", 2.5, 0)
    assert not legitimate_wait(False, "red", 2.5, 0)
    assert not legitimate_wait(True, "red", 2.5, 2)


def test_red_light_crossings_keep_physics_tick_resolution(cfg):
    env = CarlaDrivingEnv.__new__(CarlaDrivingEnv)
    env.cfg = cfg
    env.carla = SimpleNamespace(TrafficLightState=SimpleNamespace(Red="red"))
    light = {"kind": "light", "centre": np.zeros(2), "forward": np.array([1, 0]),
             "half_width": 2, "actor": SimpleNamespace(get_state=lambda: "red")}
    env.controls = [light]
    events = env._traffic_events(np.array([-1, 0]), np.array([1, 0]))
    assert events == {"red_light": 1}
    assert env._traffic_events(np.array([-1, 0]), np.array([-1, 0])) == {"red_light": 0}
    light["actor"].get_state = lambda: "green"
    assert env._traffic_events(np.array([-1, 0]), np.array([1, 0])) == {"red_light": 0}


def deadline_environment(cfg):
    """Exercise the actual environment step with deterministic, sensor-free measurements."""
    env = CarlaDrivingEnv.__new__(CarlaDrivingEnv)
    env.cfg = cfg
    env.carla = SimpleNamespace(VehicleControl=lambda **kwargs: kwargs)
    env.ego = SimpleNamespace(apply_control=lambda control: None)
    env.previous_action = np.zeros(3, np.float32)
    env.steps = env.active_steps = env.frame = env.committed_frame = 0
    env.sim_time = env.stuck_time = env.offroad_time = 0.0
    env.ended, env.controls = False, []
    env.route = SimpleNamespace(high_water=1, length=10)
    env.measurement = {"speed": 2.0, "longitudinal_acceleration": 0.0,
        "position": np.zeros(2), "front": np.zeros(2), "speed_limit": 15,
        "weather_limit": 15, "child_near": False, "offroad": False, "wrong_way": False,
        "progress_distance": 0.1, "success": False, "blocked": False,
        "lane_offset": 0.0, "heading_error": 0.0}
    env.incoming = {"collision": 0, "illegal_lane": 0}
    def tick(settle=False):
        env.frame += 1
        env.sim_time += cfg.environment.fixed_delta_seconds
    def measure():
        obs = Observation.empty()
        obs.continuous[5:8] = env.previous_action
        return obs, dict(env.measurement)
    env._tick = tick
    env._events_at_boundary = lambda frame: dict(env.incoming)
    env.extractor = SimpleNamespace(measure=measure)
    env.provider = SimpleNamespace(observe=lambda current_env, oracle: oracle)
    env.reward_calculator = RewardCalculator(cfg)
    return env


def test_light_waits_can_exceed_the_budget_without_spending_it_and_green_resumes(cfg):
    cfg.environment.max_steps = 3
    env = deadline_environment(cfg)
    env.measurement.update(speed=0.0, progress_distance=0.0, blocked=True)
    for _ in range(8):
        obs, _, terminated, truncated, info = env.step([0, 0, 1])
        assert not terminated and not truncated
        assert obs.continuous[15] == pytest.approx(0.3)
        assert env.active_steps == 0 and env.stuck_time == 0
        assert info["events"]["time_limit"] == info["components"]["time_limit"] == 0
    env.measurement["blocked"] = False  # Green releases the waiting exemption.
    for index in range(3):
        obs, reward, terminated, truncated, info = env.step([0, 0.5, 0])
        assert obs.continuous[15] == pytest.approx((2 - index) * 0.1)
        assert terminated == (index == 2) and not truncated
    assert env.steps == 11 and env.active_steps == 3
    assert info["terminal_reason"] == "time_limit"
    assert info["events"]["time_limit"] == 1
    assert info["components"]["time_limit"] == -cfg.reward.time_limit
    assert reward == pytest.approx(sum(info["components"].values()))
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step([0, 0, 1])


@pytest.mark.parametrize("reason", ["success", "collision", "offroad"])
def test_other_task_endings_take_priority_over_timeout_cost(cfg, reason):
    cfg.environment.max_steps = 1
    env = deadline_environment(cfg)
    if reason == "collision":
        env.incoming["collision"] = 1
        env.measurement["success"] = True
    else:
        env.measurement[reason] = True
    if reason == "offroad":
        env.offroad_time = cfg.environment.offroad_seconds
    _, _, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated and not truncated and info["terminal_reason"] == reason
    assert info["events"]["time_limit"] == info["components"]["time_limit"] == 0
    if reason == "collision":
        assert info["components"]["success"] == 0


def test_stuck_remains_a_separate_truncation_without_timeout_cost(cfg):
    cfg.environment.stuck_seconds = 0.2
    env = deadline_environment(cfg)
    env.measurement.update(speed=0.0, progress_distance=0.0)
    env.step([0, 0, 1])
    _, _, terminated, truncated, info = env.step([0, 0, 1])
    assert not terminated and truncated and info["terminal_reason"] == "stuck"
    assert info["components"]["time_limit"] == 0


def test_timeout_transition_reaches_recordings_model_targets_and_sac_replay(cfg):
    from world_model_rl.data import EpisodeBuilder
    from world_model_rl.sac import real_replay
    from world_model_rl.schema import SCHEMA_VERSION

    cfg.environment.max_steps = 1
    env = deadline_environment(cfg)
    metadata = {"schema": SCHEMA_VERSION, "config": cfg.to_dict(), "episode_id": 0}
    initial = Observation.empty()
    initial.continuous[15] = cfg.environment.decision_dt
    builder = EpisodeBuilder(initial, {"frame": 0, "sim_time": 0.0}, metadata)
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    builder.append([0, 0, 0], obs, reward, terminated, truncated, info)
    episode = builder.finish()
    assert episode.arrays["terminated"].tolist() == [True]
    assert episode.arrays["truncated"].tolist() == [False]
    assert episode.arrays["events"][0, EVENTS.index("time_limit")] == 1
    batch = sample_sequences([episode], 1, 1, np.random.default_rng(0), reward_cfg=cfg)
    assert batch["terminated"][0, 0] and batch["rewards"][0, 0] == pytest.approx(reward)
    norm, model = Normalizer.fit([episode]), RSSM(cfg.model)
    replay = real_replay([episode], model, norm, "cpu", 1, reward_cfg=cfg)
    assert replay.arrays["terminated"][0] and replay.arrays["rewards"][0] == pytest.approx(reward)


def test_clock_is_never_masked_when_no_light_is_present(cfg):
    from world_model_rl.model import tensor_observation
    from world_model_rl.policy import physical_observation, predicted_observation

    episode = make_episode(cfg)
    episode.arrays["continuous"][:, 15] = np.linspace(6, 0, len(episode) + 1)
    norm = Normalizer.fit([episode])
    obs = tensor_observation(episode.arrays, norm, "cpu")
    restored = physical_observation(obs, norm)
    np.testing.assert_allclose(restored["continuous"][:, 15], episode.arrays["continuous"][:, 15])
    prediction = {"continuous": obs["continuous"], "present": obs["present"].float() * 0 - 100,
        "controls": obs["controls"].float() * 0 - 100,
        "classes": torch.zeros(len(episode) + 1, 12, 3), "light": torch.zeros(len(episode) + 1, 5)}
    imagined = predicted_observation(prediction, torch.zeros(len(episode) + 1, 3),
                                    obs["continuous"][:, 16:20], norm)
    torch.testing.assert_close(imagined["continuous"][:, 15], obs["continuous"][:, 15])


def test_custom_perception_cannot_replace_the_runtime_clock(cfg):
    cfg.environment.observation_provider = "example:camera_provider"
    env = deadline_environment(cfg)
    env.active_steps = 2
    camera = Observation.empty()
    camera.continuous[11] = 12
    camera.continuous[15] = 999
    env.oracle = Observation.empty()
    env.provider.observe = lambda current_env, oracle: camera
    observation = env._policy_observation()
    assert observation.continuous[15] == pytest.approx((cfg.environment.max_steps - 2) * 0.1)


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


@pytest.mark.parametrize('legacy_refresh_updates', [0, 7])
def test_collection_and_resume_never_fit_world_model(cfg, tmp_path, monkeypatch, legacy_refresh_updates):
    """New replay, old refresh requests, and SAC updates preserve all model tensors."""
    from world_model_rl import runner
    cfg.training.refresh_updates = legacy_refresh_updates
    dataset, source = build_ready_model(cfg, tmp_path)
    payload, model, normalizer, optimizer = runner.load_checkpoint(source, cfg, 'cpu')
    # Populate optimizer moments so equality checks also cover saved fitting state.
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.termination_logit_offset.fill_(-2.5)
    terminal_fit = {'available': True, 'updates': 8}
    save_checkpoint(source, cfg, model, normalizer, optimizer, 3, 17, kind='world_model',
                    termination_updates=8, termination_fit=terminal_fit)
    before = torch.load(source, weights_only=False)
    source_bytes = source.read_bytes()
    loaded = []
    original_load = runner.load_checkpoint
    def load(*args, **kwargs):
        """Retain loaded models to check that SAC explicitly freezes their parameters."""
        result = original_load(*args, **kwargs)
        loaded.append(result[1])
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('SAC must not fit the world model, its ending head, or normalization')
    class Env:
        """Simulate collection boundaries without using a CARLA server."""
        def __init__(self, cfg):
            self.cfg = cfg
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def drive(env, identity, **kwargs):
        """Supply only nonterminal/truncated episodes, regardless of model state."""
        episode = make_episode(cfg, identity)
        return episode, {'episode_id': identity, 'steps': len(episode)}
    policy_updates = []
    original_update = runner.SAC.update
    def update(sac, *args, **kwargs):
        """Check actual parameter changes during fresh and resumed mixed-replay SAC."""
        assert torch.is_grad_enabled()
        networks = {name: getattr(sac, name) for name in ('actor', 'q1', 'q2', 'target1', 'target2')}
        before = {name: [p.detach().clone() for p in net.parameters()] for name, net in networks.items()}
        alpha = sac.log_alpha.detach().clone()
        result = original_update(sac, *args, **kwargs)
        for name, net in networks.items():
            assert any(not torch.equal(old, new) for old, new in zip(before[name], net.parameters())), name
            assert all(p.requires_grad == (not name.startswith('target')) for p in net.parameters())
        assert sac.log_alpha.requires_grad and not torch.equal(alpha, sac.log_alpha)
        policy_updates.append(sac.updates)
        return result
    monkeypatch.setattr(runner, 'load_checkpoint', load)
    monkeypatch.setattr(runner, 'fit_updates', forbidden)
    monkeypatch.setattr(runner, 'fit_termination_updates', forbidden)
    monkeypatch.setattr(runner.Normalizer, 'fit', forbidden)
    monkeypatch.setattr(runner, 'CarlaDrivingEnv', Env)
    monkeypatch.setattr(runner, 'drive_episode', drive)
    monkeypatch.setattr(runner.SAC, 'update', update)
    monkeypatch.setattr(runner, 'imagination_gate', lambda *a: (True, 'numeric fixture only'))
    output = tmp_path / 'policy'
    result = runner.train(cfg, dataset.path, output, model_path=source, episodes=2)
    assert result['additional_collection_episodes'] == 2
    agent_path = output / 'agent.pt'
    agent = torch.load(agent_path, weights_only=False)
    # Emulate a checkpoint from the old trainer interrupted after collection.
    agent.pop('replay_pending')
    agent['refresh_pending'] = True
    torch.save(agent, agent_path)
    cfg.training.refresh_updates = 0  # Deprecated refresh budgets need not match on resume.
    result = runner.train(cfg, dataset.path, output, resume=agent_path, policy_updates=2)
    after = torch.load(agent_path, weights_only=False)
    assert result['additional_policy_updates'] == 2
    assert len(policy_updates) == after['sac']['updates']
    assert result['training_episodes'] == 2
    assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in loaded)
    assert all(torch.equal(value, after['model'][key]) for key, value in before['model'].items())
    assert after['normalizer'] == before['normalizer']
    assert after['model_version'] == 3 and after['model_updates'] == 17
    assert after['termination_updates'] == 8 and after['termination_fit'] == terminal_fit
    assert after['model_optimizer']['param_groups'] == before['model_optimizer']['param_groups']
    for identity, moments in before['model_optimizer']['state'].items():
        for name, value in moments.items():
            assert torch.equal(value, after['model_optimizer']['state'][identity][name])
    assert after['world_model_frozen'] and not after['refresh_pending'] and not after['replay_pending']
    assert after['sac']['actor_optimizer']['state'] and after['sac']['q_optimizer']['state']
    assert not (output / 'world_model_training.jsonl').exists()
    assert not (output / 'termination_training.jsonl').exists()
    logs = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    assert any(item['stage'] == 'replay_refresh' for item in logs)
    assert not any(item['stage'] == 'model_refresh' for item in logs)
    assert source.read_bytes() == source_bytes

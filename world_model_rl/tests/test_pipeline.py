"""Offline numerical/integration checks. No test connects to CARLA or drives a car."""

import copy
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from world_model_rl.config import Config
from world_model_rl.control import apply_action, apply_action_tensor, visibility_range, weather_limits
from world_model_rl.data import Dataset, Episode, split_for
from world_model_rl.environment import CarlaDrivingEnv
from world_model_rl.geometry import Route, TrackedSlots, crosses_line, ego_relative
from world_model_rl.model import RSSM, tensor_observation
from world_model_rl.rewards import RewardCalculator
from world_model_rl.rewards import REWARD_VERSION
from world_model_rl.sac import Replay, SAC, mixed_batch, real_replay
from world_model_rl.schema import (CONTINUOUS, EVENTS, OBJECT_OFFSET, OBS_KEYS, SCHEMA_VERSION,
                                   SLOTS, Normalizer, Observation)
from world_model_rl.training import (choose_device, evaluate_model, imagination_gate, imagine,
                                     load_checkpoint, restore_rng, rng_state, save_checkpoint)


@pytest.fixture
def cfg():
    result = Config.load()
    result.model.embedding = 8
    result.model.attention_heads = 2
    result.model.deterministic = 8
    result.model.stochastic = 4
    result.model.hidden = 16
    result.model.batch_size = 2
    result.model.burn_in = 1
    result.model.sequence_length = 3
    result.environment.max_steps = 6
    result.training.batch_size = 4
    result.training.imagination_starts = 8
    result.training.validation_branches = 8
    result.training.minimum_real_steps = 1
    result.training.policy_block_updates = 3
    result.training.collection_episodes = 1
    result.training.refresh_updates = 1
    result.training.termination_fit_updates = 2
    choose_device("cpu")
    torch.manual_seed(4)
    return result


def make_episode(cfg, identity=0, steps=6, terminal=False):
    observations = []
    for index in range(steps + 1):
        obs = Observation.empty()
        obs.continuous[0] = 2 + 0.1 * index
        obs.continuous[10:14] = [1, 12, 8, 80 - index]
        obs.continuous[19] = 60
        obs.continuous[OBJECT_OFFSET:OBJECT_OFFSET + 7] = [10 - index, 1, -1, 0, 2, 1, 1]
        obs.present[0], obs.ids[0] = True, 37
        observations.append(obs.arrays())
    arrays = {key: np.stack([obs[key] for obs in observations]) for key in OBS_KEYS}
    arrays.update(requested_actions=np.zeros((steps, 3), np.float32),
                  applied_actions=np.zeros((steps, 3), np.float32), rewards=np.ones(steps, np.float32),
                  terminated=np.zeros(steps, bool), truncated=np.zeros(steps, bool),
                  events=np.zeros((steps, len(EVENTS)), np.float32), components=np.ones((steps, 1), np.float32),
                  frames=np.arange(steps + 1) * cfg.environment.action_repeat,
                  times=np.arange(steps + 1) * cfg.environment.decision_dt)
    arrays["terminated" if terminal else "truncated"][-1] = True
    metadata = {"episode_id": identity, "split": split_for(identity, cfg.training.seed),
                "schema": SCHEMA_VERSION, "reward_version": REWARD_VERSION, "config": cfg.to_dict(), "component_names": ["test"],
                "seed": cfg.training.seed + identity, "weather": "ClearNoon", "collector": "test"}
    return Episode(arrays, metadata).validate()


def test_action_adapter_has_identical_real_and_imagined_semantics(cfg):
    rng = np.random.default_rng(3)
    requests = rng.uniform([-2, -1, -1], [2, 2, 2], (100, 3)).astype(np.float32)
    previous = apply_action(rng.random((100, 3)), np.zeros((100, 3)), cfg.environment)
    real = apply_action(requests, previous, cfg.environment)
    imagined = apply_action_tensor(torch.tensor(requests), torch.tensor(previous), cfg.environment).numpy()
    np.testing.assert_allclose(real, imagined, atol=1e-7)
    assert np.all(real[:, 1] * real[:, 2] == 0)
    assert np.max(np.abs(real[:, 0] - previous[:, 0])) <= cfg.environment.steering_rate * cfg.environment.decision_dt + 1e-6
    with pytest.raises(ValueError):
        apply_action([np.nan, 0, 0], np.zeros(3), cfg.environment)


def test_coordinate_conversion_and_line_crossings():
    np.testing.assert_allclose(ego_relative([1, -2], 0), [1, 2])
    np.testing.assert_allclose(ego_relative([0, 1], np.pi / 2), [1, 0], atol=1e-6)
    assert crosses_line([-1, 0], [1, 0], [0, 0], [1, 0], 2)
    assert not crosses_line([-1, 5], [1, 5], [0, 0], [1, 0], 2)
    assert not crosses_line([1, 0], [-1, 0], [0, 0], [1, 0], 2)


def test_route_progress_cannot_be_farmed_by_reversing_or_jumping_at_crossings():
    route = Route([[0, 0], [10, 0], [20, 0]])
    assert route.update([8, 0]) == 8
    assert route.update([4, 0]) == -2
    assert route.update([8, 0]) == 0
    assert route.update([10, 0]) == 2
    crossing = Route([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [10, 0]])
    crossing.update([5, 0])
    assert crossing.progress == 5


def test_tracks_retain_slots_and_handle_disappearance():
    slots = TrackedSlots(2)
    slots.assign([(10, 1, "a"), (20, 2, "b")])
    assert slots.assign([(20, 0, "b2"), (10, 3, "a2")]) == ["a2", "b2"]
    assert slots.assign([(30, 0, "c"), (20, 1, "b3")]) == ["c", "b3"]
    assert slots.ids == [30, 20]
    slots.assign([(30, 1, "c"), (20, 2, "b3"), (40, -10, "child")])
    assert 40 in slots.ids and 20 not in slots.ids


def test_normalizer_uses_valid_training_values_and_shared_object_scales(cfg):
    episode = make_episode(cfg)
    episode.arrays["continuous"][:, OBJECT_OFFSET + 7:OBJECT_OFFSET + 14] = 100000
    norm = Normalizer.fit([episode])
    assert norm.mean[OBJECT_OFFSET] < 11
    np.testing.assert_allclose(norm.mean[OBJECT_OFFSET:OBJECT_OFFSET + 7], norm.mean[OBJECT_OFFSET + 7:OBJECT_OFFSET + 14])
    normalized = norm.normalize(episode.arrays["continuous"], episode.arrays["present"], episode.arrays["controls"])
    assert np.all(normalized[:, OBJECT_OFFSET + 7:] == 0)
    np.testing.assert_allclose(Normalizer.from_dict(norm.to_dict()).scale, norm.scale)


def test_episode_contract_and_atomic_dataset_roundtrip(cfg, tmp_path):
    dataset = Dataset(tmp_path / "data", cfg, create=True)
    for i in range(10):
        dataset.save(make_episode(cfg, i, terminal=i % 2 == 0))
    assert dataset.next_id() == 10
    assert [len(dataset.load(split)) for split in ("train", "validation", "test")] == [8, 1, 1]
    np.testing.assert_equal(dataset.load()[0].arrays["frames"], np.arange(7) * 2)
    with pytest.raises(FileExistsError):
        dataset.save(make_episode(cfg, 0))
    invalid = make_episode(cfg)
    invalid.arrays["frames"][-1] += 1
    with pytest.raises(ValueError, match="Frame interval"):
        invalid.validate()
    invalid = make_episode(cfg)
    invalid.arrays["continuous"][-1, 5] = 0.4
    with pytest.raises(ValueError, match="action history"):
        invalid.validate()
    changed = copy.deepcopy(cfg)
    changed.reward.collision += 1
    with pytest.raises(ValueError, match="reward"):
        Dataset(tmp_path / "data", changed)


def test_attention_masks_absent_objects_and_ignores_actor_ids(cfg):
    episode = make_episode(cfg)
    normalizer = Normalizer.fit([episode])
    obs = tensor_observation({key: value[:1] for key, value in episode.arrays.items() if key in OBS_KEYS}, normalizer, "cpu")
    model = RSSM(cfg.model)
    encoded = model.encoder(obs)
    changed = {key: value.clone() for key, value in obs.items()}
    changed["continuous"][:, OBJECT_OFFSET + 7:] = 10000
    changed["classes"][:, 1:] = 2
    torch.testing.assert_close(encoded, model.encoder(changed))
    episode.arrays["ids"][:] = 999
    obs_ids_changed = tensor_observation({key: value[:1] for key, value in episode.arrays.items() if key in OBS_KEYS}, normalizer, "cpu")
    torch.testing.assert_close(encoded, model.encoder(obs_ids_changed))
    obs["present"][:] = False
    assert torch.isfinite(model.encoder(obs)).all()


def test_rssm_sequence_updates_attention_memory_and_all_prediction_heads(cfg):
    from world_model_rl.data import sample_sequences
    episode = make_episode(cfg, terminal=True)
    normalizer = Normalizer.fit([episode])
    model = RSSM(cfg.model)
    batch = sample_sequences([episode], 2, 6, np.random.default_rng(0))
    loss, metrics = model.sequence_loss(batch, normalizer, "cpu")
    loss.backward()
    assert np.isfinite(list(metrics.values())).all()
    for module in (model.encoder.attention, model.recurrent, model.prior_net, model.posterior_net,
                   model.continuous_head, model.presence_head, model.class_head,
                   model.reward_head, model.termination_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    replay = real_replay([episode], model, normalizer, "cpu", 3)
    assert replay.arrays["features"].shape == (6, model.feature_size)
    assert not replay.arrays["terminated"][:-1].any()
    assert replay.arrays["terminated"][-1]
    different = copy.deepcopy(episode)
    different.arrays["applied_actions"][0, 0] = 0.8
    assert not np.allclose(model.encode_episode(episode, normalizer, "cpu")[1],
                           model.encode_episode(different, normalizer, "cpu")[1])


def test_short_failure_episodes_are_trained_and_padding_is_ignored(cfg):
    from world_model_rl.data import sample_sequences
    episodes = [make_episode(cfg), make_episode(cfg, 1, steps=1, terminal=True)]
    batch = sample_sequences(episodes, 32, 6, np.random.default_rng(3))
    assert 1 in batch["sequence_valid"].sum(1) and 6 in batch["sequence_valid"].sum(1)
    model = RSSM(cfg.model)
    normalizer = Normalizer.fit(episodes)
    torch.manual_seed(9)
    original, _ = model.sequence_loss(batch, normalizer, "cpu")
    changed = copy.deepcopy(batch)
    changed["rewards"][~batch["sequence_valid"]] = 1e6
    torch.manual_seed(9)
    padded, _ = model.sequence_loss(changed, normalizer, "cpu")
    torch.testing.assert_close(original, padded)
    original.backward()
    assert torch.isfinite(original)


def test_sac_targets_bootstrap_time_limits_and_stop_at_task_termination(cfg):
    sac = SAC(5, cfg, "cpu")
    for net in (sac.q1, sac.q2, sac.target1, sac.target2):
        for parameter in net.parameters():
            torch.nn.init.zeros_(parameter)
    sac.target1[-1].bias.data.fill_(5)
    sac.target2[-1].bias.data.fill_(5)
    batch = {"features": np.zeros((2, 5), np.float32), "next_features": np.zeros((2, 5), np.float32),
             "actions": np.zeros((2, 3), np.float32), "rewards": np.array([2, 2], np.float32),
             "terminated": np.array([True, False])}
    torch.manual_seed(100)
    with torch.no_grad():
        _, log_prob = sac.actor(torch.zeros(2, 5))
        targets = torch.tensor([2, 2]) + cfg.training.gamma * torch.tensor([0, 1]) * (
            5 - cfg.training.initial_alpha * log_prob)
    torch.manual_seed(100)
    metrics = sac.update(batch)
    assert metrics["critic_loss"] == pytest.approx(float(2 * targets.square().mean()), rel=1e-5)
    assert targets[0] == 2 and targets[1] != 2
    assert sac.updates == 1 and np.isfinite(list(metrics.values())).all()


def test_imagination_never_observes_future_and_horizon_does_not_create_terminal(cfg):
    cfg.training.imagination_horizon = 3
    episode = make_episode(cfg)
    # A fixed route may exceed the random-route length limit; observed scaling accommodates it.
    episode.arrays["continuous"][:, 13] += 1000
    episode.arrays["continuous"][:, 16:20] = [80, 80, 30, -25]
    clear_episode = copy.deepcopy(episode)
    clear_episode.arrays["continuous"][:, 16:20] = [0, 0, 0, 60]
    normalizer = Normalizer.fit([episode, clear_episode])
    model = RSSM(cfg.model)
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    model.termination_head[-1].bias.data.fill_(-100)
    sac = SAC(model.feature_size, cfg, "cpu")
    real = real_replay([episode], model, normalizer, "cpu", 7)
    def forbidden(*args, **kwargs):
        raise AssertionError("Imagination must not call the observation encoder")
    model.encoder.forward = forbidden
    synthetic, counts = imagine(model, sac, real, normalizer, cfg, "cpu", np.random.default_rng(2))
    assert len(synthetic) == cfg.training.imagination_starts * 3
    assert counts == {"generated": 24, "accepted": 24}
    assert not synthetic.arrays["terminated"].any()
    np.testing.assert_allclose(synthetic.arrays["next_features"][:, -3:], apply_action(
        synthetic.arrays["actions"], synthetic.arrays["features"][:, -3:], cfg.environment), atol=1e-7)
    h, z = cfg.model.deterministic, cfg.model.stochastic
    np.testing.assert_equal(synthetic.arrays["features"][:, h + z:h + z + 4],
                            synthetic.arrays["next_features"][:, h + z:h + z + 4])
    batch = mixed_batch(real, synthetic, 8, 0.25, np.random.default_rng(4))
    assert len(batch["rewards"]) == 8
    with pytest.raises(ValueError, match="different representation"):
        mixed_batch(real, Replay(synthetic.arrays, 6), 8, 0.25, np.random.default_rng(0))
    assert not real.arrays["terminated"].any()  # The fixture ends by time limit.


def test_validation_reports_event_support_and_blocks_unsupported_imagination(cfg):
    episode = make_episode(cfg)
    model = RSSM(cfg.model)
    norm = Normalizer.fit([episode])
    report = evaluate_model(model, [episode], norm, cfg, "cpu")
    assert list(report["horizons"]) == ["1", "3", "5"]
    assert report["termination"]["negative_count"] == 6
    assert report["termination"]["positive_count"] == 0
    assert report["collision_modeling"] == "unvalidated"
    assert not imagination_gate(report, cfg)[0]
    assert not imagination_gate({"available": False, "reason": "missing"}, cfg)[0]


def test_rain_night_and_child_caution_penalties(cfg):
    e = cfg.environment
    assert weather_limits(10, 80, 80, 0, 60, e) == 7
    assert weather_limits(10, 0, 0, 0, -20, e) == 7
    assert weather_limits(10, 0, 0, 50, -20, e) == 5
    assert visibility_range(80, 0, -20, e) < visibility_range(0, 0, 60, e)
    calc = RewardCalculator(cfg)
    before = {"speed": 12, "acceleration": np.zeros(2), "longitudinal_acceleration": 0.0}
    after = {"speed": 12, "speed_limit": 15, "weather_limit": 8, "child_near": True,
             "acceleration": np.zeros(2), "longitudinal_acceleration": 0.0,
             "offroad": False, "wrong_way": False, "progress_distance": 0}
    penalty, _ = calc.step(before, after, {}, 0.1, np.zeros(3), np.zeros(3))
    assert penalty["unsafe_weather"] < 0
    assert penalty["unsafe_child"] == 0  # Initial reaction/braking allowance.
    for _ in range(30):
        penalty, events = calc.step(before, after, {}, 0.1, np.zeros(3), np.zeros(3))
    assert penalty["unsafe_child"] < 0 and events["unsafe_child"] == 1
    after["speed"] = 3
    penalty, _ = calc.step(before, after, {}, 0.1, np.zeros(3), np.zeros(3))
    assert penalty["unsafe_child"] == penalty["unsafe_weather"] == 0


def test_frame_barrier_rejects_late_events_and_other_ticking_clients(cfg):
    env = CarlaDrivingEnv.__new__(CarlaDrivingEnv)
    env.cfg, env.committed_frame, env.pending_events = cfg, 10, []
    env.imu_queue, env.event_queue = queue.Queue(), queue.Queue()
    env.world = SimpleNamespace(tick=lambda timeout: 12,
        get_snapshot=lambda: SimpleNamespace(frame=12, timestamp=SimpleNamespace(elapsed_seconds=1.0)))
    env.imu_queue.put(SimpleNamespace(frame=12))
    env.event_queue.put(("collision", 11, 1))
    assert env._tick()["collision"] == 1
    env.imu_queue.put(SimpleNamespace(frame=12))
    env.event_queue.put(("collision", 10, 1))
    with pytest.raises(RuntimeError, match="Late event"):
        env._tick()
    env.imu_queue.put(SimpleNamespace(frame=13))
    with pytest.raises(RuntimeError, match="Another client"):
        env._tick()
    env.event_queue.put(("collision", 10, 1))
    with pytest.raises(RuntimeError, match="Late final event"):
        env.finish_episode()


def test_checkpoint_preserves_normalizer_optimizer_policy_and_rng(cfg, tmp_path):
    episode = make_episode(cfg)
    norm = Normalizer.fit([episode])
    model = RSSM(cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    sac = SAC(model.feature_size, cfg, "cpu")
    rng = np.random.default_rng(33)
    state = rng_state(rng)
    expected_numpy, expected_torch = rng.random(4), torch.rand(4)
    restore_rng(state, rng)
    np.testing.assert_equal(rng.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch)
    path = tmp_path / "agent.pt"
    save_checkpoint(path, cfg, model, norm, optimizer, 4, 12, sac=sac.state_dict(), kind="agent")
    payload, loaded, loaded_norm, _ = load_checkpoint(path, cfg, "cpu")
    copied = SAC(loaded.feature_size, cfg, "cpu")
    copied.load_state_dict(payload["sac"])
    test_features = np.zeros((1, model.feature_size), np.float32)
    np.testing.assert_equal(sac.act(test_features, True), copied.act(test_features, True))
    np.testing.assert_equal(norm.mean, loaded_norm.mean)
    assert payload["model_version"] == 4 and payload["model_updates"] == 12


def test_offline_fit_entry_point_with_synthetic_fixture_only(cfg, tmp_path):
    from world_model_rl.runner import fit, validate_dataset
    cfg.training.gate_min_terminal_negative = 1  # Six-step numeric episodes.
    dataset = Dataset(tmp_path / "data", cfg, create=True)
    for identity in range(10):
        dataset.save(make_episode(cfg, identity, terminal=identity % 2 == 0))
    report = fit(cfg, dataset.path, tmp_path / "world_model.pt", updates=1)
    assert report["available"]
    assert validate_dataset(cfg, dataset.path)["transitions"] == 60


def test_environment_preserves_fixed_dt_and_collision_overrides_goal_reward(cfg):
    env = CarlaDrivingEnv.__new__(CarlaDrivingEnv)
    env.cfg = cfg
    env.carla = SimpleNamespace(VehicleControl=lambda **kwargs: kwargs)
    env.ego = SimpleNamespace(apply_control=lambda control: None,
                             bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2)))
    env.previous_action = np.zeros(3, np.float32)
    env.steps, env.frame, env.sim_time, env.committed_frame = 0, 0, 0.0, 0
    env.stuck_time = env.offroad_time = 0.0
    env.ended, env.controls = False, []
    env.route = SimpleNamespace(high_water=1, length=10)
    measure = {"speed": 2.0, "acceleration": np.zeros(2), "longitudinal_acceleration": 0.0, "position": np.zeros(2),
        "front": np.zeros(2), "speed_limit": 15, "weather_limit": 15, "child_near": False,
        "offroad": False, "wrong_way": False, "progress_distance": 0.1, "route_s": 1,
        "success": False, "blocked": False, "lane_offset": 0, "heading_error": 0}
    env.measurement = measure
    def tick(settle=True):
        env.frame += 1
        env.sim_time += cfg.environment.fixed_delta_seconds
        return {"collision": int(env.frame == 1), "illegal_lane": 0}
    def measured():
        obs = Observation.empty()
        obs.continuous[5:8] = env.previous_action
        return obs, {**measure, "position": np.array([env.frame * 0.1, 0]), "success": True}
    env._tick = tick
    env._events_at_boundary = lambda frame: {"collision": 1, "illegal_lane": 0}
    env.extractor = SimpleNamespace(measure=measured)
    env.reward_calculator = RewardCalculator(cfg)
    env.provider = SimpleNamespace(observe=lambda env, obs: obs)
    obs, _, terminated, truncated, info = env.step([0, 0.5, 0])
    assert terminated and not truncated and info["terminal_reason"] == "collision"
    assert env.frame == cfg.environment.action_repeat
    assert env.sim_time == cfg.environment.decision_dt
    assert info["components"]["collision"] == -cfg.reward.collision
    assert info["components"]["success"] == 0
    np.testing.assert_equal(obs.continuous[5:8], info["applied_action"])
    with pytest.raises(RuntimeError, match="Call reset"):
        env.step([0, 0, 1])


def test_orchestration_refresh_resume_and_evaluation_with_offline_env_only(cfg, tmp_path, monkeypatch):
    from world_model_rl import runner
    cfg.training.bootstrap_episodes = 10
    cfg.training.pretrain_updates = 1
    cfg.training.collection_episodes = 2
    cfg.training.gate_min_terminal_negative = 1  # Six-step numeric episodes.
    class OfflineEnv:
        """A deterministic array fixture, not a running CARLA simulation."""
        def __init__(self, config):
            self.cfg = config
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def finish_episode(self):
            pass
        def reset(self, identity, seed=None):
            self.index = 0
            self.previous = np.zeros(3, np.float32)
            self.episode = make_episode(self.cfg, identity)
            self.metadata = {**self.episode.metadata, "observation_source": "offline_fixture"}
            self.metadata["weather"] = self.cfg.environment.weather[identity % len(self.cfg.environment.weather)]
            self.metadata["seed"] = seed if seed is not None else self.cfg.training.seed + identity
            return self.observation(), {"frame": 0, "sim_time": 0}
        def observation(self):
            obs = Observation(**{key: self.episode.arrays[key][self.index].copy() for key in OBS_KEYS})
            obs.continuous[5:8] = self.previous
            return obs
        def expert_action(self, rng=None):
            return np.array([0.1, 0.5, 0], np.float32)
        def step(self, requested):
            self.previous = apply_action(requested, self.previous, self.cfg.environment)
            self.index += 1
            ended = self.index == len(self.episode)
            return self.observation(), 1.0, ended, False, {
                "frame": self.index * self.cfg.environment.action_repeat,
                "sim_time": self.index * self.cfg.environment.decision_dt,
                "applied_action": self.previous, "events": {**dict.fromkeys(EVENTS, 0), "success": int(ended)},
                "components": {"progress": 1.0}, "terminal_reason": "success" if ended else "running",
                "route_completion": 0.1, "distance_metres": 0.2, "lane_offset": 0.0,
                "heading_error": 0.0, "acceleration_mps2": 0.0, "jerk_mps3": 0.0,
                "steering_rate_per_s": 0.0}
    monkeypatch.setattr(runner, "CarlaDrivingEnv", OfflineEnv)
    # Model quality is independently tested; orchestration uses an explicit mocked pass.
    monkeypatch.setattr(runner, "imagination_gate", lambda report, config: (True, "offline fixture gate"))
    output = tmp_path / "run"
    runner.run(cfg, output, episodes=2)
    first, _, norm, _ = load_checkpoint(output / "agent.pt", cfg, "cpu")
    assert first["training_episodes"] == 2 and first["model_version"] == 2
    assert first["termination_fit"] == first["validation"]["termination_fit"]
    runner.run(cfg, output, episodes=1, resume=True)
    second, _, norm_after, _ = load_checkpoint(output / "agent.pt", cfg, "cpu")
    assert second["next_episode_id"] == 13 and second["model_version"] == 3
    assert second["sac"]["updates"] == 6
    recorded = Dataset(output / "data", cfg).load()
    assert [ep.metadata["policy_updates"] for ep in recorded[-3:]] == [3, 3, 6]
    # Resume an interrupted two-episode collection phase using an optimizer budget.
    result = runner.run(cfg, output, policy_updates=2, resume=True)
    third, _, _, _ = load_checkpoint(output / "agent.pt", cfg, "cpu")
    assert result["additional_policy_updates"] == 2 and result["additional_collection_episodes"] == 1
    assert third["next_episode_id"] == 14 and third["model_version"] == 4
    assert third["block_updates_done"] == 2 and third["collection_remaining"] == 0
    assert third["termination_fit"] == third["validation"]["termination_fit"]
    np.testing.assert_equal(norm.mean, norm_after.mean)  # Fixed scaling across representation refreshes.
    report = runner.evaluate(cfg, output / "agent.pt", output / "evaluation.json", episodes=2)
    assert len(report["episodes"]) == 2
    assert report["episodes"][0]["seed"] == cfg.training.seed + 100000


def test_perception_caution_does_not_leak_ground_truth_child_presence(cfg):
    from world_model_rl.observations import perception_caution
    observation = Observation.empty()
    observation.continuous[11:13] = [12, cfg.environment.child_speed_mps]
    observation.continuous[19] = 60
    result = perception_caution(observation, cfg.environment)
    assert result.continuous[12] == 12  # Oracle target is removed when predictions contain no child.
    result.present[6], result.classes[6], result.ids[6] = True, 2, 10
    result.continuous[OBJECT_OFFSET + 6 * 7:OBJECT_OFFSET + 6 * 7 + 7] = [20, 2, 0, 0, 0.3, 0.2, 0.9]
    assert perception_caution(result, cfg.environment).continuous[12] == pytest.approx(cfg.environment.child_speed_mps)


def test_oracle_child_visibility_uses_field_of_view_occlusion_and_weather(cfg):
    carla = pytest.importorskip("carla")
    from world_model_rl.observations import OracleExtractor
    transform = carla.Transform()
    child_transform = carla.Transform(carla.Location(x=20, y=2))
    ego = SimpleNamespace(id=1, type_id="vehicle.test", get_transform=lambda: transform,
        get_location=lambda: transform.location, get_velocity=lambda: carla.Vector3D(x=5),
        get_acceleration=lambda: carla.Vector3D(), get_angular_velocity=lambda: carla.Vector3D(),
        get_speed_limit=lambda: 36, bounding_box=carla.BoundingBox(carla.Location(), carla.Vector3D(2, 1, 1)))
    child = SimpleNamespace(id=2, type_id="walker.pedestrian.0009", attributes={"age": "child"},
        get_transform=lambda: child_transform, get_location=lambda: child_transform.location,
        get_velocity=lambda: carla.Vector3D(x=1),
        bounding_box=carla.BoundingBox(carla.Location(z=0.7), carla.Vector3D(0.3, 0.2, 0.7)))
    road = SimpleNamespace(transform=transform, lane_width=3.5)
    weather = carla.WeatherParameters(sun_altitude_angle=60)
    hits = []
    world = SimpleNamespace(get_actors=lambda: [ego, child], get_weather=lambda: weather,
                            cast_ray=lambda start, end: hits)
    env = SimpleNamespace(cfg=cfg, carla=carla, ego=ego, world=world,
        map=SimpleNamespace(get_waypoint=lambda *args, **kwargs: road),
        previous_action=np.zeros(3, np.float32), route=Route([[0, 0], [100, 0]]),
        upcoming_controls=lambda: {"light": None, "stop": None})
    extractor = OracleExtractor(env)
    observation, measurement = extractor.measure()
    assert measurement["child_near"]
    assert observation.classes[6] == 2 and observation.present[6]
    assert observation.continuous[12] == pytest.approx(cfg.environment.child_speed_mps)
    np.testing.assert_allclose(observation.continuous[OBJECT_OFFSET + 6 * 7:OBJECT_OFFSET + 6 * 7 + 4], [20, -2, -4, 0])
    hits.append(SimpleNamespace(location=carla.Location(x=10, y=1, z=1)))
    assert not extractor.measure()[1]["child_near"]
    hits.clear()
    child_transform.location.x = -20
    assert not extractor.measure()[1]["child_near"]
    child_transform.location.x = 20
    weather.sun_altitude_angle, weather.precipitation, weather.fog_density = -25, 80, 30
    observation, measurement = extractor.measure()
    assert not measurement["child_near"]  # Beyond the reduced rain/fog/night visibility range.
    assert observation.continuous[12] == 5

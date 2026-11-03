from contextlib import nullcontext
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from dreamerv3_attention import __main__ as cli
from dreamerv3_attention.config import Config
from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingState, RewardConfig, compute_reward
from dreamerv3_attention.evaluation import evaluate, compare_results
from dreamerv3_attention.model import Agent, expert_actor_loss, corrective_steering_loss
from dreamerv3_attention.synthetic import SyntheticEnv
from dreamerv3_attention.tests.test_environment import fake_environment
from dreamerv3_attention.tests.test_model import small_config, synthetic_batch
from dreamerv3_attention.training import BudgetExpired, Trainer


def test_controller_uses_navigation_and_fails_safe():
    obs = {"tokens": np.zeros((5, 16)), "mask": np.ones(5, dtype=bool)}
    obs["tokens"][0, 15] = 1
    obs["tokens"][1, :2] = [.05, .02]
    obs["tokens"][1, 15] = -1
    obs["tokens"][2, :2] = [.15, .02]
    obs["tokens"][2, 15] = -1
    controller = RouteController()
    action, _ = controller.act(obs, None)
    assert action[0] > 0 and action[1] > 0
    obs["tokens"][0, 0] = 5/30
    assert controller.act(obs, None)[0][1] < 0
    obs["mask"][2] = False
    np.testing.assert_array_equal(controller.act(obs, None)[0], [0, -1])


def test_training_controller_uses_resolved_driving_config(monkeypatch, tmp_path):
    import dreamerv3_attention.controller as controller_module

    received = {}

    class CapturingController(controller_module.RouteController):
        def __init__(self, target_speed=None, config=None):
            received["target_speed"] = target_speed
            received["config"] = config
            super().__init__(target_speed, config)

    monkeypatch.setattr(controller_module, "RouteController", CapturingController)
    cfg = small_config(steps=4, max_episode_steps=6, max_objects=0,
                       warmup_policy="controller",
                       driving={"default_speed_mps": 6.0, "min_speed_limit_confidence": 0.8})
    cfg.warmup_frames = 8
    trainer = Trainer(cfg, 36)
    trainer.run(SyntheticEnv(cfg), tmp_path / "collection", collect_only=True)
    assert received["target_speed"] is None
    assert received["config"].default_speed_mps == 6.0
    assert received["config"].min_speed_limit_confidence == 0.8


def test_synthetic_env_uses_reward_overrides_and_route_highwater():
    cfg = small_config(dt=.1, max_episode_steps=200, max_objects=0,
                       reward={"waypoint": 7., "lane_centering": 2., "progress": 0.,
                               "lane": 0., "target_tracking": 0., "time": 0., "idle": 0.})
    env = SyntheticEnv(cfg)
    env.reward_progress_highwater = 33.
    env.reset()
    assert env.reward_progress_highwater == 0.
    assert env.reward_config.waypoint == 7.
    assert env.reward_config.lane_centering == 2.
    reached = 0
    waypoint_reward = 0.
    for _ in range(100):
        _, _, terminated, _, info = env.step((0., 1.))
        reached += info["waypoints_reached"]
        waypoint_reward += info["reward_components"]["waypoint"]
        if terminated or env.state.route_progress_m >= 31.:
            break
    assert reached == 2
    assert waypoint_reward == 14.
    env.close()


@pytest.mark.parametrize("event", ("collision", "pedestrian_collision"))
def test_synthetic_env_terminates_on_collision_events(event):
    env = SyntheticEnv(small_config(dt=.1, max_episode_steps=10, max_objects=0))
    env.reset()
    setattr(env.state, event, True)
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert terminated and not truncated
    assert info["reward_components"]["collision"] == -100.
    env.close()


def test_idle_reward_accepts_low_target_and_exempts_required_stop():
    cfg = RewardConfig()
    state = DrivingState(speed_limit_mps=8., speed_mps=.4, observations_valid=True)
    reward = compute_reward(state, (0., 0.), state, .05, target_speed_mps=.4, config=cfg)
    assert reward.components["idle"] == 0
    stopped = replace(state, speed_mps=0.)
    assert compute_reward(stopped, (0., 0.), stopped, .05,
                          target_speed_mps=.4).components["idle"] < 0
    assert compute_reward(stopped, (0., 0.), stopped, .05,
                          target_speed_mps=0.).components["idle"] == 0
    assert compute_reward(stopped, (0., -1.), stopped, .05,
                          target_speed_mps=.4, mandated_stop=True).components["idle"] == 0


def test_radar_safety_inputs_are_separate_from_oracle():
    env, bundle = fake_environment()
    bundle.radars["left"] = np.asarray([[1., 1., .1, 0.]], dtype=np.float32)
    state, _ = env._observe(42)
    assert state.side_hazard_m == 0
    assert env.oracle_state.side_hazard_m is None
    assert env.radar_hazards["left"]["clearance_m"] == 0
    np.testing.assert_allclose(env.radar_hazards["left"]["point_ego_m"], [1, 1, .1])


def test_checkpoint_freezes_rewards_and_rejects_changed_or_legacy_resume(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(dt=.025)
    trainer = Trainer(cfg, 5)
    checkpoint = tmp_path / "latest.pt"
    trainer.checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["config"]["reward"] == asdict(RewardConfig())
    assert saved["config"]["camera"]["sensor_tick"] == .025
    saved["schema"] = "carla-dreamerv3-tokens-v1"
    torch.save(saved, checkpoint)
    assert Trainer.restore(checkpoint).frames == 0
    saved["schema"] = "carla-dreamerv3-tokens-v2"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="reward"):
        Trainer.restore(checkpoint, replace(cfg, reward={"progress": 99.}))
    saved["config"]["reward"] = {}
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        Trainer.restore(checkpoint, cfg)
    assert Trainer.restore(checkpoint).frames == 0  # Historical inference remains possible.


def test_baseline_cli_uses_exact_route_and_writes_trace(monkeypatch, tmp_path):
    cfg = small_config(steps=12, max_episode_steps=6, max_objects=0)
    path = tmp_path / "config.json"
    cfg.save(path)
    seen = []

    def factory(run_cfg, scenario):
        seen.append(scenario)
        env = SyntheticEnv(run_cfg)
        env.close = lambda: None
        return env

    monkeypatch.setattr(cli, "CarlaEnv", factory)
    monkeypatch.setattr(cli, "workload_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *a: None)
    output = tmp_path / "baseline"
    result = cli.main(["baseline", "--config", str(path), "--output", str(output),
                       "--budget-seconds", "30"])
    assert len(seen) == 1 and seen[0]["name"] == "basic"
    assert seen[0]["route_start"] == cfg.route_start and seen[0]["weather"] == cfg.weather
    assert result["scope"] == "fixed_training_sanity" and result["checkpoint"] is None
    assert len((output / "trace.jsonl").read_text().splitlines()) == 6
    metrics = result["episodes"][0]["metrics"]
    assert 0 <= metrics["stopped_fraction"] <= 1
    with pytest.raises(ValueError, match="complete"):
        cli.check_baseline(cfg, result)
    result["episodes"][0]["success"] = True
    cli.check_baseline(cfg, result)
    cli.check_baseline(replace(cfg, capacity=20000), result)
    resized = {**result, "config": {**result["config"], "capacity": 20000}}
    assert compare_results(result, resized, fixed_scenario=True)["episode_count"] == 1
    with pytest.raises(ValueError, match="match"):
        cli.check_baseline(replace(cfg, weather="ClearNight"), result)
    assert compare_results(result, result, fixed_scenario=True)["episode_count"] == 1
    with pytest.raises(ValueError, match="configs identical"):
        compare_results(result, {**result, "config": {**result["config"], "weather": "ClearNight"}},
                        fixed_scenario=True)


def test_fixed_evaluation_rejects_multiple_scenarios():
    cfg = Config(scenarios=["train_heavy_rain", "train_low_light"])
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(cfg, RouteController(), fixed_scenario=True)


def test_replay_and_recurrent_carry_use_applied_action(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(steps=4, max_episode_steps=6, max_objects=0)
    trainer = Trainer(cfg, 36)
    env = SyntheticEnv(cfg)
    original_step = env.step
    env.step = lambda requested: original_step(np.asarray([0., -.4], dtype=np.float32))
    original_act = trainer.agent.act
    observed_carries = []

    def act(observation, carry):
        observed_carries.append(carry["previous_action"].clone())
        return original_act(observation, carry)

    trainer.agent.act = act
    trainer.run(env, tmp_path / "collection", collect_only=True)
    torch.testing.assert_close(torch.stack(observed_carries[1:]),
                               torch.tensor([[0., -.4]]).expand(3, 2))
    sample = trainer.replay.sample()
    torch.testing.assert_close(sample["action"], torch.tensor([[0., -.4]]).expand(sample.numel(), 2))
    assert not sample["next", "terminated"].any()


def test_controller_warmup_populates_moving_actions_before_learning(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(steps=4, max_episode_steps=6, max_objects=0)
    cfg = replace(cfg, warmup_frames=8, warmup_policy="controller", actor_bc_weight=1.)
    env = SyntheticEnv(cfg)
    reset, step = env.reset, env.step

    def navigation(obs):
        obs["tokens"][0, 15] = 1
        obs["tokens"][1, :2] = [.05, 0]
        obs["tokens"][1, 15] = -1
        obs["tokens"][2, :2] = [.15, 0]
        obs["tokens"][2, 15] = -1
        obs["mask"][:3] = True
        return obs

    env.reset = lambda: navigation(reset())

    def advance(action):
        obs, *rest = step(action)
        return (navigation(obs), *rest)

    env.step = advance
    trainer = Trainer(cfg, 36)
    result = trainer.run(env, tmp_path / "seeded")
    rows = [json.loads(line) for line in (tmp_path / "seeded" / "metrics.jsonl").read_text().splitlines()]
    assert result["updates"] == 0
    assert all(row["action_source"] == "controller_warmup" for row in rows)
    assert all(row["applied_action"][1] > 0 for row in rows)
    assert trainer.replay[:]["expert"].all()
    assert Trainer.restore(tmp_path / "seeded" / "latest.pt").replay[:]["expert"].all()


def test_budget_stop_saves_checkpoint_and_reports_reason(tmp_path):
    cfg = small_config(steps=4, max_episode_steps=6, max_objects=0)
    env = SyntheticEnv(cfg)
    step = env.step

    def advance(action):
        if env.steps == 2:
            raise BudgetExpired("test budget")
        return step(action)

    env.step = advance
    trainer = Trainer(cfg, 36)
    result = trainer.run(env, tmp_path / "bounded", collect_only=True)
    assert result["frames"] == 2 and result["stop_reason"] == "budget_exhausted"
    assert Trainer.restore(tmp_path / "bounded" / "latest.pt").frames == 2


def test_world_model_bootstrap_preserves_actor_until_scheduled(tmp_path):
    torch.set_num_threads(1)
    cfg = replace(small_config(steps=5, max_episode_steps=6, max_objects=0, updates_per_step=1.),
                  world_model_warmup_updates=2, warmup_policy="controller", actor_bc_weight=1.)
    trainer = Trainer(cfg, 36)
    actor_before = [p.detach().clone() for p in trainer.agent.actor.parameters()]
    trainer.run(SyntheticEnv(cfg), tmp_path / "bootstrap")
    assert trainer.updates == 2
    for p, before in zip(trainer.agent.actor.parameters(), actor_before):
        torch.testing.assert_close(p, before)
    rows = [json.loads(line) for line in (tmp_path / "bootstrap" / "metrics.jsonl").read_text().splitlines()]
    assert [row["learner_mode"] for row in rows if "learner_mode" in row] == ["world_model"] * 2
    continued_cfg = replace(cfg, steps=6)
    resumed = Trainer.restore(tmp_path / "bootstrap" / "latest.pt", continued_cfg)
    resumed.run(SyntheticEnv(continued_cfg), tmp_path / "continued")
    rows = [json.loads(line) for line in (tmp_path / "continued" / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["learner_mode"] == "dreamer"
    assert "actor_bc_loss" in rows[0]
    assert abs(rows[0]["actor_bc_loss"]) > 1e-6


def test_expert_action_alignment_uses_next_action_and_skips_terminals():
    from tensordict import TensorDict

    seen = {}

    class Policy:
        def get_dist(self, features):
            seen["features"] = features["state"].clone()
            return self

        def log_prob(self, action):
            seen["actions"] = action.clone()
            return action.sum(-1)

    posterior = TensorDict({"state": torch.tensor([[[10.], [20.], [30.]]]),
                            "belief": torch.zeros(1, 3, 1)}, [1, 3])
    sample = TensorDict({"action": torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]]),
                         "expert": torch.tensor([[True, True, True]]),
                         "next": {"done": torch.tensor([[[False], [True], [False]]])}}, [1, 3])
    loss = expert_actor_loss(Policy(), posterior, sample)
    torch.testing.assert_close(seen["features"], torch.tensor([[10.]]))
    torch.testing.assert_close(seen["actions"], torch.tensor([[.3, .4]]))
    torch.testing.assert_close(loss, torch.tensor(-.7))


def test_expert_supervision_updates_actor_without_world_model_gradients():
    from tensordict import TensorDict

    cfg = small_config()
    agent = Agent(cfg, 5)
    state = torch.randn(2, 3, agent.state_dim, requires_grad=True)
    belief = torch.randn(2, 3, cfg.belief, requires_grad=True)
    posterior = TensorDict({"state": state, "belief": belief}, [2, 3])
    actions = torch.zeros(2, 3, 2, requires_grad=True)
    sample = TensorDict({"action": actions, "expert": torch.ones(2, 3, dtype=torch.bool),
                         "next": {"done": torch.zeros(2, 3, 1, dtype=torch.bool)}}, [2, 3])
    loss = expert_actor_loss(agent.actor, posterior, sample)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert state.grad is None and belief.grad is None and actions.grad is None
    assert all(p.grad is None for p in agent.prior.parameters())


def test_expert_refresh_is_labelled_and_resumes_episode_schedule(tmp_path):
    torch.set_num_threads(1)
    cfg = replace(small_config(steps=12, max_episode_steps=4, max_objects=0),
                  warmup_policy="controller", actor_bc_weight=1., expert_episode_interval=2)
    trainer = Trainer(cfg, 36)
    trainer.run(SyntheticEnv(cfg), tmp_path / "refresh", collect_only=True)
    rows = [json.loads(line) for line in (tmp_path / "refresh" / "metrics.jsonl").read_text().splitlines()]
    assert [row["action_source"] for row in rows] == (
        ["controller_warmup"] * 4 + ["actor"] * 4 + ["controller_refresh"] * 4)
    flags = trainer.replay[:]["expert"].tolist()
    assert flags == [True]*4 + [False]*4 + [True]*4
    cfg = replace(cfg, steps=16)
    resumed = Trainer.restore(tmp_path / "refresh" / "latest.pt", cfg)
    resumed.run(SyntheticEnv(cfg), tmp_path / "continued_refresh", collect_only=True)
    rows = [json.loads(line) for line in (tmp_path / "continued_refresh" / "metrics.jsonl").read_text().splitlines()]
    assert all(row["action_source"] == "actor" for row in rows)


def test_corrective_steering_matches_controller_and_does_not_supervise_throttle():
    from tensordict import TensorDict

    class Policy:
        loc = torch.zeros(2, requires_grad=True)

        def get_dist(self, features):
            assert not features["state"].requires_grad
            features["loc"] = self.loc.expand(*features.batch_size, 2)
            features["scale"] = torch.ones_like(features["loc"])

    tokens = torch.zeros(1, 3, 5, 16)
    tokens[..., 0, 15] = 1
    tokens[..., 1:3, 15] = -1
    tokens[..., 1, 0] = .05
    tokens[0, :, 1, 1] = torch.tensor([-.02, 0., .02])
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    sample = TensorDict({"next": {"tokens": tokens, "mask": mask,
                                   "done": torch.tensor([[[False], [False], [True]]])}}, [1, 3])
    posterior = TensorDict({"state": torch.randn(1, 3, 2, requires_grad=True),
                            "belief": torch.randn(1, 3, 2, requires_grad=True)}, [1, 3])
    policy = Policy()
    loss = corrective_steering_loss(policy, posterior, sample)
    reference = [RouteController().act({"tokens": tokens[0, i].numpy(), "mask": mask[0, i].numpy()}, None)[0][0]
                 for i in range(2)]
    expected = ((torch.tensor(reference) / 0.1) ** 2).mean() + 10. * (0.9 ** 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert policy.loc.grad[0] > 0 and policy.loc.grad[1] == 0
    assert posterior["state"].grad is None


def test_corrective_steering_penalizes_wide_sigma_but_not_a_matching_narrow_policy():
    from tensordict import TensorDict

    tokens = torch.zeros(1, 1, 3, 16)
    tokens[..., 0, 15] = 1.
    tokens[..., 1:3, 15] = -1.
    tokens[..., 1, :2] = torch.tensor([.05, -.02])
    sample = TensorDict({"next": {"tokens": tokens, "mask": torch.ones(1, 1, 3, dtype=torch.bool),
                                   "done": torch.zeros(1, 1, 1, dtype=torch.bool)}}, [1, 1])
    posterior = TensorDict({"state": torch.zeros(1, 1, 2)}, [1, 1])
    label = RouteController().act({"tokens": tokens[0, 0].numpy(), "mask": np.ones(3, bool)}, None)[0][0]

    class Policy:
        def __init__(self, sigma):
            self.loc, self.sigma = torch.atanh(torch.tensor(float(label))), sigma

        def get_dist(self, features):
            features["loc"] = torch.stack((self.loc, torch.zeros(())), -1).expand(*features.batch_size, 2)
            features["scale"] = torch.full_like(features["loc"], self.sigma)

    assert corrective_steering_loss(Policy(.05), posterior, sample) < 1e-6
    assert corrective_steering_loss(Policy(1.), posterior, sample) > 1.


def test_corrective_term_reaches_real_stepper():
    torch.set_num_threads(1)
    cfg = small_config(actor_steering_bc_weight=1.)
    agent = Agent(cfg, 5)
    sample = synthetic_batch(cfg)
    sample["next", "tokens"][..., 0, 15] = 1
    sample["next", "tokens"][..., 1, 15] = -1
    sample["next", "tokens"][..., 1, :2] = torch.tensor([.05, .02])
    sample["next", "mask"][..., :2] = True
    result = agent.update(sample)
    assert torch.isfinite(result["actor_steering_bc_loss"])
    assert abs(float(result["actor_steering_bc_loss"])) > 1e-6


def test_corrective_steering_keeps_straight_approach_despite_far_turn():
    from tensordict import TensorDict

    class Policy:
        loc = torch.zeros(2, requires_grad=True)

        def get_dist(self, features):
            features["loc"] = self.loc.expand(*features.batch_size, 2)
            features["scale"] = torch.ones_like(features["loc"])

    tokens = torch.zeros(1, 1, 4, 16)
    tokens[..., 0, 15] = 1.
    tokens[..., 1:3, 15] = -1.
    tokens[..., 1, :2] = torch.tensor([.05, 0.])
    tokens[..., 2, :2] = torch.tensor([.10, .05])
    sample = TensorDict({"next": {"tokens": tokens, "mask": torch.ones(1, 1, 4, dtype=torch.bool),
                                   "done": torch.zeros(1, 1, 1, dtype=torch.bool)}}, [1, 1])
    posterior = TensorDict({"state": torch.zeros(1, 1, 2)}, [1, 1])
    policy = Policy()
    loss = corrective_steering_loss(policy, posterior, sample)
    loss.backward()
    torch.testing.assert_close(policy.loc.grad, torch.zeros(2))
    sample["next", "mask"][..., 1] = False
    assert corrective_steering_loss(policy, posterior, sample) == 0.
    sample["next", "mask"][..., 1] = True
    sample["next", "tokens"][..., 1, 0] = float("nan")
    assert corrective_steering_loss(policy, posterior, sample) == 0.


def test_corrective_objective_fork_preserves_actor_without_flags(tmp_path):
    cfg = small_config()
    trainer = Trainer(cfg, 5)
    path = tmp_path / "source.pt"
    trainer.checkpoint(path)
    changed = replace(cfg, actor_steering_bc_weight=1.)
    fork = Trainer.restore(path, changed)
    assert fork.supervision_fork == str(path.resolve())
    assert fork.cfg.actor_steering_bc_weight == 1.
    for p, q in zip(trainer.agent.actor.parameters(), fork.agent.actor.parameters()):
        torch.testing.assert_close(p, q)


def test_evaluation_preserves_tensor_device_without_container_device():
    from tensordict import TensorDict

    class Agent:
        def initial(self):
            # Meta tensors reproduce a non-CPU state without requiring a GPU.
            return TensorDict({"state": torch.empty(1, device="meta"),
                               "belief": torch.empty(1, device="meta")}, [])

        def act(self, observation, carry, deterministic):
            assert carry.device is None
            if "is_init" in carry:
                assert carry["is_init"].device == carry["state"].device
                assert carry["previous_action"].device == carry["state"].device
            return np.zeros(2, dtype=np.float32), carry

    cfg = small_config(max_episode_steps=6, max_objects=0)

    def factory(cfg, scenario):
        env = SyntheticEnv(cfg)
        env.close = lambda: None
        return env

    result = evaluate(cfg, Agent(), factory, fixed_scenario=True)
    assert result["episodes"][0]["metrics"]["steps"] == 6


def test_synthetic_reward_cutoff_and_reset(monkeypatch):
    from types import SimpleNamespace
    from dreamerv3_attention import synthetic
    rewards = iter([-500., -.1])
    monkeypatch.setattr(synthetic, "compute_reward", lambda *args, **kwargs:
                        SimpleNamespace(reward=next(rewards), components={}))
    env = SyntheticEnv(small_config(max_episode_steps=2))
    env.reset()
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert not terminated and not truncated and info["episode_return"] == -500.
    _, _, terminated, truncated, info = env.step((0., 0.))
    assert terminated and not truncated and info["termination_reason"] == "reward_limit"
    env.reset()
    assert env.episode_return == 0.

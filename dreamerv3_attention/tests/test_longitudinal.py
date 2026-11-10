"""Offline checks for speed supervision; no CARLA server or training job."""

from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict

from dreamerv3_attention.config import Config
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor, compute_target_speed
from dreamerv3_attention.model import Agent, corrective_longitudinal_loss
from dreamerv3_attention.tests.test_environment import fake_step_environment
from dreamerv3_attention.tests.test_model import small_config, synthetic_batch
from dreamerv3_attention.training import Trainer


@pytest.mark.parametrize("speed,target_sign", [(0., 1), (3., -1)])
def test_reference_teaches_acceleration_and_braking_without_changing_supervisor(speed, target_sign):
    supervisor = SafetySupervisor(DrivingConfig(max_speed_mps=2.))
    state = DrivingState(speed_mps=speed, speed_limit_mps=8.)
    before = vars(supervisor).copy()
    reference = supervisor.longitudinal_reference(state, .05)
    assert reference * target_sign > 0.
    assert vars(supervisor) == before
    assert state.speed_mps == speed


@pytest.mark.parametrize("settings", [
    {"red_light": True, "red_light_distance_m": 1.},
    {"yellow_light": True, "yellow_light_distance_m": 2., "yellow_light_confidence": 1.},
    {"pedestrian_conflict": True},
    {"crossing_occupied": True},
    {"front_hazard_m": 1.},
    {"crossing_id": "crosswalk", "crossing_distance_m": 1.},
    {"observations_valid": False},
])
def test_reference_keeps_required_stops_and_does_not_advance_phase(settings):
    supervisor = SafetySupervisor()
    state = DrivingState(speed_limit_mps=8., **settings)
    before = vars(supervisor).copy()
    assert supervisor.longitudinal_reference(state, .05) == -1.
    assert vars(supervisor) == before


def test_reference_uses_live_crossing_resume_phase_and_releases_after_green():
    supervisor = SafetySupervisor()
    supervisor.crossing_phase = "resume"
    supervisor._crossing_id = "crosswalk"
    state = DrivingState(speed_limit_mps=8., crossing_id="crosswalk", crossing_distance_m=1.)
    before = vars(supervisor).copy()
    assert supervisor.longitudinal_reference(state, .05) > 0.
    assert vars(supervisor) == before
    state = replace(state, crossing_id=None, crossing_distance_m=None,
                    red_light=True, red_light_distance_m=1.)
    supervisor.step(state, (0., 1.), .05)
    assert supervisor._red_stopping
    assert supervisor.longitudinal_reference(replace(state, red_light=False), .05) > 0.
    assert supervisor._red_stopping  # Copy must not clear the live latch.


def test_speed_cap_applies_to_trusted_limits_and_preserves_hazard_reductions():
    state = DrivingState(speed_limit_mps=12.)
    assert compute_target_speed(state) == 12.
    cfg = DrivingConfig(max_speed_mps=2.)
    assert compute_target_speed(state, cfg) == 2.
    assert compute_target_speed(replace(state, front_hazard_m=5.), cfg) == .5
    assert compute_target_speed(replace(state, rain=1.), cfg) < 2.
    assert compute_target_speed(replace(state, red_light=True, red_light_distance_m=0.), cfg) == 0.


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_longitudinal_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match="actor_longitudinal_bc_weight"):
        Config(actor_longitudinal_bc_weight=weight).validate()


@pytest.mark.parametrize("cap", [-1., float("nan"), float("inf"), True])
def test_speed_cap_validation(cap):
    with pytest.raises(ValueError):
        Config(driving={"max_speed_mps": cap}).validate()


def test_environment_labels_next_policy_state_without_oracle_signal_leakage():
    before = DrivingState(speed_limit_mps=8.)
    after = DrivingState(speed_limit_mps=8., red_light=True, red_light_distance_m=20.)
    env = fake_step_environment(before=before, after=after, max_steps=100, timeout_s=10., blocked_timeout_s=20.)
    env.cfg = replace(env.cfg, actor_longitudinal_bc_weight=4.)
    env.supervisor = SafetySupervisor()
    observe = env._observe

    def policy_observe(frame):
        _, obs = observe(frame)
        return replace(after, red_light=False, red_light_distance_m=None), obs

    env._observe = policy_observe
    _, _, terminated, truncated, info = env.step((0., -1.))
    assert not terminated and not truncated
    assert info["applied_action"][1] == -1.  # Actor action is not replaced.
    assert info["red_light"]  # Oracle telemetry still sees the light.
    assert info["longitudinal_target"] > 0.
    assert info["longitudinal_target_valid"]


@pytest.mark.parametrize("valid,max_steps", [(False, 100), (True, 1)])
def test_environment_excludes_invalid_and_terminal_next_states(valid, max_steps):
    env = fake_step_environment(after=DrivingState(observations_valid=valid), max_steps=max_steps)
    env.cfg = replace(env.cfg, actor_longitudinal_bc_weight=4.)
    # An invalid/terminal label must not call the reference policy at all.
    env.supervisor.longitudinal_reference = lambda *_: pytest.fail("invalid reference")
    info = env.step((0., 0.))[-1]
    assert not info["longitudinal_target_valid"]
    assert info["longitudinal_target"] == 0.


class Policy:
    def __init__(self):
        self.loc = torch.zeros(2, requires_grad=True)
        self.scale = torch.ones(2, requires_grad=True)
        self.seen = None

    def get_dist(self, features):
        self.seen = features["state"].clone()
        features["loc"] = self.loc.expand(*features.batch_size, 2)
        features["scale"] = self.scale.expand(*features.batch_size, 2)


def test_longitudinal_loss_alignment_masking_and_detached_actor_only_gradient():
    state = torch.tensor([[[10.], [20.], [30.], [40.], [50.]]], requires_grad=True)
    targets = torch.tensor([[[.6], [-.4], [1.], [float("nan")], [-1.]]], requires_grad=True)
    posterior = TensorDict({"state": state, "belief": torch.zeros_like(state)}, [1, 5])
    sample = TensorDict({"expert": torch.zeros(1, 5, dtype=torch.bool),
                         "next": {"longitudinal_target": targets,
                                  "longitudinal_target_valid": torch.tensor([[[True], [True], [False], [True], [True]]]),
                                  "done": torch.tensor([[[False], [False], [False], [False], [True]]])}}, [1, 5])
    policy = Policy()
    loss = corrective_longitudinal_loss(policy, posterior, sample)
    expected = -torch.distributions.Normal(0., 1.).log_prob(torch.atanh(torch.tensor([.6, -.4]))).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(policy.seen, torch.tensor([[10.], [20.]]))
    loss.backward()
    assert policy.loc.grad[0] == 0 and policy.loc.grad[1] != 0
    assert policy.scale.grad[0] == 0
    assert state.grad is None and targets.grad is None


@pytest.mark.parametrize("target,gradient_sign", [(.9, -1), (-.9, 1)])
def test_longitudinal_gradient_moves_actor_towards_acceleration_or_braking(target, gradient_sign):
    policy = Policy()
    posterior = TensorDict({"state": torch.zeros(1, 1, 1)}, [1, 1])
    sample = TensorDict({"next": {"longitudinal_target": torch.tensor([[[target]]]),
                                  "longitudinal_target_valid": torch.ones(1, 1, 1, dtype=torch.bool),
                                  "done": torch.zeros(1, 1, 1, dtype=torch.bool)}}, [1, 1])
    corrective_longitudinal_loss(policy, posterior, sample).backward()
    assert policy.loc.grad[1] * gradient_sign > 0
    assert policy.loc.grad[0] == 0


def test_missing_or_invalid_targets_do_not_invent_supervision():
    posterior = TensorDict({"state": torch.zeros(1, 1, 1)}, [1, 1])
    policy = Policy()
    assert corrective_longitudinal_loss(policy, posterior, TensorDict({}, [1, 1])) == 0.
    sample = TensorDict({"next": {"longitudinal_target": torch.tensor([[[2.]]]),
                                  "longitudinal_target_valid": torch.ones(1, 1, 1, dtype=torch.bool),
                                  "done": torch.zeros(1, 1, 1, dtype=torch.bool)}}, [1, 1])
    assert corrective_longitudinal_loss(policy, posterior, sample) == 0.
    assert policy.seen is None


def test_longitudinal_term_is_connected_to_real_dreamer_loss_without_optimizer_step():
    torch.set_num_threads(1)
    cfg = small_config(actor_longitudinal_bc_weight=4.)
    agent = Agent(cfg, 5)
    sample = synthetic_batch(cfg)
    sample["next", "longitudinal_target"] = torch.full((*sample.batch_size, 1), .8)
    sample["next", "longitudinal_target_valid"] = torch.ones((*sample.batch_size, 1), dtype=torch.bool)
    result = agent.loss(sample)
    assert torch.isfinite(result["actor_longitudinal_bc_loss"])
    assert result["actor_longitudinal_bc_loss"].abs() > 1e-6
    assert agent.loss.actor_longitudinal_bc_weight == 4.


@pytest.mark.parametrize("count", [0, 3, 8])
def test_objective_fork_preserves_legacy_replay_and_writer_cursor(tmp_path, count):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8)
    trainer = Trainer(cfg, 5)
    data = synthetic_batch(cfg).reshape(-1)
    data["episode"] = torch.arange(8) // 4
    for row in data[:count]:
        trainer.replay.add(row)
    trainer.frames, trainer.updates, trainer.completed_episodes = count, 17, 2
    path = tmp_path / "legacy.pt"
    trainer.checkpoint(path)
    changed = replace(cfg, actor_longitudinal_bc_weight=4., expert_episode_interval=0)
    fork = Trainer.restore(path, changed)
    assert (fork.frames, fork.updates, fork.completed_episodes) == (count, 17, 2)
    assert fork.supervision_fork == str(path.resolve())
    assert len(fork.replay) == count
    if count:
        torch.testing.assert_close(fork.replay[:]["action"], trainer.replay[:]["action"])
        assert not fork.replay[:]["next", "longitudinal_target_valid"].any()
    row = data[0].clone()
    row["episode"] = torch.tensor(99)
    row["next", "longitudinal_target"] = torch.tensor([.7])
    row["next", "longitudinal_target_valid"] = torch.tensor([True])
    fork.replay.add(row)
    index = count % cfg.capacity
    assert fork.replay[index]["episode"] == 99
    assert fork.replay[index]["next", "longitudinal_target_valid"].item()
    fork.checkpoint(tmp_path / "fork.pt")
    restored = Trainer.restore(tmp_path / "fork.pt")
    assert restored.replay[index]["next", "longitudinal_target"].item() == pytest.approx(.7)
    for old, new in zip(trainer.agent.actor.parameters(), restored.agent.actor.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_refresh_interval_five_covers_current_scenario_cycle():
    cfg = Config.load("runs/experiment_25/config.json")
    assert cfg.expert_episode_interval == 5
    count = len(cfg.scenarios)
    assert count > 0
    selected = {cfg.scenarios[episode % count] for episode in range(count * 5) if episode % 5 == 0}
    assert selected == set(cfg.scenarios)


@pytest.mark.parametrize("count", [11, 12])
def test_refresh_interval_five_covers_original_and_reduced_curricula(count):
    selected = {episode % count for episode in range(count * 5) if episode % 5 == 0}
    assert selected == set(range(count))


def test_fake_collection_stores_next_state_labels_without_training(tmp_path):
    from dreamerv3_attention.synthetic import SyntheticEnv

    cfg = small_config(steps=8, max_episode_steps=4, max_objects=0, actor_longitudinal_bc_weight=4.)
    env = SyntheticEnv(cfg)
    step = env.step

    def labelled_step(action):
        observation, reward, terminated, truncated, info = step(action)
        info["longitudinal_target"] = .6
        info["longitudinal_target_valid"] = not (terminated or truncated)
        return observation, reward, terminated, truncated, info

    env.step = labelled_step
    trainer = Trainer(cfg, 36)
    trainer.frames = cfg.warmup_frames
    before = [p.clone() for p in trainer.agent.loss.parameters()]
    trainer.run(env, tmp_path / "collection", collect_only=True)
    assert trainer.updates == 0 and len(trainer.replay) == 4
    replay = trainer.replay[:]
    torch.testing.assert_close(replay["next", "longitudinal_target"], torch.full((4, 1), .6))
    assert replay["next", "longitudinal_target_valid"].squeeze(-1).tolist() == [True, True, True, False]
    rows = [json.loads(line) for line in (tmp_path / "collection" / "metrics.jsonl").read_text().splitlines()]
    assert all(row["action_source"] == "actor" for row in rows)
    torch.testing.assert_close(replay["action"], torch.tensor([row["applied_action"] for row in rows]))
    for old, new in zip(before, trainer.agent.loss.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_real_actor_longitudinal_supervision_detaches_world_model_features():
    agent = Agent(small_config(), 5)
    state = torch.randn(2, 3, agent.state_dim, requires_grad=True)
    belief = torch.randn(2, 3, agent.cfg.belief, requires_grad=True)
    posterior = TensorDict({"state": state, "belief": belief}, [2, 3])
    sample = TensorDict({"next": {"longitudinal_target": torch.full((2, 3, 1), .7),
                                  "longitudinal_target_valid": torch.ones(2, 3, 1, dtype=torch.bool),
                                  "done": torch.zeros(2, 3, 1, dtype=torch.bool)}}, [2, 3])
    corrective_longitudinal_loss(agent.actor, posterior, sample).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert state.grad is None and belief.grad is None
    assert all(p.grad is None for p in agent.prior.parameters())


def test_pilot_config_keeps_rewards_and_enables_motion_objective():
    from dreamerv3_attention.evaluation import collection_scenarios

    baseline = Config.load("runs/experiment_25/config.json").resolved()
    pilot = Config.load("runs/experiment_25/motion_pilot_config.json").resolved()
    assert pilot.reward == baseline.reward
    assert pilot.driving["max_speed_mps"] == 2.
    assert pilot.actor_longitudinal_bc_weight == 4.
    assert pilot.expert_episode_interval == 5
    assert pilot.world_model_warmup_updates == 200
    assert not pilot.randomize_routes
    scenario, = collection_scenarios(pilot)
    assert scenario["category"] == "straight" and scenario["weather"] == "ClearNoon"
    assert scenario["route_length_m"] == 40. and not scenario["background_traffic"]


def test_yellow_reference_preserves_existing_proceed_decision_when_too_close_to_stop():
    supervisor = SafetySupervisor()
    state = DrivingState(speed_limit_mps=8., yellow_light=True,
                         yellow_light_distance_m=1., yellow_light_confidence=1.)
    assert supervisor.longitudinal_reference(state, .05) > 0.
    assert supervisor._yellow_stop is None

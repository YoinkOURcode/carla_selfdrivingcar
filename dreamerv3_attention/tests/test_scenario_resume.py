"""Reward-only resume compatibility preserves learned training state."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import random

import numpy as np
import pytest
import torch

from dreamerv3_attention import __main__ as cli
from dreamerv3_attention.config import Config
from dreamerv3_attention.driving import RewardConfig
from dreamerv3_attention.tests.test_model import small_config, synthetic_batch
from dreamerv3_attention.training import Trainer


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8, scenarios=["train_front_hazard", "train_moving_traffic"])
    trainer = Trainer(cfg, 5)
    rows = synthetic_batch(cfg).reshape(-1)
    rows["episode"] = torch.arange(8) // 4
    trainer.replay.extend(rows)
    trainer.frames, trainer.updates, trainer.episode = 1234, 250, 9
    trainer.completed_episodes, trainer.debt = 8, .75
    trainer.training_mode = "dreamer"
    path = tmp_path / "original.pt"
    trainer.checkpoint(path)
    return cfg, trainer, path


@pytest.mark.parametrize("selectors", [
    {"scenarios": ["train_moving_traffic"]},
    {"scenarios": ["train_moving_traffic", "train_front_hazard"]},
    {"scenarios": ["train_moving_traffic", "train_low_light"]},
    {"scenarios": [], "scenario": "train_moving_traffic"},
])
def test_scenario_changes_keep_replay_and_progress_without_flags(checkpoint, selectors):
    cfg, original, path = checkpoint
    changed = replace(cfg, **selectors)
    fork = Trainer.restore(path, changed)
    assert fork.cfg.scenario == changed.scenario and fork.cfg.scenarios == changed.scenarios
    assert fork.frames == original.frames and fork.updates == original.updates
    assert fork.completed_episodes == original.completed_episodes
    assert fork.debt == original.debt
    assert fork.episode == original.episode + 1
    assert fork.scenario_fork == str(path.resolve())
    assert len(fork.replay) == len(original.replay) == 8
    torch.testing.assert_close(fork.replay[:], original.replay[:], rtol=0, atol=0)
    torch.testing.assert_close(fork.agent.loss.state_dict(), original.agent.loss.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(fork.agent.stepper.state_dict(), original.agent.stepper.state_dict(), rtol=0, atol=0)
    saved = torch.load(path, weights_only=False)
    assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
    assert random.getstate() == saved["python_rng"]
    actual = np.random.get_state()
    assert actual[0] == saved["numpy_rng"][0]
    np.testing.assert_array_equal(actual[1], saved["numpy_rng"][1])
    assert actual[2:] == saved["numpy_rng"][2:]


@pytest.mark.parametrize("overrides", [
    {"driving": {"max_speed_mps": 2., "default_speed_mps": 6., "following_min_gap_m": 6.}},
    {"driving": {"pedestrian_speed_factor": .5}},
    {"driving": {"pedestrian_speed_factor": .5, "child_pedestrian_speed_factor": .15}},
    {"split": "validation"}, {"town": "Town02"}, {"scenario_file": "another_catalog.json"},
    {"seed": 42}, {"dt": .025}, {"gamma": .99}, {"learning_rate": .001},
    {"actor_steering_bc_weight": 4.}, {"actor_longitudinal_bc_weight": 4.},
    {"warmup_policy": "controller", "actor_bc_weight": 4.},
])
def test_nonreward_config_changes_are_allowed(checkpoint, overrides):
    cfg, _, path = checkpoint
    changed = replace(cfg, **overrides)
    fork = Trainer.restore(path, changed)
    assert asdict(fork.cfg) == asdict(changed.resolved())
    assert fork.frames == 1234 and len(fork.replay) == 8


@pytest.mark.parametrize("field", list(asdict(RewardConfig())))
def test_every_reward_setting_still_requires_an_exact_match(checkpoint, field):
    cfg, _, path = checkpoint
    rewards = asdict(RewardConfig())
    rewards[field] += .1
    changed = replace(cfg, scenarios=["train_moving_traffic"], driving={"max_speed_mps": 2.}, reward=rewards)
    with pytest.raises(ValueError, match="incompatible config field reward"):
        Trainer.restore(path, changed)


def test_schema_and_dependency_versions_do_not_gate_restore(checkpoint):
    cfg, _, path = checkpoint
    saved = torch.load(path, weights_only=False)
    saved["schema"] = "different-schema"
    saved["versions"] = {"torch": "different-version"}
    torch.save(saved, path)
    assert Trainer.restore(path, cfg).frames == 1234
    del saved["schema"], saved["versions"]
    torch.save(saved, path)
    assert Trainer.restore(path, cfg).frames == 1234


def test_matching_partial_reward_config_resolves_defaults(checkpoint):
    cfg, _, path = checkpoint
    fork = Trainer.restore(path, replace(cfg, reward={"progress": RewardConfig().progress}))
    assert fork.cfg.reward == asdict(RewardConfig())


@pytest.mark.parametrize("overrides", [{"gamma": 2.}, {"driving": {"max_speed_mps": -1.}}])
def test_basic_config_validation_remains(checkpoint, overrides):
    cfg, _, path = checkpoint
    with pytest.raises(ValueError):
        Trainer.restore(path, replace(cfg, **overrides))


def test_incompatible_model_still_fails_strict_weight_loading(checkpoint):
    cfg, _, path = checkpoint
    with pytest.raises(ValueError, match="Failed to update") as error:
        Trainer.restore(path, replace(cfg, hidden=32))
    cause = error.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, RuntimeError)
    assert "size of tensor" in str(cause)


def test_combined_changes_and_provenance_roundtrip(checkpoint, tmp_path):
    cfg, _, path = checkpoint
    changed = replace(cfg, scenarios=["train_moving_traffic"], driving={"max_speed_mps": 2.},
                      actor_longitudinal_bc_weight=4.)
    fork = Trainer.restore(path, changed)
    assert fork.scenario_fork == fork.supervision_fork == fork.speed_cap_fork == str(path.resolve())
    assert not fork.replay[:]["next", "longitudinal_target_valid"].any()
    destination = tmp_path / "fork.pt"
    fork.checkpoint(destination)
    resumed = Trainer.restore(destination)
    assert resumed.scenario_fork == resumed.supervision_fork == resumed.speed_cap_fork == str(path.resolve())
    assert resumed.cfg == fork.cfg
    original = Trainer.restore(path)
    assert original.scenario_fork is original.supervision_fork is original.speed_cap_fork is None
    assert original.cfg.scenarios == cfg.scenarios
    assert (original.frames, original.updates) == (resumed.frames, resumed.updates)


@pytest.mark.parametrize("cap", [0., 6.])
def test_speed_cap_change_preserves_state_without_flags(checkpoint, cap):
    cfg, original, path = checkpoint
    original.cfg = replace(original.cfg, driving={**original.cfg.driving, "max_speed_mps": 2.})
    original.checkpoint(path)
    fork = Trainer.restore(path, replace(cfg, driving={"max_speed_mps": cap}))
    assert fork.cfg.driving["max_speed_mps"] == cap
    assert fork.speed_cap_fork == str(path.resolve())
    assert (fork.frames, fork.updates, fork.completed_episodes, fork.debt) == (1234, 250, 8, .75)
    torch.testing.assert_close(fork.replay[:], original.replay[:], rtol=0, atol=0)
    torch.testing.assert_close(fork.agent.loss.state_dict(), original.agent.loss.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(fork.agent.stepper.state_dict(), original.agent.stepper.state_dict(), rtol=0, atol=0)


def test_retired_signal_timing_config_loads_and_does_not_change_checkpoint(checkpoint, tmp_path):
    cfg, original, path = checkpoint
    config_path = tmp_path / "legacy.json"
    config_path.write_text(json.dumps({"traffic_light_red_seconds": 5.,
                                      "traffic_light_yellow_seconds": 1.}))
    loaded = Config.load(config_path)
    assert not {"traffic_light_red_seconds", "traffic_light_yellow_seconds"} & asdict(loaded).keys()
    saved_config = tmp_path / "saved.json"
    loaded.save(saved_config)
    assert not {"traffic_light_red_seconds", "traffic_light_yellow_seconds"} & json.loads(
        saved_config.read_text()).keys()

    saved = torch.load(path, weights_only=False)
    saved["config"]["traffic_light_red_seconds"] = 5.
    saved["config"]["traffic_light_yellow_seconds"] = 1.
    torch.save(saved, path)
    restored = Trainer.restore(path)

    assert restored.cfg == original.cfg
    assert (restored.frames, restored.updates, restored.episode,
            restored.completed_episodes, restored.debt) == (
                original.frames, original.updates, original.episode + 1,
                original.completed_episodes, original.debt)
    assert len(restored.replay) == len(original.replay)
    torch.testing.assert_close(restored.replay[:], original.replay[:], rtol=0, atol=0)
    torch.testing.assert_close(restored.agent.loss.state_dict(), original.agent.loss.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("flag", [
    "--allow-scenario-change", "--allow-actor-supervision-change", "--allow-speed-cap-change",
])
def test_custom_resume_flags_are_removed(flag):
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args(["train", flag])
    assert error.value.code == 2


def test_cli_resumes_without_custom_flags_or_opening_environment(monkeypatch, tmp_path):
    cfg = small_config(scenarios=["train_moving_traffic"], driving={"max_speed_mps": 0.})
    config = tmp_path / "config.json"
    cfg.save(config)
    monkeypatch.setattr(cli, "workload_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *a: None)
    monkeypatch.setattr(cli, "CarlaEnv", lambda *a, **kw: pytest.fail("must not open CARLA"))

    def restore(path, actual_cfg):
        assert path == "source.pt"
        assert actual_cfg.scenarios == cfg.scenarios
        assert actual_cfg.driving["max_speed_mps"] == 0.
        raise RuntimeError("restore reached without training")

    monkeypatch.setattr(cli.Trainer, "restore", restore)
    with pytest.raises(RuntimeError, match="restore reached without training"):
        cli.main(["train", "--config", str(config), "--output", str(tmp_path / "unused"),
                  "--checkpoint", "source.pt", "--budget-seconds", "10"])

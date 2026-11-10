from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from dreamerv3_attention.synthetic import SyntheticEnv
from dreamerv3_attention.inactivity import InactivityWatchdog
from dreamerv3_attention.training import Trainer, seed_all
from dreamerv3_attention.tests.test_model import small_config, synthetic_batch


def test_collect_replay_optimize_and_resume(tmp_path):
    torch.set_num_threads(1)
    seed_all(2)
    cfg = small_config(steps=12, max_episode_steps=6, max_objects=0, updates_per_step=0.25)
    trainer = Trainer(cfg, 36)
    result = trainer.run(SyntheticEnv(cfg), tmp_path / "first")
    assert result["frames"] == 12 and result["updates"] > 0
    assert len(trainer.replay) == 12
    restored = Trainer.restore(tmp_path / "first" / "latest.pt", replace(cfg, steps=16))
    assert restored.frames == 12 and restored.updates == trainer.updates
    assert restored.debt == trainer.debt
    for p, q in zip(trainer.agent.loss.parameters(), restored.agent.loss.parameters()):
        torch.testing.assert_close(p, q)
    batch = restored.replay.sample().reshape(cfg.batch, cfg.sequence)
    # Sequences are contiguous and never cross episode boundaries.
    assert (batch["episode"] == batch["episode"][:, :1]).all()
    assert not batch["next", "terminated"].any()
    assert batch["next", "done"].dtype == torch.bool
    restored.run(SyntheticEnv(cfg), tmp_path / "continued")
    assert restored.frames == 16
    data = [json.loads(line) for line in (tmp_path / "continued" / "metrics.jsonl").read_text().splitlines()]
    assert data[0]["episode"] > 1
    assert data[-1]["updates"] > trainer.updates


def test_checkpoint_rng_and_optimizer_exact_continuation(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config()
    seed_all(1)
    trainer = Trainer(cfg, 5)
    trainer.agent.update(synthetic_batch(cfg))
    trainer.checkpoint(tmp_path / "test.pt")
    batch = synthetic_batch(cfg)
    rng = torch.get_rng_state()
    trainer.agent.update(batch.clone())
    expected = [p.detach().clone() for p in trainer.agent.loss.parameters()]
    restored = Trainer.restore(tmp_path / "test.pt")
    torch.set_rng_state(rng)
    restored.agent.update(batch.clone())
    for actual, wanted in zip(restored.agent.loss.parameters(), expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="state_dict"):
        Trainer.restore(tmp_path / "test.pt", replace(cfg, attention=False))


def test_configuration_rejects_invalid_observation_and_training_settings():
    for overrides in ({"dt": 0.2}, {"capacity": 2}, {"perception": "external"},
                      {"perception": "mixed", "provider": "a:b"}, {"image_size": 15}):
        with pytest.raises(ValueError):
            replace(small_config(), **overrides).validate()


def test_collected_replay_can_resume_world_model_only_and_checkpoint_mode(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(steps=12, max_episode_steps=6, max_objects=0)
    trainer = Trainer(cfg, 36)
    trainer.run(SyntheticEnv(cfg), tmp_path / "collection", collect_only=True)
    assert trainer.updates == 0
    trained = Trainer.restore(tmp_path / "collection" / "latest.pt", replace(cfg, steps=16))
    before_actor = {key: value.clone() for key, value in trained.agent.actor.state_dict().items()}
    result = trained.run(SyntheticEnv(cfg), tmp_path / "model", world_model_only=True)
    assert result["updates"] > 0 and result["training_mode"] == "world_model"
    for key, value in trained.agent.actor.state_dict().items():
        torch.testing.assert_close(value, before_actor[key], rtol=0, atol=0)
    restored = Trainer.restore(tmp_path / "model" / "latest.pt")
    assert restored.training_mode == "world_model"
    assert restored.updates == trained.updates


def test_watchdog_truncation_preserves_final_observation_and_resets_carry(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(steps=6, max_episode_steps=6, max_objects=0, updates_per_step=0.)
    trainer = Trainer(cfg, 36)
    env = SyntheticEnv(cfg)
    watchdog = InactivityWatchdog(timeout_s=.1, min_progress_m=.5,
                                  blocked_timeout_s=.3, grace_s=0.)
    original_reset, original_step = env.reset, env.step
    resets, carries = [], []

    def reset():
        resets.append(True)
        watchdog.reset(0.)
        return original_reset()

    def step(action):
        observation, reward, terminated, truncated, info = original_step(action)
        reason = watchdog.update(0., cfg.dt, can_move=True)
        observation["tokens"][0, 0] = env.steps
        info["truncation_reason"] = reason
        return observation, reward, terminated, truncated or reason is not None, info

    original_act = trainer.agent.act

    def act(observation, carry):
        carries.append((carry["is_init"].item(), carry["previous_action"].clone()))
        return original_act(observation, carry)

    env.reset, env.step, trainer.agent.act = reset, step, act
    trainer.run(env, tmp_path / "watchdog", collect_only=True)
    replay = trainer.replay[:]
    assert len(resets) == 3
    assert trainer.completed_episodes == 3
    assert [initial for initial, _ in carries] == [True, False, True, False, True, False]
    for initial, action in carries:
        if initial:
            assert not action.any()
    assert replay["next", "truncated"].squeeze(-1).tolist() == [False, True] * 3
    assert not replay["next", "terminated"].any()
    assert replay["episode"].tolist() == [0, 0, 1, 1, 2, 2]
    assert replay["next", "tokens"][:, 0, 0].tolist() == [1., 2.] * 3


def test_simulator_crash_saves_progress_and_resume_keeps_completed_episodes(tmp_path, capsys):
    torch.set_num_threads(1)
    cfg = small_config(steps=12, max_episode_steps=6, max_objects=0, checkpoint_every=500)
    trainer = Trainer(cfg, 36)
    env = SyntheticEnv(cfg)
    original_step = env.step

    def step(action):
        if trainer.frames == 7:
            raise RuntimeError("trying to operate on a destroyed actor")
        return original_step(action)

    env.step = step
    crashed = tmp_path / "crashed"
    with pytest.raises(RuntimeError, match="destroyed actor"):
        trainer.run(env, crashed)
    assert "checkpoint saved:" in capsys.readouterr().out
    checkpoint = crashed / "latest.pt"
    restored = Trainer.restore(checkpoint)
    assert restored.frames == 7 and restored.updates == trainer.updates > 0
    assert restored.completed_episodes == trainer.completed_episodes == 1
    assert len(restored.replay) == len(trainer.replay) == 7
    assert restored.episode > trainer.episode
    for p, q in zip(trainer.agent.loss.parameters(), restored.agent.loss.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    restored.run(SyntheticEnv(cfg), tmp_path / "continued")
    assert restored.frames == 12 and len(restored.replay) == 12
    # A new output must never overwrite the checkpoint it was resumed from.
    assert Trainer.restore(checkpoint).frames == 7

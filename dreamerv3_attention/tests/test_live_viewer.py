import json
import time
from types import SimpleNamespace as NS

import pytest
import torch

from dreamerv3_attention.live import LiveTraining, TrainingMonitor
from dreamerv3_attention.evaluation import evaluate
from dreamerv3_attention.tests.test_model import small_config
from dreamerv3_attention.tests.test_viewer import FakeEnv, observation
from dreamerv3_attention.training import BudgetEnv, Trainer


def test_monitor_idle_discovery_explicit_run_and_publisher_lifetime(tmp_path):
    monitor = TrainingMonitor(root=tmp_path)
    assert monitor.read()["status"] == "idle"
    cfg = small_config(steps=4)
    old = tmp_path / "old"
    old.mkdir()
    active = tmp_path / "active"
    active.mkdir()
    with LiveTraining(active, cfg, "dreamer") as publisher:
        # Even a newer ended run must not displace the active publisher.
        with LiveTraining(old, cfg, "dreamer") as completed:
            completed.finish("steps")
        assert monitor.read()["run"] == str(active)
        assert monitor.read()["active"]
        assert TrainingMonitor(old).read()["stop_reason"] == "steps"
        state = json.loads((active / "live.json").read_text())
        state["updated_at"] = time.time() - 100
        (active / "live.json").write_text(json.dumps(state))
        assert TrainingMonitor(active).read()["active"]
        assert TrainingMonitor(active).read()["age_s"] >= 100
    # Closing without a final record models a lost publisher, with stale
    # metadata left behind. It must never look like active training.
    assert TrainingMonitor(active).read()["status"] == "ended"
    assert not TrainingMonitor(active).read()["active"]
    assert publisher.lock.closed


def test_live_training_errors_and_corrupt_snapshot_do_not_look_active(tmp_path):
    with pytest.raises(RuntimeError, match="sensor failure"):
        with LiveTraining(tmp_path, small_config(), "dreamer"):
            raise RuntimeError("sensor failure")
    state = TrainingMonitor(tmp_path).read()
    assert state["status"] == "error" and not state["active"]
    assert "sensor failure" in state["message"]
    traceback = (tmp_path / "error.txt").read_text()
    assert "Traceback (most recent call last)" in traceback
    assert "test_live_training_errors_and_corrupt_snapshot_do_not_look_active" in traceback
    assert "RuntimeError: sensor failure" in traceback
    (tmp_path / "live.json").write_text("partial JSON")
    assert TrainingMonitor(tmp_path).read()["status"] == "idle"


def test_live_training_keeps_latest_learner_mode_and_episode_rewards_across_throttle(tmp_path):
    cfg = small_config(steps=20, max_objects=5)
    env = FakeEnv()
    env.cfg = cfg
    with LiveTraining(tmp_path, cfg, "dreamer") as publisher:
        publisher.last_publish = time.monotonic() + 10
        publisher.publish(env, observation(), {"frames": 1, "updates": 1, "learner_mode": "world_model",
                          "reward": 2., "reward_components": {"progress": 2.}, "new_progress_m": 1.})
        publisher.publish(env, observation(), {"frames": 2, "updates": 2, "reward": -1.,
                          "reward_components": {"red_light_violation": -1.}})
        publisher.last_publish = 0
        publisher.publish(env, observation(), {"frames": 3, "updates": 2, "reward": 3.,
                          "reward_components": {"progress": 3.}, "new_progress_m": 1.})
        state = TrainingMonitor(tmp_path).read()
        assert state["metrics"]["learner_mode"] == "world_model"
        assert state["episode_return"] == 4.
        assert sum(row["reward"] for row in state["reward_breakdown"]) == state["episode_return"]
        assert {row["name"]: row["reward"] for row in state["reward_breakdown"]} == {
            "Progress": 5., "Red light": -1.}

        publisher.publish(env, observation(), {"frames": 4, "updates": 2, "episode": 1}, reset=True)
        state = TrainingMonitor(tmp_path).read()
        assert state["metrics"]["learner_mode"] == "world_model"
        assert state["reward_breakdown"] == [] and state["episode_return"] is None

        publisher.last_publish = time.monotonic() + 10
        publisher.publish(env, observation(), {"frames": 5, "updates": 3, "learner_mode": "dreamer",
                          "reward": 1., "reward_components": {"success": 1.}, "success": True})
        publisher.last_publish = 0
        publisher.publish(env, observation(), {"frames": 6, "updates": 3, "reward": 0.,
                          "reward_components": {"success": 0.}})
        state = TrainingMonitor(tmp_path).read()
        assert state["metrics"]["learner_mode"] == "dreamer"
        assert state["episode_return"] == 1.
        assert state["reward_breakdown"][0]["name"] == "Success"

        other_path = tmp_path / "other"
        other_path.mkdir()
        with LiveTraining(other_path, cfg, "dreamer") as other:
            other.publish(env, observation(), {"frames": 1, "updates": 0, "reward": 0.,
                           "reward_components": {"progress": 0.}})
            other_state = TrainingMonitor(other_path).read()
            assert "learner_mode" not in other_state["metrics"]


def test_trainer_publishes_real_episode_snapshots_through_budget_wrapper(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(steps=4, max_objects=5)
    trainer = Trainer(cfg, len(observation()["mask"]))
    env = FakeEnv()
    env.cfg = cfg
    original_reset, original_step = env.reset, env.step

    def reset():
        env.steps = 0
        env.position.x = 0
        env.state.route_progress_m = 0
        return original_reset()

    def step(action):
        obs, _, done, truncated, info = original_step(action)
        return obs, 1.5, done, truncated, info

    env.reset, env.step = reset, step
    env.scenario = {"route_start": 5, "route_end": 8}
    observed = []
    with LiveTraining(tmp_path, cfg, "collection") as publisher:
        publish = publisher.publish

        def observe(*args, **kwargs):
            publish(*args, **kwargs)
            observed.append(TrainingMonitor(tmp_path).read())

        publisher.publish = observe
        result = trainer.run(BudgetEnv(env, 20), tmp_path, collect_only=True, observer=publisher)
        publisher.finish(result["stop_reason"])
    assert result["frames"] == 4 and result["updates"] == 0
    assert all(state["active"] for state in observed)
    terminals = [s for s in observed if s["metrics"].get("terminated")]
    assert len(terminals) == 2
    assert [s["metrics"]["episode"] for s in terminals] == [0, 1]
    assert all(s["distance_m"] == 6 and s["episode_return"] == 3 for s in terminals)
    assert all(s["image"].startswith("data:image/jpeg;base64,") for s in terminals)
    assert all(s["controls"]["throttle"] == .3 for s in terminals)
    assert all(s["route_start"] == 5 and s["route_end"] == 8 for s in terminals)
    final = TrainingMonitor(tmp_path).read()
    assert final["metrics"]["frames"] == 4 and final["stop_reason"] == "steps"
    assert final["status"] == "ended" and not final["active"]
    assert not final["attention_enabled"]


def test_evaluation_publishes_scenario_resets_final_frames_and_completion(tmp_path):
    cfg = small_config(split="validation", max_objects=5)
    environments, observed = [], []

    class EvaluationEnv(FakeEnv):
        def __init__(self, run_cfg, scenario):
            super().__init__()
            self.cfg, self.scenario = run_cfg, scenario
            self.closed = False
            environments.append(self)

        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            info.update(route_length_m=6., route_progress_m=self.state.route_progress_m)
            return obs, 1.5, terminated, truncated, info

        def close(self):
            self.closed = True

    agent = NS(initial=lambda: None, act=lambda obs, carry, deterministic: ([0., .2], carry))

    with LiveTraining(tmp_path, cfg, "evaluation") as publisher:
        publish = publisher.publish

        def observe(*args, **kwargs):
            publish(*args, **kwargs)
            observed.append(TrainingMonitor(tmp_path).read())

        publisher.publish = observe

        def factory(run_cfg, scenario):
            env = EvaluationEnv(run_cfg, scenario)
            wrapped = BudgetEnv(env, 20)
            wrapped.close = env.close
            return wrapped

        result = evaluate(cfg, agent, factory, episodes=2, observer=publisher)
        publisher.finish("completed")

    assert result["summary"]["episode_count"] == 2
    assert all(env.closed for env in environments)
    assert all(state["active"] and state["mode"] == "evaluation" for state in observed)
    finals = [state for state in observed if state["metrics"].get("terminated")]
    assert len(finals) == 2
    assert [state["metrics"]["frames"] for state in finals] == [2, 4]
    assert [state["metrics"]["episode"] for state in finals] == [0, 1]
    assert all(state["metrics"]["total_episodes"] == 2 for state in finals)
    assert all(state["distance_m"] == 6. and state["episode_return"] == 3. for state in finals)
    assert all(state["image"].startswith("data:image/jpeg;base64,") for state in finals)
    assert [state["route_start"] for state in finals] == [env.scenario["route_start"] for env in environments]
    final = TrainingMonitor(tmp_path).read()
    assert final["status"] == "ended" and not final["active"]
    assert final["stop_reason"] == "completed" and final["mode"] == "evaluation"


def test_monitor_discovers_nested_evaluation_runs_and_marks_lost_publisher_ended(tmp_path):
    output = tmp_path / "experiment" / "evaluation"
    output.mkdir(parents=True)
    with LiveTraining(output, small_config(), "evaluation") as publisher:
        publisher.state["status"] = "evaluating"
        publisher.write()
        assert TrainingMonitor(root=tmp_path).read()["active"]
    state = TrainingMonitor(root=tmp_path).read()
    assert state["status"] == "ended" and not state["active"]


def test_training_snapshot_publishes_object_and_radar_policy_tokens(tmp_path):
    import numpy as np
    from dreamerv3_attention.observations import pack_observation

    env = FakeEnv()
    cfg = env.cfg
    rows = np.zeros((5, 16), np.float32)
    rows[:3, 15] = -1.
    rows[3, [0, 8, 10]] = [-.1, 1., 1.]
    obs = pack_observation(np.zeros((24, 32, 3), np.uint8), np.zeros(16, np.float32), rows[:4],
                           {"front": np.array([[3., 4., 0., 0.]], np.float32)},
                           max_objects=5, model_size=(16, 16))
    with LiveTraining(tmp_path, cfg, "dreamer") as publisher:
        publisher.publish(env, obs, {"frames": 1}, reset=True)
        state = TrainingMonitor(tmp_path).read()
    assert state["policy_objects"][0]["camera"] == "outside frame"
    assert state["radar_readings"][0]["distance_m"] == pytest.approx(5.)
    assert state["radar_readings"][0]["attention"] is None

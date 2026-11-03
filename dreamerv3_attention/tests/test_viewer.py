import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace as NS
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import carla
import numpy as np
import pytest
import torch

from dreamerv3_attention.config import Config
from dreamerv3_attention.observations import FrameBundle, pack_observation
from dreamerv3_attention.viewer import EpisodeRewards, Viewer, camera_objects, make_server, snapshot


def arguments(**kwargs):
    return NS(**({"budget_seconds": 10., "checkpoint": "model.pt",
                  "ue4_root": None, "output": None} | kwargs))


def observation():
    rows = np.zeros((5, 16), np.float32)
    rows[:3, 15] = -1  # navigation tokens
    rows[3, [0, 6, 8, 10]] = [.10, 1, 1, .9]
    rows[4, [0, 8, 10]] = [-.10, 1, 1]  # behind camera
    return pack_observation(np.zeros((24, 32, 3), np.uint8), np.zeros(16, np.float32), rows,
                            max_objects=5, model_size=(16, 16))


def test_projects_only_objects_in_camera_and_maps_attention_after_image_token():
    obs = observation()
    weights = np.arange(len(obs["mask"]) + 2) / 100.
    intrinsics = {"cx": 16, "cy": 12, "fx": 16, "fy": 16}
    result = camera_objects(obs, weights, np.eye(4), intrinsics, (32, 24), 2, "ground_truth")
    assert len(result) == 1
    assert result[0]["label"] == "pedestrian"
    assert result[0]["slot"] == 0 and result[0]["attention"] == .05
    assert result[0]["x"] == result[0]["y"] == .5
    assert result[0]["distance_m"] == 10.
    obs["tokens"][4, 1] = 2  # outside image
    assert camera_objects(obs, None, np.eye(4), intrinsics, (32, 24), 2, "external") == []
    obs["tokens"][4, 1] = 0
    obs["tokens"][4, 6] = 0
    result = camera_objects(obs, None, np.eye(4), intrinsics, (32, 24), 2, "external")
    assert result[0]["attention"] is None and result[0]["label"] == "object"
    obs["mask"][4] = False
    assert camera_objects(obs, weights, np.eye(4), intrinsics, (32, 24), 2, "ground_truth") == []


class FakeEnv:
    def __init__(self):
        self.cfg = Config(image_size=16, max_objects=2)
        self.steps = 0
        self.position = carla.Location()
        self.ego = NS(get_location=lambda: self.position,
                      get_transform=lambda: carla.Transform(self.position),
                      get_control=lambda: NS(steer=.1, throttle=.3, brake=0.))
        camera = NS(get_transform=lambda: carla.Transform(self.position))
        self.rig = NS(sensors=[camera])
        self.route = NS(length=6.)
        self.state = NS(speed_mps=2., speed_limit_mps=10., route_progress_m=0.)
        self.last_bundle = FrameBundle(1, .05, rgb=np.zeros((24, 32, 3), np.uint8), valid={"rgb": True},
                                      calibration={"sensors": {"rgb": {"resolution_px": (32, 24),
                                                   "intrinsics_px": {"cx": 16, "cy": 12, "fx": 16, "fy": 16}}}})

    def reset(self):
        return observation()

    def step(self, action):
        self.steps += 1
        self.position = carla.Location(x=3*self.steps)
        self.state.route_progress_m = 3*self.steps
        self.last_bundle.frame += 1
        return observation(), 0., self.steps == 2, False, {"success": self.steps == 2, "applied_action": (.1, .2)}


def test_snapshot_uses_actual_controls_full_camera_and_measured_units():
    env = FakeEnv()
    state = snapshot(env, observation(), None, 3.25, {"interventions": ["pedestrian"]}, "dreamer")
    assert state["speed_kmh"] == 7.2 and state["speed_limit_kmh"] == 36.
    assert state["distance_m"] == 3.25 and state["controls"]["throttle"] == .3
    assert state["image"].startswith("data:image/jpeg;base64,")
    assert state["interventions"] == ["pedestrian"]
    assert state["objects"][0]["attention"] is None
    env.last_bundle.valid["rgb"] = False
    state = snapshot(env, observation(), None, 0., {}, "dreamer")
    assert state["image"] is None and state["objects"] == []


def test_episode_rewards_sums_actual_components_and_reports_recorded_values():
    rewards = EpisodeRewards()
    rewards.add({"reward": 52., "reward_components": {"waypoint": 50., "progress": 2.},
                 "waypoints_reached": 1, "new_progress_m": 2.}, dt_s=.1)
    rewards.add({"reward": -100., "reward_components": {"red_light_violation": -100.,
                 "future_component": 0.}}, dt_s=.1)
    state = rewards.snapshot()
    assert state["episode_return"] == -48.
    assert sum(row["reward"] for row in state["reward_breakdown"]) == state["episode_return"]
    assert {row["name"]: row["reward"] for row in state["reward_breakdown"]} == {
        "Waypoint (15 m)": 50., "Progress": 2., "Red light": -100., "Future Component": 0.}
    values = {row["name"]: row["value"] for row in state["reward_breakdown"]}
    assert values["Waypoint (15 m)"] == "1 checkpoints"
    assert values["Progress"] == "2.0 m new"
    assert values["Red light"] == "—"
    rewards.reset()
    assert rewards.snapshot() == {"reward_breakdown": [], "episode_return": None}


def test_episode_rewards_uses_only_measured_values_and_reward_speed_fallback():
    rewards = EpisodeRewards()
    rewards.add({"reward": -5., "reward_components": {"lane": -1., "target_tracking": -1.,
                 "speeding": -3., "lane_departure": -1.}, "lane_error_m": 2.,
                 "speed_mps": 4., "target_speed_mps": 3., "speed_limit_mps": None})
    rewards.add({"reward": 0., "reward_components": {"lane": 0., "target_tracking": 0.,
                 "speeding": 0., "lane_departure": 0.}})
    rows = {row["name"]: row for row in rewards.snapshot()["reward_breakdown"]}
    assert rows["Lane keeping"]["value"] == "mean 2.00 m offset"
    assert rows["Target tracking"]["value"] == "mean 1.00 m/s error"
    assert rows["Speeding"]["value"] == "max 1.00 m/s over limit"
    assert rows["Lane departure"]["value"] == "—"

    for limit, confidence in ((None, "missing"), (float("nan"), "missing"),
                              (10., None), (10., float("nan")), (10., .1), (10., 1.)):
        fallback = EpisodeRewards()
        record = {"reward": -1., "reward_components": {"speeding": -1.},
                  "speed_mps": 4., "speed_limit_mps": limit}
        if confidence != "missing":
            record["speed_limit_confidence"] = confidence
        fallback.add(record)
        expected = "max 1.00 m/s over limit" if confidence != 1. else "max 0.00 m/s over limit"
        assert fallback.snapshot()["reward_breakdown"][0]["value"] == expected


def test_checkpoint_episode_tracks_distance_success_attention_and_executed_carry():
    viewer = Viewer(Config(), arguments())
    carries = []

    def act(obs, carry, deterministic):
        carries.append(carry)
        return np.array([.1, .9]), {}

    weights = torch.arange(len(observation()["mask"]) + 2) / 100.
    agent = NS(initial=lambda: {}, act=act, encoder=NS(last_attention=weights), device="cpu")
    env = FakeEnv()
    env.route_start, env.route_end = 4, 12
    original_step = env.step

    def reward_step(action):
        obs, _, done, truncated, info = original_step(action)
        return obs, 2.0, done, truncated, {**info, "reward_components": {"progress": 1.5,
                                                                            "time": .5}}

    env.step = reward_step
    viewer._episode(env, agent, time.monotonic() + 10)
    state = viewer.read()
    assert state["status"] == "arrived" and state["success"]
    assert (state["route_start"], state["route_end"]) == (4, 12)
    assert state["distance_m"] == 6. and state["remaining_m"] == 0
    assert state["frame"] == 3 and state["objects"][0]["attention"] == pytest.approx(.05)
    assert carries[1]["previous_action"].tolist() == pytest.approx([.1, .2])
    assert not carries[1]["is_init"].item()
    assert viewer.read()["episode_return"] == 4.0
    assert {row["name"]: row["reward"] for row in viewer.read()["reward_breakdown"]} == {
        "Progress": 3.0, "Time": 1.0}


@pytest.mark.parametrize("stop", [True, False])
def test_stop_or_expired_budget_never_applies_an_action(stop):
    viewer = Viewer(Config(), arguments())
    if stop:
        viewer.stop()
    env = FakeEnv()
    agent = NS(initial=lambda: {})
    viewer._episode(env, agent, time.monotonic() + (10 if stop else -1))
    assert env.steps == 0 and viewer.read()["status"] == "stopped"


@pytest.mark.parametrize("ending,truncated,detail,message", [
    ("lane_deviation", False, {"lane_error_m": -6.2, "lane_departure": True}, "exceeded 6 m"),
    ("offroad", False, {"offroad": True}, "off-road"),
    ("reward_limit", False, {}, "below -500"),
    ("inactivity_timeout", True, {}, "inactivity timeout"),
    ("blocked_timeout", True, {}, "remained blocked"),
    ("time_limit", True, {}, "step limit"),
])
def test_episode_exposes_ending_reason_and_safety_fields(ending, truncated, detail, message):
    viewer = Viewer(Config(), arguments())
    env = FakeEnv()
    key = "truncation_reason" if truncated else "termination_reason"
    env.step = lambda _: (observation(), -1., not truncated, truncated,
                          {key: ending, "applied_action": (0., 0.), **detail})
    agent = NS(initial=lambda: {}, act=lambda *_a, **_kw: (np.zeros(2), {}),
               encoder=NS(last_attention=None), device="cpu")
    viewer._episode(env, agent, time.monotonic() + 10)
    state = viewer.read()
    assert state["status"] == "ended"
    assert state[key] == ending and message in state["message"]
    if ending == "lane_deviation":
        assert state["lane_error_m"] == -6.2
        assert state["safety_events"] == ["lane_departure", "lane_deviation"]
    elif ending == "offroad":
        assert state["safety_events"] == ["offroad"]


def test_demo_drive_preserves_selected_route_and_saves_terminal_diagnostics(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from dreamerv3_attention import __main__ as cli, environment, training

    cfg = Config.load("runs/experiment_26/curriculum/town02_demo.json")
    monkeypatch.setattr(training, "workload_lock", lambda *_a, **_kw: nullcontext())
    monkeypatch.setattr(training, "register_run", lambda *_a: None)
    monkeypatch.setattr(training, "seed_all", lambda *_a: None)
    monkeypatch.setattr(cli, "server", lambda *_a: nullcontext())
    monkeypatch.setattr(training.Trainer, "restore", lambda *_a, **_kw: NS(cfg=cfg, agent=object()))
    captured = []

    class Environment:
        def __init__(self, run_cfg, scenario):
            captured.append((run_cfg, scenario))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(environment, "CarlaEnv", Environment)
    output = tmp_path / "drive"
    viewer = Viewer(cfg, arguments(output=str(output)))
    monkeypatch.setattr(viewer, "_episode", lambda *_: viewer.update(
        status="ended", termination_reason="lane_deviation", lane_error_m=6.2,
        safety_events=["lane_deviation"]))
    viewer.drive(2, 56)
    run_cfg, scenario = captured[0]
    assert (run_cfg.route_start, run_cfg.route_end) == (2, 56)
    assert (scenario["route_start"], scenario["route_end"]) == (2, 56)
    assert scenario["fixed_route"] and scenario["requires_traffic_light"]
    assert scenario["name"] == "town02_demonstration"
    result = json.loads((output / "result.json").read_text())
    assert result["termination_reason"] == "lane_deviation"
    assert result["lane_error_m"] == 6.2 and result["safety_events"] == ["lane_deviation"]


def test_driving_requires_budget_checkpoint_and_valid_route():
    for args, request, error in [
        (arguments(budget_seconds=None), {}, "budget-seconds"),
        (arguments(budget_seconds=float("nan")), {}, "budget-seconds"),
        (arguments(budget_seconds=0), {}, "budget-seconds"),
        (arguments(budget_seconds=86401), {}, "budget-seconds"),
        (arguments(checkpoint=None), {}, "checkpoint"),
        (arguments(), {"start": True}, "spawn indices"),
        (arguments(), {"start": 1, "destination": 1}, "spawn indices"),
    ]:
        with pytest.raises(ValueError, match=error):
            Viewer(Config(), args).start(request)


def test_viewer_starts_with_budget_and_checkpoint(monkeypatch):
    viewer = Viewer(Config(), arguments())
    routes = []
    monkeypatch.setattr(viewer, "drive", lambda start, end: routes.append((start, end)))
    viewer.start({"start": 0, "destination": 10})
    viewer.thread.join(timeout=2)
    assert routes == [(0, 10)]


def test_existing_output_is_preserved_on_failed_start(monkeypatch, tmp_path):
    from dreamerv3_attention import training

    @contextmanager
    def lock(*args, **kwargs):
        yield

    monkeypatch.setattr(training, "workload_lock", lock)
    result = tmp_path / "result.json"
    result.write_text("original experiment")
    viewer = Viewer(Config(), arguments(output=str(tmp_path)))
    viewer.drive(0, 10)
    assert viewer.read()["status"] == "error"
    assert result.read_text() == "original experiment"


def test_http_dashboard_status_and_local_control_contract():
    viewer = Viewer(Config(), arguments(budget_seconds=None))
    with make_server(viewer, 0) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(url) as response:
                assert b"Live drive" in response.read()
                assert response.headers["Cache-Control"] == "no-store"
            with urlopen(url + "/api/state") as response:
                assert json.load(response)["status"] == "idle"
            with urlopen(url + "/api/training") as response:
                assert "active" in json.load(response)
            request = Request(url + "/api/start", data=b"{}", headers={"Content-Type": "application/json"})
            with pytest.raises(HTTPError) as exc:
                urlopen(request)
            assert exc.value.code == 400 and b"budget-seconds" in exc.value.read()
            request = Request(url + "/api/stop", data=b"{}", headers={"Content-Type": "application/json",
                                                                       "Origin": "http://untrusted.example"})
            with pytest.raises(HTTPError) as exc:
                urlopen(request)
            assert exc.value.code == 403 and not viewer.stop_event.is_set()
            request = Request(url + "/api/stop", data=b"{}", headers={"Content-Type": "application/json"})
            with urlopen(request) as response:
                assert json.load(response)["ok"]
            assert viewer.stop_event.is_set()
        finally:
            server.shutdown()
            worker.join()


def test_snapshot_lists_offscreen_objects_and_when_camera_unavailable():
    env = FakeEnv()
    obs = observation()
    state = snapshot(env, obs, None, 0., {}, "training")
    assert state["policy_object_count"] == 2
    assert [item["camera"] for item in state["policy_objects"]] == ["in frame", "outside frame"]
    env.last_bundle.valid["rgb"] = False
    state = snapshot(env, obs, None, 0., {}, "training")
    assert len(state["policy_objects"]) == 2 and not state["objects"]
    assert all(item["camera"] == "unavailable" for item in state["policy_objects"])


def test_snapshot_limits_object_list_and_reads_radar_without_object_labels():
    env = FakeEnv()
    env.cfg.max_objects = 25
    tokens = np.zeros((61, 16), np.float32)
    mask = np.zeros(61, dtype=bool)
    mask[4:29] = True
    tokens[4:29, 8] = 1.
    mask[29] = mask[60] = True
    tokens[29, :3] = [.03, .04, 0.]
    tokens[60, 0] = -.1
    weights = np.arange(63) / 100.
    obs = {"tokens": tokens, "mask": mask}
    state = snapshot(env, obs, weights, 0., {}, "training")
    assert len(state["policy_objects"]) == 20 and state["policy_object_count"] == 25
    assert len(state["radar_readings"]) == 2
    assert state["radar_readings"][0]["distance_m"] == pytest.approx(5.)
    assert state["radar_readings"][0]["attention"] == pytest.approx(.30)
    assert set(state["radar_readings"][0]) == {"distance_m", "attention"}


def test_snapshot_reports_only_available_traffic_light_state():
    env = FakeEnv()
    obs = observation()
    obs["tokens"][0, 12] = 1.
    obs["tokens"][2, 3:5] = [.12, 1.]
    assert snapshot(env, obs, None, 0., {}, "training")["traffic_signal"] == {
        "state": "red", "distance_m": pytest.approx(12.)}
    obs["tokens"][0, 12] = 0.
    obs["tokens"][2, 5:8] = [1., .08, .9]
    signal = snapshot(env, obs, None, 0., {}, "training")["traffic_signal"]
    assert signal["state"] == "yellow" and signal["distance_m"] == pytest.approx(8.)
    obs["tokens"][2, 5] = 0.
    assert snapshot(env, obs, None, 0., {}, "training")["traffic_signal"] is None

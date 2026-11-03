from contextlib import nullcontext
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import carla
import pytest

from dreamerv3_attention import __main__ as cli
from dreamerv3_attention.config import Config
from dreamerv3_attention.environment import CarlaEnv
from dreamerv3_attention.evaluation import collection_scenarios
from dreamerv3_attention.synthetic import SyntheticEnv
from dreamerv3_attention.tests.test_model import small_config
from dreamerv3_attention.tests.test_environment import reset_test_environment
from dreamerv3_attention.tests.test_viewer import FakeEnv
from dreamerv3_attention.training import Trainer


NAMES = ["train_heavy_rain", "train_low_light", "train_combined_weather_hazard"]
WEATHER = ["HardRainNoon", "ClearNight", "HardRainNight"]


@pytest.mark.parametrize("names", ["rain", [""], [1], ["rain", "rain"], [[]]])
def test_invalid_scenario_list(names):
    with pytest.raises(ValueError, match="scenarios"):
        replace(Config(), scenarios=names).validate()


def test_scenario_selection_preserves_order_and_split():
    cfg = Config(scenarios=NAMES)
    records = collection_scenarios(cfg)
    assert [s["name"] for s in records] == NAMES
    assert [s["weather"] for s in records] == WEATHER
    assert collection_scenarios(Config())[0]["weather"] == "ClearNoon"
    assert collection_scenarios(Config(scenario=NAMES[0])) == records[:1]
    with pytest.raises(ValueError, match="selected split"):
        collection_scenarios(replace(cfg, split="validation"))
    with pytest.raises(ValueError, match="not both"):
        replace(cfg, scenario=NAMES[0]).validate()


def fake_carla_env(cfg, *, scenarios, start_episode=0):
    """Exercise real weather switching with synthetic observations and physics."""
    env = object.__new__(CarlaEnv)
    env.cfg, env.carla = cfg, carla
    env.scenarios, env.scenario_episode = scenarios, start_episode
    env.world = NS(set_weather=lambda weather: applied.append(weather))
    env._destroy = lambda: None
    env.close = lambda: None
    synthetic = SyntheticEnv(cfg)
    telemetry = FakeEnv()
    for name in ("ego", "rig", "route", "state", "last_bundle"):
        setattr(env, name, getattr(telemetry, name))
    applied = []

    def reset():
        assert applied[-1] == getattr(carla.WeatherParameters, env.scenario["weather"])
        return synthetic.reset()

    def step(action):
        obs, reward, terminated, truncated, info = synthetic.step(action)
        telemetry.step(action)
        info.update(scenario=env.scenario["name"], weather=env.scenario["weather"])
        return obs, reward, terminated, truncated, info

    env._reset, env.step = reset, step
    return env


@pytest.mark.parametrize("command", ["collect", "train"])
@pytest.mark.parametrize("steps", [15, 18])
def test_cli_cycles_shared_replay_and_resumes_cycle(monkeypatch, tmp_path, command, steps):
    cfg = small_config(scenarios=NAMES, steps=steps, max_episode_steps=6, max_objects=0,
                       checkpoint_every=6)
    config = tmp_path / "config.json"
    cfg.save(config)
    monkeypatch.setattr(cli, "CarlaEnv", fake_carla_env)
    monkeypatch.setattr(cli, "workload_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *a: None)
    output = tmp_path / "first"
    argv = [command, "--config", str(config), "--output", str(output), "--budget-seconds", "60"]
    result = cli.main(argv)
    assert result["frames"] == steps
    assert (result["updates"] > 0) == (command == "train")
    rows = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["scenario"] for row in rows] == [NAMES[i // 6] for i in range(steps)]
    assert [row["weather"] for row in rows] == [WEATHER[i // 6] for i in range(steps)]
    live = json.loads((output / "live.json").read_text())
    assert live["status"] == "ended" and live["metrics"]["frames"] == steps
    assert live["metrics"]["weather"] == WEATHER[(steps - 1) // 6]
    trainer = Trainer.restore(output / "latest.pt")
    assert trainer.completed_episodes == steps // 6
    assert len(trainer.replay) == steps
    assert trainer.cfg.scenarios == NAMES

    replace(cfg, steps=steps + 6).save(config)
    continued = tmp_path / "continued"
    cli.main([command, "--config", str(config), "--output", str(continued),
              "--budget-seconds", "60", "--checkpoint", str(output / "latest.pt")])
    rows = [json.loads(row) for row in (continued / "metrics.jsonl").read_text().splitlines()]
    assert {row["scenario"] for row in rows} == {NAMES[(steps // 6) % len(NAMES)]}
    trainer = Trainer.restore(continued / "latest.pt")
    assert len(trainer.replay) == steps + 6
    sample = trainer.replay.sample().reshape(cfg.batch, cfg.sequence)
    assert (sample["episode"] == sample["episode"][:, :1]).all()


def varied_route_env(monkeypatch, *, cfg=None, start_episode=0):
    env = reset_test_environment(monkeypatch)
    env.cfg = cfg or Config(route_end=1)
    env.scenarios = [{"name": "rain", "seed": 10, "category": "heavy_rain"},
                     {"name": "night", "seed": 11, "category": "low_light"}]
    env.scenario_episode = start_episode
    env.world.set_weather = lambda _: None
    spawns = [carla.Transform(carla.Location(x=i*40., y=(i % 3)*30.)) for i in range(8)]
    env.map.get_spawn_points = lambda: spawns
    planned = []

    def trace(start, end):
        planned.append((start, end))
        return [(NS(transform=carla.Transform(point), is_junction=False, lane_width=3.5), None)
                for point in (start, end)]

    env.route_planner = NS(trace_route=trace)
    return env, planned


def route_record(env):
    env.reset()
    return (env.scenario["name"], env.route_start, env.route_end, env.episode_seed,
            env.rng.uniform(size=4).tolist())


def test_each_scenario_varies_starts_destinations_and_actor_seeds_reproducibly(monkeypatch):
    env, planned = varied_route_env(monkeypatch)
    records = [route_record(env) for _ in range(32)]
    plans = len(planned)
    for name in ("rain", "night"):
        rows = [row for row in records if row[0] == name]
        assert len({row[1:3] for row in rows}) == 16
        assert len({row[1] for row in rows}) >= 4
        assert len({row[2] for row in rows}) >= 4
        assert len({row[3] for row in rows}) == 16
    again, _ = varied_route_env(monkeypatch)
    assert [route_record(again) for _ in range(32)] == records
    # Cached route geometry must not retain the previous episode's projection.
    env.route_pools["rain"][0][3].index = 123
    route_record(env)
    assert env.route.index == 0
    assert len(planned) == plans


def test_route_and_actor_seed_resume_at_completed_episode(monkeypatch):
    env, _ = varied_route_env(monkeypatch)
    records = [route_record(env) for _ in range(18)]
    resumed, _ = varied_route_env(monkeypatch, start_episode=13)
    assert [route_record(resumed) for _ in range(5)] == records[13:]
    changed, _ = varied_route_env(monkeypatch, cfg=Config(seed=8, route_end=1))
    assert route_record(changed) != records[0]


@pytest.mark.parametrize("split,randomize,collection", [
    ("train", False, True), ("validation", True, True), ("held_out", True, True), ("train", True, False)])
def test_fixed_and_evaluation_routes_stay_authored(monkeypatch, split, randomize, collection):
    env, _ = varied_route_env(monkeypatch, cfg=Config(route_end=1, split=split, randomize_routes=randomize))
    if not collection:
        env.scenarios = None
    for _ in range(3):
        env.reset()
        assert (env.route_start, env.route_end) == (0, 1)


def test_varied_routes_reserve_other_split_endpoint_pairs(monkeypatch, tmp_path):
    records = [{"name": "val", "split": "validation", "seed": 101,
                "route_start": 2, "route_end": 3, "category": "straight"},
               {"name": "test", "split": "held_out", "seed": 102,
                "route_start": 4, "route_end": 5, "category": "straight"}]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(records))
    env, _ = varied_route_env(monkeypatch, cfg=Config(route_end=1, scenario_file=str(path)))
    rows = [route_record(env) for _ in range(32)]
    assert not {row[1:3] for row in rows} & {(2, 3), (3, 2), (4, 5), (5, 4)}


def test_route_pool_filters_geometry_and_bounds_unsupported_search(monkeypatch):
    env, planned = varied_route_env(monkeypatch)
    env.scenarios[0]["category"] = "intersection"
    trace = env.route_planner.trace_route

    def junction_trace(start, end):
        waypoints = trace(start, end)
        waypoints[0][0].is_junction = start.x % 80 == 0
        return waypoints

    env.route_planner.trace_route = junction_trace
    for _ in range(16):
        env.reset()
        if env.scenario["name"] == "rain":
            assert env.route.junctions.any()
    assert len(planned) <= 256
    env, planned = varied_route_env(monkeypatch)
    env.scenarios[0]["requires_crosswalk"] = True
    with pytest.raises(RuntimeError, match="no valid varied route"):
        env.reset()
    assert len(planned) <= 128


def test_route_pool_does_not_suppress_simulator_errors(monkeypatch):
    env, _ = varied_route_env(monkeypatch)
    env.scenarios[0]["requires_speed_limit_transition"] = True

    def disconnected(_):
        raise RuntimeError("simulator disconnected")

    env.map.get_all_landmarks_of_type = disconnected
    with pytest.raises(RuntimeError, match="simulator disconnected"):
        env.reset()


@pytest.mark.parametrize("scenario,points,valid", [
    ({"category": "straight"}, [(0, 0), (50, 0), (100, 0)], True),
    ({"category": "straight"}, [(0, 0), (50, 0), (50, 50)], False),
    ({"category": "curved"}, [(0, 0), (50, 0), (100, 0)], False),
    ({"category": "curved"}, [(0, 0), (50, 0), (50, 50)], True),
    ({"requires_crosswalk": True}, [(0, 0), (100, 0)], True),
    ({"requires_crosswalk": True}, [(0, 20), (100, 20)], False),
    ({"requires_speed_limit_transition": True}, [(0, 0), (100, 0)], True),
    ({"requires_speed_limit_transition": True}, [(0, 20), (100, 20)], False),
])
def test_varied_routes_retain_scenario_capability_checks(monkeypatch, scenario, points, valid):
    env, _ = varied_route_env(monkeypatch)
    env.scenario = scenario
    env.map.get_crosswalks = lambda: [carla.Location(x=x, y=y) for x, y in
                                    [(40, -3), (45, -3), (45, 3), (40, 3), (40, -3)]]
    # reset_test_environment stubs the polygon parser; use the real parser here.
    from dreamerv3_attention.navigation import crosswalk_polygons
    import dreamerv3_attention.environment as environment_module
    monkeypatch.setattr(environment_module, "crosswalk_polygons", crosswalk_polygons)
    env.map.get_all_landmarks_of_type = lambda _: [
        NS(transform=carla.Transform(carla.Location(x=20.)), value=30.),
        NS(transform=carla.Transform(carla.Location(x=70.)), value=50.)]
    waypoints = [(NS(transform=carla.Transform(carla.Location(x=x, y=y)), is_junction=False), None)
                 for x, y in points]
    if valid:
        route, _ = env._route_geometry(waypoints)
        assert route.length >= 30.
    else:
        with pytest.raises(RuntimeError, match="Unsupported scenario"):
            env._route_geometry(waypoints)


@pytest.mark.parametrize("setting", [0, 1, "true", None])
def test_randomize_routes_requires_boolean(setting):
    with pytest.raises(ValueError, match="randomize_routes"):
        replace(Config(), randomize_routes=setting).validate()

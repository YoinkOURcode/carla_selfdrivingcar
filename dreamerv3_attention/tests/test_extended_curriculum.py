"""Town01 low-visibility and varied rehearsal curriculum contracts."""

from dataclasses import asdict
import math
from pathlib import Path

import carla
import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.driving import DrivingState, compute_target_speed
from dreamerv3_attention.evaluation import collection_scenarios
from dreamerv3_attention.weather import scenario_weather


ROOT = Path("runs/experiment_26/curriculum")
STAGES = ("stage5_wet_weather", "stage6_low_visibility", "stage7_town01_variety")


def test_stage_extensions_preserve_previous_tasks_and_checkpoint_contract():
    previous = Config.load(ROOT / (STAGES[0] + ".json")).resolved()
    for label, count in zip(STAGES[1:], (15, 51)):
        cfg = Config.load(ROOT / (label + ".json")).resolved()
        before, after = asdict(previous), asdict(cfg)
        assert {key for key in before if before[key] != after[key]} == {
            "steps", "scenarios", "scenario_file"}
        assert cfg.steps > previous.steps
        assert cfg.town == "Town01" and cfg.split == "train"
        assert cfg.scenarios[:len(previous.scenarios)] == previous.scenarios
        records = collection_scenarios(cfg)
        assert records[:len(previous.scenarios)] == collection_scenarios(previous)
        assert len(records) == count
        assert math.gcd(cfg.expert_episode_interval, count) == 1
        previous = cfg


@pytest.mark.parametrize("label", STAGES[1:])
def test_all_extended_cases_have_valid_applied_weather_and_supported_hazards(label):
    cfg = Config.load(ROOT / (label + ".json")).resolved()
    for record in collection_scenarios(cfg):
        weather = scenario_weather(carla, record, cfg.weather)
        for field in ("wetness", "fog_density", "fog_distance", "sun_altitude_angle"):
            if field in record:
                assert getattr(weather, field) == pytest.approx(record[field])
        assert record["split"] == "train"
        actors = record.get("actors", [])
        if record.get("requires_pedestrians"):
            assert any(actor["kind"] == "pedestrian" for actor in actors)
        if record.get("requires_vehicles"):
            assert any(actor["kind"] == "vehicle" for actor in actors)
        assert len(actors) <= cfg.max_objects


def test_stage6_adds_fog_dusk_night_and_combined_conditions():
    cfg = Config.load(ROOT / "stage6_low_visibility.json").resolved()
    records = collection_scenarios(cfg)[9:]
    weather = [scenario_weather(carla, record, cfg.weather) for record in records]
    assert any(w.fog_density >= 60 for w in weather)
    assert any(0 <= w.sun_altitude_angle <= 5 for w in weather)
    assert any(w.sun_altitude_angle < 0 for w in weather)
    assert any(w.sun_altitude_angle < 0 and w.precipitation > 0 for w in weather)
    assert any(r.get("requires_traffic_light") for r in records)
    assert any(r.get("requires_pedestrians") for r in records)
    for w in weather:
        state = DrivingState(speed_limit_mps=10., rain=w.precipitation / 100.,
                             wetness=w.wetness / 100., night=float(w.sun_altitude_angle < 0),
                             visibility_m=max(5., 100. * (1. - w.fog_density / 110.)))
        assert 0 < compute_target_speed(state) < 10.


def test_stage7_adds_route_weather_and_actor_variety_for_every_skill():
    cfg = Config.load(ROOT / "stage7_town01_variety.json").resolved()
    records = collection_scenarios(cfg)[15:]
    assert len({r["seed"] for r in records}) == 36
    for skill in range(1, 7):
        variants = [r for r in records if r["name"].startswith(f"train_stage7_skill{skill}_")]
        assert len(variants) == 6
        assert all(r["randomize_routes"] for r in variants)
        assert len({r["weather"] for r in variants}) >= 4
        if skill != 3:
            assert len({r.get("route_length_m") for r in variants}) >= 3
    mixed = [r for r in records if r["name"].startswith("train_stage7_skill4_")]
    assert len({r["actors"][0]["x"] for r in mixed}) == 6
    assert len({r["actors"][0]["speed_mps"] for r in mixed}) == 6
    assert any(a.get("child") for r in mixed for a in r["actors"])

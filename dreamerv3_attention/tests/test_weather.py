"""Offline tests for validated copies of CARLA weather presets."""

import math

import carla
import pytest

from dreamerv3_attention.evaluation import _validate_catalog
from dreamerv3_attention.weather import scenario_weather


def numeric_fields(weather):
    return {name: getattr(weather, name) for name in dir(weather)
            if not name.startswith("_") and isinstance(getattr(weather, name), (int, float))
            and not isinstance(getattr(weather, name), bool)}


def test_without_override_returns_preset_unchanged():
    preset = carla.WeatherParameters.WetNoon
    assert scenario_weather(carla, {}, "WetNoon") is preset
    assert scenario_weather(carla, {"weather": "WetNoon"}, "ClearNoon") is preset


def test_weather_overrides_copy_other_preset_fields_without_mutating_it():
    preset = carla.WeatherParameters.SoftRainNoon
    before = numeric_fields(preset)
    result = scenario_weather(carla, {"weather": "SoftRainNoon", "wetness": 60,
                                     "fog_density": 40, "fog_distance": 30,
                                     "sun_altitude_angle": -8}, "ClearNoon")

    assert result is not preset
    assert result.wetness == 60.
    assert result.fog_density == 40.
    assert result.fog_distance == 30.
    assert result.sun_altitude_angle == -8.
    assert result.precipitation == 30.
    overridden = {"wetness", "fog_density", "fog_distance", "sun_altitude_angle"}
    assert {k: v for k, v in numeric_fields(result).items() if k not in overridden} == {
        k: v for k, v in before.items() if k not in overridden}
    assert numeric_fields(preset) == before


def test_wet_noon_override_supplies_nonzero_wetness():
    preset = carla.WeatherParameters.WetNoon
    assert preset.wetness == 0.
    weather = scenario_weather(carla, {"weather": "WetNoon", "wetness": 60.}, "ClearNoon")
    assert weather.wetness == 60.
    assert weather is not preset


@pytest.mark.parametrize(("field", "values"), [
    ("wetness", [True, "60", -1, 101, math.nan, math.inf, -math.inf]),
    ("fog_density", [False, "60", -1, 101, math.nan, math.inf]),
    ("fog_distance", [True, "60", -1, math.nan, math.inf]),
    ("sun_altitude_angle", [False, "-8", -91, 91, math.nan, math.inf]),
])
def test_invalid_weather_override_is_rejected(field, values):
    for value in values:
        with pytest.raises(ValueError, match=field):
            scenario_weather(carla, {field: value}, "ClearNoon")


@pytest.mark.parametrize(("field", "value"), [
    ("wetness", 101), ("fog_density", -1), ("fog_distance", math.inf),
    ("sun_altitude_angle", -91),
])
def test_catalog_rejects_invalid_weather_override(field, value):
    record = {"name": "invalid-weather", "split": "train", "seed": 1,
              "route_start": 0, "route_end": 1, "category": "straight", field: value}
    with pytest.raises(ValueError, match=field):
        _validate_catalog([record])

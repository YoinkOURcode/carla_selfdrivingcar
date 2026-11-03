"""Scenario-specific CARLA weather values."""

import math


_OVERRIDE_BOUNDS = {
    "wetness": (0., 100.),
    "fog_density": (0., 100.),
    "fog_distance": (0., None),
    "sun_altitude_angle": (-90., 90.),
}


def validate_weather_overrides(scenario):
    """Validate supported finite scenario weather fields and return overrides."""
    overrides = {}
    for name, (minimum, maximum) in _OVERRIDE_BOUNDS.items():
        if name not in scenario:
            continue
        value = scenario[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < minimum or
                (maximum is not None and value > maximum)):
            upper = f" and at most {maximum:g}" if maximum is not None else ""
            raise ValueError(f"scenario {name} must be finite, at least {minimum:g}{upper}")
        overrides[name] = float(value)
    return overrides


def scenario_weather(carla, scenario, fallback_name):
    """Return a preset or a fresh preset copy with scenario weather overrides."""
    name = scenario.get("weather", fallback_name)
    preset = getattr(carla.WeatherParameters, name)
    overrides = validate_weather_overrides(scenario)
    if not overrides:
        return preset

    weather = carla.WeatherParameters()
    for field in dir(preset):
        if field.startswith("_"):
            continue
        value = getattr(preset, field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(weather, field, value)
    for field, value in overrides.items():
        setattr(weather, field, value)
    return weather

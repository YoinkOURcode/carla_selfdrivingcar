"""Deterministic checks for bounded red-light restart assistance."""

from dataclasses import replace
from types import SimpleNamespace as NS

import carla
import numpy as np
import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor
from dreamerv3_attention.environment import CarlaEnv
from dreamerv3_attention.tests.test_environment import fake_environment, fake_step_environment, safe_external_fields
from dreamerv3_attention.sensors import PerceptionPrediction


def supervisor(**settings):
    return SafetySupervisor(DrivingConfig(restart_assist_enabled=True, **settings))


def signal_state(phase="Red", **settings):
    return DrivingState(speed_mps=0., speed_limit_mps=8., traffic_light_id="light-1",
                        traffic_light_phase=phase, traffic_light_distance_m=4.5, **settings)


def red_stop(driver):
    red = signal_state(red_light=True, red_light_distance_m=4.5)
    for _ in range(11):
        result = driver.step(red, (0., -1.), .05)
    assert result.restart_assist_phase == "stopped"


def enter_assist(driver):
    red_stop(driver)
    green = signal_state("Green")
    for _ in range(11):
        result = driver.step(green, (0., -1.), .05)
    assert result.restart_assist_phase == "assist"
    return green, result


def test_restart_assist_is_opt_in_and_preserves_default_red_release():
    state = signal_state("Green")
    disabled = SafetySupervisor()
    assert disabled.step(state, (0., -1.), .05).action[1] == -1.
    assert Config().resolved().driving["restart_assist_enabled"] is False
    assert Config(driving={"restart_assist_enabled": True}).resolved().driving[
        "restart_assist_enabled"] is True


@pytest.mark.parametrize("value", [1, "true", None])
def test_assist_enable_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        Config(driving={"restart_assist_enabled": value}).validate()


@pytest.mark.parametrize("name", [
    "restart_stop_distance_m", "restart_stop_confirm_s", "restart_green_grace_s",
    "restart_target_speed_mps", "restart_max_throttle", "restart_timeout_s",
    "restart_handover_s",
])
def test_restart_limits_must_be_positive_finite(name):
    for value in (0., -1., float("nan"), float("inf"), True, None, "fast"):
        with pytest.raises(ValueError):
            Config(driving={name: value}).validate()


def test_confirmed_green_overrides_actor_brake_but_keeps_speed_target_and_steering():
    driver = supervisor()
    green, result = enter_assist(driver)
    assert result.action[1] == pytest.approx(.4)
    assert result.action[0] == pytest.approx(0.)
    assert result.target_speed_mps == pytest.approx(8.)
    assert "restart_assist_active" in result.intervention_reasons
    assert driver.intervention_count > 0


def test_assist_runs_fake_plant_until_forward_progress_and_hands_back():
    driver = supervisor()
    green, _ = enter_assist(driver)
    speed, progress = 0., 0.
    handed_back = False
    for _ in range(100):
        current = replace(green, speed_mps=speed, route_progress_m=progress,
                           traffic_light_distance_m=4.5-progress)
        actor = .2 if speed >= 1.0 else -1.
        result = driver.step(current, (.1, actor), .05)
        old_speed = speed
        speed = max(0., speed + result.action[1] * 1.5 * .05)
        progress += .5 * (old_speed + speed) * .05
        if result.restart_assist_event == "handover":
            handed_back = True
            assert result.action[1] == actor
            break
    assert handed_back
    assert progress >= .5
    assert speed >= 1.


def test_natural_actor_restart_completes_during_grace_without_assist():
    driver = supervisor()
    red_stop(driver)
    green = signal_state("Green")
    events = []
    for _ in range(11):
        state = replace(green, speed_mps=.6, route_progress_m=.3)
        events.append(driver.step(state, (0., .4), .05))
    assert all(result.restart_assist_phase != "assist" for result in events)
    assert events[-1].restart_assist_event == "natural_restart"
    assert events[-1].action[1] == .4
    assert "restart_assist_natural_restart" not in events[-1].intervention_reasons


def test_transient_actor_motion_does_not_prevent_assist():
    driver = supervisor()
    red_stop(driver)
    green = signal_state("Green")
    for _ in range(3):
        driver.step(replace(green, speed_mps=.6, route_progress_m=.3), (0., .4), .05)
    result = None
    for _ in range(11):
        result = driver.step(green, (0., -1.), .05)
    assert result.restart_assist_phase == "assist"
    assert result.action[1] > 0.


def test_timeout_spends_attempt_and_green_cannot_retrigger():
    driver = supervisor(restart_timeout_s=.6)
    green, _ = enter_assist(driver)
    events = []
    for _ in range(30):
        result = driver.step(green, (0., -1.), .05)
        events.append(result.restart_assist_event)
    assert "timeout" in events
    assert driver._restart_phase == "spent"
    assert driver.step(green, (0., -1.), .05).action[1] == -1.


def test_fresh_red_cycle_rearms_after_timeout_but_a_different_green_does_not():
    driver = supervisor(restart_timeout_s=.6)
    green, _ = enter_assist(driver)
    for _ in range(20):
        driver.step(green, (0., -1.), .05)
    other_green = replace(green, traffic_light_id="light-2")
    assert driver.step(other_green, (0., -1.), .05).action[1] == -1.
    new_red = replace(signal_state("Red", red_light=True, red_light_distance_m=3.))
    for _ in range(11):
        driver.step(new_red, (0., -1.), .05)
    restarted_green = replace(new_red, red_light=False, red_light_distance_m=None,
                              traffic_light_phase="Green")
    result = None
    for _ in range(11):
        result = driver.step(restarted_green, (0., -1.), .05)
    assert result.restart_assist_phase == "assist"


@pytest.mark.parametrize("change", [
    {"front_hazard_m": 2.}, {"side_hazard_m": 2.5}, {"pedestrian_risk": .5},
    {"child_pedestrian_risk": .5}, {"pedestrian_conflict": True},
    {"crossing_occupied": True}, {"lane_error_m": 1.5}, {"heading_error_rad": .5},
    {"lane_error_m": float("nan")}, {"speed_mps": float("nan")},
    {"side_hazard_m": float("nan")},
    {"offroad": True}, {"lane_departure": True}, {"collision": True},
    {"pedestrian_collision": True},
])
def test_hazard_or_bad_alignment_cancels_and_cannot_add_throttle(change):
    driver = supervisor()
    green, _ = enter_assist(driver)
    red_stop(driver)
    # Re-enter with a clean red stop after the previous green attempt.
    green = signal_state("Green")
    for _ in range(11):
        driver.step(green, (0., -1.), .05)
    state = replace(green, **change)
    result = driver.step(state, (0., -1.), .05)
    assert result.restart_assist_phase == "spent"
    assert result.action[1] <= 0.


def test_invalid_observation_action_and_unknown_signal_cancel_active_attempt():
    driver = supervisor()
    green, _ = enter_assist(driver)
    invalid = driver.step(replace(green, observations_valid=False), (0., -1.), .05)
    assert invalid.action[1] == -1.
    assert invalid.restart_assist_phase == "spent"
    driver.reset()
    enter_assist(driver)
    invalid_action = driver.step(green, (0., float("nan")), .05)
    assert invalid_action.action[1] == -1.
    assert invalid_action.restart_assist_phase == "spent"
    driver.reset()
    enter_assist(driver)
    unknown = driver.step(replace(green, traffic_light_phase=None), (0., -1.), .05)
    assert unknown.action[1] == -1.
    assert unknown.restart_assist_event == "cancel_signal_unknown"


@pytest.mark.parametrize("metadata", [
    {"traffic_light_id": None}, {"traffic_light_id": ""},
    {"traffic_light_distance_m": float("nan")},
    {"traffic_light_phase": "Unknown"},
    {"traffic_light_phase": []},
])
def test_missing_or_malformed_signal_metadata_fails_closed(metadata):
    driver = supervisor()
    green, _ = enter_assist(driver)
    result = driver.step(replace(green, **metadata), (0., -1.), .05)
    assert result.action[1] <= 0.
    assert result.restart_assist_phase == "spent"


def test_future_red_and_crosswalk_stop_keep_final_safety_authority():
    driver = supervisor()
    green, _ = enter_assist(driver)
    red = replace(green, red_light=True, red_light_distance_m=1.5)
    result = driver.step(red, (0., 1.), .05)
    assert result.action[1] == -1.
    assert "red_light_stop" in result.intervention_reasons
    driver.reset()
    enter_assist(driver)
    crossing = replace(green, crossing_id="crosswalk", crossing_distance_m=1.)
    result = driver.step(crossing, (0., 1.), .05)
    assert result.action[1] < 0.
    assert "crosswalk_stop" in result.intervention_reasons


@pytest.mark.parametrize("signal", [
    {"red_light": True, "red_light_distance_m": 25.},
    {"yellow_light": True, "yellow_light_distance_m": 25., "yellow_light_confidence": 1.},
])
def test_new_signal_cancels_assist_even_before_its_braking_zone(signal):
    driver = supervisor()
    green, _ = enter_assist(driver)
    result = driver.step(replace(green, **signal), (0., -1.), .05)
    assert result.restart_assist_phase == "spent"
    assert result.restart_assist_event == "cancel_hazard_or_alignment"
    assert result.action[1] == -1.


def test_assist_brakes_above_its_crawl_target_without_changing_reward_target():
    driver = supervisor()
    green, _ = enter_assist(driver)
    result = driver.step(replace(green, speed_mps=2.), (0., 1.), .05)
    assert result.action[1] < 0.
    assert result.target_speed_mps == 8.


def test_reset_discards_previous_red_stop_and_green_alone_cannot_arm():
    driver = supervisor()
    green, _ = enter_assist(driver)
    driver.reset()
    for _ in range(40):
        result = driver.step(green, (0., -1.), .05)
        assert result.action[1] == -1.
        assert result.restart_assist_phase == "idle"


def test_identity_change_cannot_reuse_a_different_lights_red_stop():
    driver = supervisor()
    red_stop(driver)
    for _ in range(40):
        result = driver.step(replace(signal_state("Green"), traffic_light_id="light-2"), (0., -1.), .05)
        assert result.action[1] == -1.
        assert result.restart_assist_phase == "idle"


def test_identity_change_records_cancellation_of_active_attempt():
    driver = supervisor()
    green, previous = enter_assist(driver)
    result = driver.step(replace(green, traffic_light_id="light-2"), (0., -1.), .05)
    assert result.action[1] == -1.
    assert result.restart_assist_phase == "idle"
    assert result.restart_assist_event == "cancel_signal_changed"
    assert result.restart_assist_elapsed_s == previous.restart_assist_elapsed_s


def test_environment_retains_tracked_green_after_passing_the_stopline():
    driver = supervisor()
    green, _ = enter_assist(driver)
    env = NS(supervisor=driver, cfg=NS(perception="ground_truth", dt=.05), carla=carla,
             traffic_stops=[(NS(id="light-1", get_state=lambda: carla.TrafficLightState.Green), 5.)])
    state = replace(green, route_progress_m=7.)
    CarlaEnv._observe_traffic_light(env, state, 1.)
    assert state.traffic_light_id == "light-1"
    assert state.traffic_light_phase == "Green"
    assert state.traffic_light_distance_m == -3.
    assert driver.step(state, (0., -1.), .05).action[1] > 0.


def test_signal_metadata_does_not_change_policy_observations():
    observations = []
    for enabled in (False, True):
        env, _ = fake_environment()
        env.supervisor = SafetySupervisor(DrivingConfig(restart_assist_enabled=enabled))
        env.traffic_stops = [(NS(id=1, get_state=lambda: carla.TrafficLightState.Red), 20.)]
        state, observation = env._observe(42)
        assert (state.traffic_light_id is not None) == enabled
        observations.append(observation)
    assert observations[0].keys() == observations[1].keys()
    for name in observations[0]:
        np.testing.assert_array_equal(observations[0][name], observations[1][name])


def test_environment_logs_assistance_and_keeps_executed_action():
    driver = supervisor()
    green, _ = enter_assist(driver)
    env = fake_step_environment(before=green, after=green, max_steps=100, timeout_s=10., blocked_timeout_s=20.)
    env.supervisor = driver
    _, _, terminated, truncated, info = env.step((.1, -1.))
    assert not terminated and not truncated
    assert info["requested_action"] == [.1, -1.]
    assert info["applied_action"] == (.1, .4)
    assert env.previous_action == info["applied_action"]
    assert info["restart_assist_phase"] == "assist"
    assert info["restart_assist_event"] == "active"
    assert info["restart_assist_elapsed_s"] > 0.


def test_restart_metadata_survives_stopline_pass_and_longitudinal_reference_ignores_assist():
    driver = supervisor()
    green, _ = enter_assist(driver)
    before = vars(driver).copy()
    reference = driver.longitudinal_reference(green, .05)
    assert reference > 0.
    assert vars(driver) == before
    moving = replace(green, speed_mps=1.5)
    assert driver.longitudinal_reference(moving, .05) > 0.
    assert vars(driver) == before
    passed = replace(green, traffic_light_distance_m=-5.)
    assert driver.step(passed, (0., -1.), .05).action[1] > 0.


def test_environment_selects_green_identity_and_still_detects_later_red():
    env = NS(supervisor=SafetySupervisor(DrivingConfig(restart_assist_enabled=True)),
             cfg=NS(perception="ground_truth", dt=.05), carla=carla,
             traffic_stops=[
                 (NS(id=1, get_state=lambda: carla.TrafficLightState.Green), 5.),
                 (NS(id=2, get_state=lambda: carla.TrafficLightState.Red), 15.)])
    state = DrivingState(speed_limit_mps=8.)
    CarlaEnv._observe_traffic_light(env, state, 1.)
    assert (state.traffic_light_id, state.traffic_light_phase) == ("1", "Green")
    assert state.red_light and state.red_light_distance_m == 14.


def test_default_detector_keeps_first_non_green_stop_flags():
    env = NS(supervisor=SafetySupervisor(), cfg=NS(perception="ground_truth", dt=.05), carla=carla,
             traffic_stops=[
                 (NS(id=1, get_state=lambda: carla.TrafficLightState.Yellow), 5.),
                 (NS(id=2, get_state=lambda: carla.TrafficLightState.Red), 15.)])
    state = DrivingState(speed_limit_mps=8.)
    CarlaEnv._observe_traffic_light(env, state, 1.)
    assert state.yellow_light and not state.red_light
    assert state.yellow_light_distance_m == 4.


def test_assist_does_not_publish_signal_metadata_in_mixed_perception():
    env = NS(supervisor=SafetySupervisor(DrivingConfig(restart_assist_enabled=True)),
             cfg=NS(perception="mixed", dt=.05), carla=carla,
             traffic_stops=[(NS(id=1, get_state=lambda: carla.TrafficLightState.Green), 5.)])
    state = DrivingState(speed_limit_mps=8.)
    CarlaEnv._observe_traffic_light(env, state, 1.)
    assert state.traffic_light_id is None
    assert state.traffic_light_phase is None


def test_external_provider_cannot_inject_supervisor_signal_metadata():
    fields = safe_external_fields()
    fields.update(traffic_light_id="fake", traffic_light_phase="Green",
                  traffic_light_distance_m=4.)
    prediction = PerceptionPrediction(np.zeros((0, 16), dtype=np.float32),
                                      driving=fields, frame=42, timestamp=2.1)
    env, _ = fake_environment("external", prediction)
    env.cfg = replace(env.cfg, driving={"restart_assist_enabled": True})
    with pytest.raises(ValueError, match="may replace perception fields only"):
        env._observe(42)


def far_red(distance=21., **settings):
    return DrivingState(speed_mps=0., speed_limit_mps=8., red_light=True,
                        red_light_distance_m=distance, **settings)


def test_signal_approach_assist_moves_a_car_stopped_far_from_a_red_light():
    driver = supervisor()
    for _ in range(5):  # the stop is not confirmed yet
        assert driver.step(far_red(), (0., -.1), .05).action[1] < 0
    for _ in range(10):
        result = driver.step(far_red(), (0., -.1), .05)
    assert "signal_approach_assist" in result.intervention_reasons
    assert result.action[1] == pytest.approx(DrivingConfig().restart_max_throttle)
    # It stays latched while the car creeps, so the actor cannot stall it again.
    moving = replace(far_red(), speed_mps=1.)
    assert driver.step(moving, (0., -.1), .05).action[1] > 0


@pytest.mark.parametrize("state", [
    far_red(1.5),                           # inside the stop area: hold the red
    far_red(21., pedestrian_conflict=True),  # hard stop wins
    far_red(21., front_hazard_m=1.),
    far_red(21., observations_valid=False),
])
def test_signal_approach_assist_never_overrides_stops(state):
    driver = supervisor()
    for _ in range(30):
        result = driver.step(state, (0., -.1), .05)
    assert result.action[1] < 0
    assert "signal_approach_assist" not in result.intervention_reasons


def test_signal_approach_assist_is_off_by_default_and_ends_with_the_red_light():
    off = SafetySupervisor()
    for _ in range(30):
        assert off.step(far_red(), (0., -.1), .05).action[1] < 0
    driver = supervisor()
    for _ in range(30):
        driver.step(far_red(), (0., -.1), .05)
    green = replace(far_red(), red_light=False, red_light_distance_m=None)
    assert "signal_approach_assist" not in driver.step(green, (0., -.1), .05).intervention_reasons

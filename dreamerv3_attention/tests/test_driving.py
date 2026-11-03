import unittest
from dataclasses import replace

import pytest

from dreamerv3_attention.config import Config

from dreamerv3_attention.driving import (
    DrivingConfig, DrivingState, RewardConfig, SafetySupervisor, action_to_controls, compute_reward,
    compute_target_speed,
    signal_wait,
)


class TargetSpeedTests(unittest.TestCase):
    def test_uncertain_limit_uses_conservative_configured_default(self):
        self.assertEqual(compute_target_speed(DrivingState(
            speed_limit_mps=20.0, speed_limit_confidence=0.1)), 3.0)

    def test_weather_and_pedestrian_risk_lower_target(self):
        clear = compute_target_speed(DrivingState(speed_limit_mps=12))
        risk = compute_target_speed(DrivingState(
            speed_limit_mps=12, rain=1, wetness=1, visibility_m=20,
            pedestrian_risk=0.8))
        self.assertGreater(clear, risk)
        self.assertGreaterEqual(risk, 0)

    def test_close_lead_vehicle_limits_target(self):
        self.assertEqual(compute_target_speed(DrivingState(
            speed_limit_mps=12, front_hazard_m=4)), 0)


class SupervisorTests(unittest.TestCase):
    def test_signed_longitudinal_maps_to_throttle_or_brake(self):
        self.assertEqual(action_to_controls((1.5, 0.7)), (1.0, 0.7, 0.0))
        self.assertEqual(action_to_controls((-0.2, -0.4)), (-0.2, 0.0, 0.4))

    def test_invalid_observation_fails_safe_and_counts_intervention(self):
        supervisor = SafetySupervisor()
        result = supervisor.step(DrivingState(observations_valid=False), (0.8, 1), 0.1)
        self.assertEqual(result.action, (0.0, -1.0))
        self.assertIn("invalid_observation", result.intervention_reasons)
        self.assertEqual(supervisor.intervention_count, 1)

    def test_action_bounds_and_side_hazard_limits_steering(self):
        result = SafetySupervisor().step(DrivingState(side_hazard_m=1), (1.5, -2), 0.1)
        self.assertEqual(result.action, (0.25, -1.0))
        self.assertIn("side_hazard_steering_limit", result.intervention_reasons)

    def test_rear_hazard_does_not_suppress_braking(self):
        result = SafetySupervisor().step(DrivingState(rear_hazard_m=1), (0, -0.8), 0.1)
        self.assertEqual(result.action[1], -0.8)

    def test_nonfinite_action_fails_safe(self):
        supervisor = SafetySupervisor()
        result = supervisor.step(DrivingState(), (float("nan"), 0.5), 0.1)
        self.assertEqual(result.action, (0.0, -1.0))
        self.assertIn("invalid_action", result.intervention_reasons)
        self.assertEqual(action_to_controls((0, float("inf"))), (0.0, 0.0, 1.0))
        malformed = supervisor.step(DrivingState(), (0,), 0.1)
        self.assertEqual(malformed.action, (0.0, -1.0))

    def test_crossing_mandates_stop_then_dwells_and_resumes_once(self):
        supervisor = SafetySupervisor()
        state = DrivingState(speed_mps=1, speed_limit_mps=12,
                             crossing_id="c1", crossing_distance_m=2)
        stopped = supervisor.step(state, (0, 0.3), 0.1)
        self.assertEqual(stopped.action[1], -1)
        self.assertEqual(stopped.target_speed_mps, 0)
        self.assertEqual(stopped.crossing_phase, "stopped")
        state.speed_mps = 0
        supervisor.step(state, (0, 0), 0.5)
        resumed = supervisor.step(state, (0, 0), 0.8)
        self.assertEqual(resumed.crossing_phase, "resume")
        self.assertEqual(resumed.target_speed_mps, 12)
        released = supervisor.step(state, (0, 0.2), 0.1)
        self.assertEqual(released.crossing_phase, "passed")
        self.assertNotIn("crosswalk_stop", released.intervention_reasons)
        still_passed = supervisor.step(state, (0, 0.2), 0.1)
        self.assertEqual(still_passed.crossing_phase, "passed")

    def test_early_crosswalk_stop_has_zero_target_even_before_stop_line(self):
        state = DrivingState(speed_mps=6, speed_limit_mps=12,
                             crossing_id="early", crossing_distance_m=8)
        result = SafetySupervisor().step(state, (0, 0.2), 0.1)
        self.assertEqual(result.crossing_phase, "stopped")
        self.assertEqual(result.target_speed_mps, 0)

    def test_speed_limit_brakes_immediately_and_predictively_limits_throttle(self):
        supervisor = SafetySupervisor()
        over = supervisor.step(DrivingState(speed_mps=3.1, speed_limit_mps=3),
                               (0, 0.5), 0.05)
        self.assertLess(over.action[1], 0)
        self.assertIn("speed_target_brake", over.intervention_reasons)
        approaching = SafetySupervisor().step(
            DrivingState(speed_mps=2.9, speed_limit_mps=3), (0, 0.5), 0.05)
        self.assertGreater(approaching.action[1], 0)
        self.assertLessEqual(2.9 + approaching.action[1] * 3.0 * 0.05, 3.0)
        self.assertIn("speed_target_throttle_limit", approaching.intervention_reasons)

    def test_occupied_crossing_does_not_release(self):
        supervisor = SafetySupervisor()
        state = DrivingState(speed_mps=1, crossing_id="c", crossing_distance_m=1,
                             crossing_occupied=True)
        supervisor.step(state, (0, 0), 0.1)
        state.speed_mps = 0
        result = supervisor.step(state, (0, 0), 20)
        self.assertEqual(result.crossing_phase, "stopped")
        self.assertEqual(result.action[1], -1)


class RewardTests(unittest.TestCase):
    def test_waypoint_rewards_are_counted_once_and_centering_needs_new_progress(self):
        config = RewardConfig(progress=0, lane=0, time=0, idle=0, lane_centering=2)
        state = DrivingState(route_progress_m=14., speed_limit_mps=10.)
        centered = compute_reward(state, (0, 0), DrivingState(route_progress_m=15., speed_limit_mps=10.),
                                  waypoints_reached=1, new_progress_m=1., config=config)
        self.assertEqual(centered.components["waypoint"], 50.)
        self.assertEqual(centered.components["lane_centering"], 2.)
        recross = compute_reward(DrivingState(route_progress_m=16., speed_limit_mps=10.), (0, 0),
                                 DrivingState(route_progress_m=15., speed_limit_mps=10.),
                                 waypoints_reached=0, new_progress_m=0., config=config)
        self.assertEqual(recross.components["waypoint"], 0.)
        self.assertEqual(recross.components["lane_centering"], 0.)

    def test_centering_prefers_route_center_and_stationary_or_reverse_never_earns(self):
        config = RewardConfig(progress=0, lane=0, time=0, idle=0, lane_centering=1)
        start = DrivingState(route_progress_m=5., speed_limit_mps=10.)
        center = compute_reward(start, (0, 0), DrivingState(route_progress_m=7., speed_limit_mps=10.,
            lane_width_m=4., lane_error_m=0.), new_progress_m=2., config=config)
        edge = compute_reward(start, (0, 0), DrivingState(route_progress_m=7., speed_limit_mps=10.,
            lane_width_m=4., lane_error_m=2.), new_progress_m=2., config=config)
        self.assertGreater(center.components["lane_centering"], edge.components["lane_centering"])
        stationary = compute_reward(start, (0, 0), start, new_progress_m=0., config=config)
        reverse = compute_reward(start, (0, 0), DrivingState(route_progress_m=4., speed_limit_mps=10.),
                                 config=config)
        self.assertEqual(stationary.components["lane_centering"], 0.)
        self.assertEqual(reverse.components["lane_centering"], 0.)

    def test_failures_suppress_positive_events_and_apply_terminal_penalties(self):
        for failure in ("offroad", "collision", "red_light_violation"):
            next_state = DrivingState(route_progress_m=15., speed_limit_mps=10.)
            if failure == "offroad":
                next_state.offroad = True
            else:
                setattr(next_state, failure, True)
            result = compute_reward(DrivingState(route_progress_m=14., speed_limit_mps=10.),
                (0, 0), next_state, waypoints_reached=1, new_progress_m=1., success=True, terminated=True)
            self.assertEqual(result.components["waypoint"], 0.)
            self.assertEqual(result.components["lane_centering"], 0.)
            self.assertEqual(result.components["success"], 0.)
            if failure == "offroad":
                self.assertEqual(result.components["offroad"], -100.)
            if failure == "collision":
                self.assertEqual(result.components["collision"], -100.)
        pedestrian = compute_reward(DrivingState(), (0, 0),
            DrivingState(pedestrian_collision=True))
        self.assertEqual(pedestrian.components["collision"], -100.)
        self.assertEqual(pedestrian.components["pedestrian_collision"], -20.)

    def test_checkpoint_arguments_reject_invalid_counts_and_progress(self):
        for kwargs in ({"waypoints_reached": -1}, {"waypoints_reached": 1.5},
                       {"new_progress_m": -1.}, {"new_progress_m": float("nan")}):
            with self.assertRaises(ValueError):
                compute_reward(DrivingState(), (0, 0), DrivingState(), **kwargs)

    def test_progress_positive_but_never_during_violation(self):
        start = DrivingState(route_progress_m=3)
        safe = compute_reward(start, (0, 0.2), DrivingState(route_progress_m=5))
        unsafe = compute_reward(start, (0, 0.2), DrivingState(
            route_progress_m=5, crossing_violation=True))
        self.assertGreater(safe.components["progress"], 0)
        self.assertEqual(unsafe.components["progress"], 0)
        self.assertLess(unsafe.reward, safe.reward)

    def test_lane_departure_and_speeding_gate_progress(self):
        start = DrivingState(route_progress_m=1)
        for violation in (DrivingState(route_progress_m=2, lane_departure=True),
                          DrivingState(route_progress_m=2, speed_mps=4)):
            result = compute_reward(start, (0, 0), violation)
            self.assertEqual(result.components["progress"], 0)

    def test_safe_speed_tracking_reward_does_not_reward_slowing(self):
        cfg = RewardConfig(target_tracking=1.0, time=0, idle=0)
        too_slow = compute_reward(DrivingState(), (0, 0), DrivingState(speed_mps=0.5), config=cfg)
        near_target = compute_reward(DrivingState(), (0, 0), DrivingState(speed_mps=3), config=cfg)
        self.assertLess(too_slow.reward, near_target.reward)

    def test_mandated_crosswalk_stop_has_no_idle_penalty(self):
        result = compute_reward(DrivingState(), (0, 0), DrivingState(
            speed_mps=0, crossing_id="x", crossing_distance_m=2))
        self.assertEqual(result.components["idle"], 0)

    def test_supervisor_target_and_stop_phase_prevent_crosswalk_reward_contradiction(self):
        cfg = RewardConfig(target_tracking=1.0, time=0, idle=1.0)
        previous = DrivingState(speed_mps=0, crossing_id="x", crossing_distance_m=7)
        stopped = DrivingState(speed_mps=0, crossing_id="x", crossing_distance_m=7)
        dwell = compute_reward(previous, (0, -1), stopped, target_speed_mps=0,
                               mandated_stop=True, config=cfg)
        self.assertEqual(dwell.components["target_tracking"], 0)
        self.assertEqual(dwell.components["idle"], 0)

        released = DrivingState(speed_mps=3, crossing_id="x", crossing_distance_m=2)
        resume = compute_reward(previous, (0, 0.2), released, target_speed_mps=3,
                                mandated_stop=False, config=cfg)
        self.assertEqual(resume.components["target_tracking"], 0)
        self.assertEqual(resume.components["idle"], 0)

    def test_invalid_observations_exempt_idle_penalty(self):
        result = compute_reward(DrivingState(), (0, 0), DrivingState(
            speed_mps=0, observations_valid=False))
        self.assertEqual(result.components["idle"], 0)

    def test_success_termination_and_truncation_are_distinct(self):
        result = compute_reward(DrivingState(), (0, 0), DrivingState(),
                                terminated=True, truncated=True, success=True,
                                config=RewardConfig(success=5))
        self.assertEqual(result.components["success"], 5)
        self.assertNotIn("termination", result.components)
        timed_out = compute_reward(DrivingState(), (0, 0), DrivingState(), truncated=True)
        self.assertNotIn("termination", timed_out.components)


if __name__ == "__main__":
    unittest.main()
def test_reverse_progress_cannot_earn_forward_progress_reward():
    from dreamerv3_attention.driving import DrivingState, compute_reward

    before = DrivingState(route_progress_m=10., speed_limit_mps=10.)
    after = DrivingState(route_progress_m=8., speed_limit_mps=10.)
    result = compute_reward(before, (0., 0.), after)
    assert result.components["progress"] < 0.


@pytest.mark.parametrize("distance", [None, -1., float("nan"), float("inf"), "unknown", True])
def test_unknown_or_malformed_red_stop_distance_fails_safe(distance):
    state = DrivingState(speed_mps=3., speed_limit_mps=10., red_light=True,
                         red_light_distance_m=distance)
    assert compute_target_speed(state) == 0.
    result = SafetySupervisor().step(state, (0., 1.), .1)
    assert result.action[1] == -1. and result.target_speed_mps == 0.
    assert "red_light_stop" in result.intervention_reasons


@pytest.mark.parametrize("deceleration", [0., -1., float("nan"), float("inf"), True])
def test_red_deceleration_must_be_positive_and_finite(deceleration):
    with pytest.raises(ValueError):
        Config(driving={"red_light_deceleration_mps2": deceleration}).validate()


def test_red_target_uses_configured_deceleration_and_crosswalk_target_is_unchanged():
    config = DrivingConfig(red_light_deceleration_mps2=1.)
    red = DrivingState(speed_limit_mps=30., red_light=True, red_light_distance_m=10.)
    crosswalk = DrivingState(speed_limit_mps=30., crossing_distance_m=10.)
    assert compute_target_speed(red, config) == pytest.approx(8. / 8.2)
    assert compute_target_speed(red, DrivingConfig(red_light_deceleration_mps2=2.)) > compute_target_speed(red, config)
    assert compute_target_speed(crosswalk, config) == (48.)**.5


def test_red_hold_resets_on_green_new_stop_and_episode_reset():
    supervisor = SafetySupervisor()
    state = DrivingState(speed_mps=8., speed_limit_mps=20., red_light=True, red_light_distance_m=24.)
    assert supervisor.step(state, (0., 1.), .1).action[1] == -1.
    state.speed_mps, state.red_light_distance_m, state.route_progress_m = 0., 2., 22.
    assert supervisor.step(state, (0., 1.), .1).action[1] == -1.
    state.red_light_distance_m = 30.  # A different line must not inherit the hold.
    assert supervisor.step(state, (0., 1.), .1).action[1] > 0.
    state.red_light_distance_m = 1.
    assert supervisor.step(state, (0., 1.), .1).action[1] == -1.
    supervisor.reset()
    state.red_light_distance_m = 10.
    assert supervisor.step(state, (0., 1.), .1).action[1] > 0.
    state.red_light_distance_m = 1.
    supervisor.step(state, (0., 1.), .1)
    state.red_light = False
    assert supervisor.step(state, (0., 1.), .1).action[1] > 0.


@pytest.mark.parametrize("signal,speed,distance", [
    ("red", 8.020601824822357, 23.85023469812186),
    ("yellow", 8.3, 26.),
])
def test_signal_braking_releases_after_stronger_than_expected_deceleration(signal, speed, distance):
    supervisor = SafetySupervisor()
    fields = ({"red_light": True, "red_light_distance_m": distance} if signal == "red" else
              {"yellow_light": True, "yellow_light_distance_m": distance, "yellow_light_confidence": 1.})
    state = DrivingState(speed_mps=speed, speed_limit_mps=10., **fields)
    braking = supervisor.step(state, (0., 1.), .05)
    assert braking.action[1] == -1.
    assert braking.target_speed_mps > 0.  # Approach remains a moving target.

    # experiment_26 stopped 18.85 m early after full brake. Preserve the same
    # stop-line route coordinate as the car advances, then request acceleration.
    stopped = replace(state, speed_mps=0., route_progress_m=distance - 18.85,
                      **{signal + "_light_distance_m": 18.85})
    released = supervisor.step(stopped, (0., 1.), .05)
    assert released.action[1] > 0.
    assert released.target_speed_mps > 0.
    assert supervisor.longitudinal_reference(stopped, .05) > 0.
    assert supervisor.step(stopped, (0., -1.), .05).action[1] == -1.
    conflict = supervisor.step(replace(stopped, pedestrian_conflict=True), (0., 1.), .05)
    assert conflict.action[1] == -1. and conflict.target_speed_mps == 0.


@pytest.mark.parametrize("signal", ["red", "yellow"])
@pytest.mark.parametrize("initial_speed,line", [(3., 20.), (8., 40.), (14., 100.)])
@pytest.mark.parametrize("brake_deceleration", [1.5, 6., 10.])
@pytest.mark.parametrize("dt", [.05, .1])
def test_signal_supervisor_reaches_buffer_with_different_brake_strengths(
        signal, initial_speed, line, brake_deceleration, dt):
    supervisor = SafetySupervisor()
    speed, position = initial_speed, 0.
    for _ in range(int(60. / dt)):
        distance = line - position
        fields = ({"red_light": True, "red_light_distance_m": distance} if signal == "red" else
                  {"yellow_light": True, "yellow_light_distance_m": distance, "yellow_light_confidence": 1.})
        state = DrivingState(speed_mps=speed, speed_limit_mps=initial_speed,
                             route_progress_m=position, **fields)
        result = supervisor.step(state, (0., 1.), dt)
        acceleration = (supervisor.config.max_acceleration_mps2 if result.action[1] >= 0.
                        else brake_deceleration)
        next_speed = max(0., speed + result.action[1] * acceleration * dt)
        position += .5 * (speed + next_speed) * dt
        speed = next_speed
        assert position < line

    assert speed == 0.
    assert 1.5 <= line - position <= 2.5
    assert result.target_speed_mps == 0. and result.action[1] == -1.
    for _ in range(20):
        hold = supervisor.step(replace(state, speed_mps=0., route_progress_m=position,
                                      **{signal + "_light_distance_m": line - position}), (0., 1.), dt)
        assert hold.action[1] == -1. and hold.target_speed_mps == 0.
    green = DrivingState(speed_limit_mps=initial_speed, route_progress_m=position)
    assert supervisor.step(green, (0., 1.), dt).action[1] > 0.


def test_yellow_supervisor_slows_stops_holds_and_releases():
    supervisor = SafetySupervisor()
    state = DrivingState(speed_mps=4., speed_limit_mps=10., yellow_light=True,
                         yellow_light_distance_m=30., yellow_light_confidence=1.)
    approach = supervisor.step(state, (0., 1.), .1)
    assert approach.target_speed_mps < 10.
    assert approach.action[1] <= 0. and "yellow_light_approach" in approach.intervention_reasons
    state.route_progress_m = 27.
    state.yellow_light_distance_m = 3.
    state.speed_mps = 2.
    stopping = supervisor.step(state, (0., 1.), .1)
    assert stopping.action[1] == pytest.approx(-.8)
    assert "yellow_light_approach" in stopping.intervention_reasons
    state.yellow_light_distance_m = 1.9
    state.route_progress_m = 28.1
    assert supervisor.step(state, (0., 1.), .1).action[1] == -1.
    state.speed_mps = 0.
    state.route_progress_m = 28.
    state.yellow_light_distance_m = 2.
    assert supervisor.step(state, (0., 1.), .1).action[1] == -1.
    state.yellow_light = False
    assert supervisor.step(state, (0., 1.), .1).action[1] > 0.


def test_yellow_too_late_to_stop_does_not_trigger_signal_braking():
    supervisor = SafetySupervisor()
    state = DrivingState(speed_mps=8., speed_limit_mps=10., yellow_light=True,
                         yellow_light_distance_m=5., yellow_light_confidence=.4)
    result = supervisor.step(state, (0., .2), .1)
    assert "yellow_light_stop" not in result.intervention_reasons
    assert result.action[1] >= 0.
    state.speed_mps = 0.
    assert "yellow_light_stop" not in supervisor.step(state, (0., .2), .1).intervention_reasons
    state.yellow_light = False
    supervisor.step(state, (0., .2), .1)
    state.yellow_light = True
    # A new yellow decision permits approach from outside the 2.5 m area.
    approach = supervisor.step(state, (0., .2), .1)
    assert "yellow_light_approach" in approach.intervention_reasons
    assert approach.target_speed_mps > 0. and approach.action[1] > 0.


def test_yellow_unknown_distance_prevents_acceleration():
    state = DrivingState(speed_mps=2., speed_limit_mps=10., yellow_light=True,
                         yellow_light_distance_m=None, yellow_light_confidence=.2)
    result = SafetySupervisor().step(state, (0., 1.), .1)
    assert result.action[1] == -1. and "yellow_light_stop" in result.intervention_reasons


@pytest.mark.parametrize("offset", [-.5, -.25, 0., .25, .5])
def test_lane_reward_allows_half_metre_from_center(offset):
    state = DrivingState(speed_mps=2., lane_width_m=4., lane_error_m=offset)
    reward = compute_reward(state, (0., 0.), state, .1, new_progress_m=1.)
    assert reward.components["lane"] == 0.
    assert reward.components["lane_centering"] == 1.


def test_lane_reward_penalizes_offset_beyond_tolerance_and_heading_error():
    state = DrivingState(speed_mps=2., lane_width_m=4., lane_error_m=.7, heading_error_rad=.2)
    reward = compute_reward(state, (0., 0.), state, .1, new_progress_m=1.)
    assert reward.components["lane"] == pytest.approx(-RewardConfig().lane * .3)
    assert reward.components["lane_centering"] == pytest.approx(1. - .2 / 1.5)
    state.lane_departure = True
    departure = compute_reward(state, (0., 0.), state, .1, new_progress_m=1.)
    assert departure.components["lane_departure"] < 0.
    assert departure.components["lane_centering"] == 0.


@pytest.mark.parametrize("signal", ["red", "yellow"])
@pytest.mark.parametrize("buffer", [1., 2.])
def test_signal_wait_area_agrees_with_target_hold_and_imitation(signal, buffer):
    config = DrivingConfig(crossing_buffer_m=buffer)
    fields = {signal + "_light": True, signal + "_light_distance_m": 2.5}
    if signal == "yellow":
        fields["yellow_light_confidence"] = 1.
    state = DrivingState(speed_mps=.1, speed_limit_mps=10., **fields)
    supervisor = SafetySupervisor(config)
    assert signal_wait(state, config)
    assert compute_target_speed(state, config) == 0.
    result = supervisor.step(state, (0., 1.), .05)
    assert result.target_speed_mps == 0. and result.action[1] == -1.
    assert supervisor.longitudinal_reference(state, .05) == -1.
    outside = replace(state, **{signal + "_light_distance_m": 2.501})
    assert not signal_wait(outside, config)
    assert compute_target_speed(outside, config) > 0.
    assert supervisor.step(outside, (0., 1.), .05).action[1] > 0.
    green = replace(state, red_light=False, yellow_light=False)
    assert not signal_wait(green, config)
    assert supervisor.step(green, (0., 1.), .05).action[1] > 0.


@pytest.mark.parametrize("speed", [.2, 1., 5.])
def test_moving_vehicle_in_signal_area_still_approaches_buffer(speed):
    state = DrivingState(speed_mps=speed, speed_limit_mps=10., red_light=True, red_light_distance_m=2.5)
    assert not signal_wait(state)
    assert compute_target_speed(state) > 0.


@pytest.mark.parametrize("overspeed", [.02, .1, 1., 3.])
def test_heavy_rain_speed_brake_is_proportional_and_preserves_stronger_brakes(overspeed):
    state = DrivingState(speed_limit_mps=8.333333333333334, rain=1., wetness=.9)
    target = compute_target_speed(state)
    assert target == pytest.approx(5.416666666666667)
    state.speed_mps = target + overspeed
    result = SafetySupervisor().step(state, (0., 1.), .05)
    expected = -min(1., .5 * overspeed)
    assert result.action[1] == pytest.approx(expected)
    assert "speed_target_brake" in result.intervention_reasons
    stronger = SafetySupervisor().step(state, (0., -1.), .05)
    assert stronger.action[1] == -1.
    conflict = SafetySupervisor().step(replace(state, pedestrian_conflict=True), (0., 1.), .05)
    assert conflict.action[1] == -1.


def test_small_positive_hazard_target_does_not_starve_throttle_at_rest():
    state = DrivingState(speed_limit_mps=8.333333333333334, rain=1., wetness=.9, night=1.,
                         pedestrian_risk=1., child_pedestrian_risk=1.)
    config = DrivingConfig(child_pedestrian_speed_factor=.01)
    result = SafetySupervisor(config).step(state, (0., 1.), .05)
    assert 0. < result.target_speed_mps < .15
    assert 0. < result.action[1] <= result.target_speed_mps / (3. * .05)


@pytest.mark.parametrize("dt", [.05, .1])
@pytest.mark.parametrize("initial_speed", [0., 8.])
@pytest.mark.parametrize("brake_deceleration", [1.5, 10.])
def test_heavy_rain_speed_loop_converges_without_repeated_brake_pulses(dt, initial_speed, brake_deceleration):
    supervisor = SafetySupervisor()
    state = DrivingState(speed_mps=initial_speed, speed_limit_mps=8.333333333333334, rain=1., wetness=.9)
    target = compute_target_speed(state)
    settled_actions = []
    for step in range(int(20. / dt)):
        result = supervisor.step(state, (0., 1.), dt)
        acceleration = 3. if result.action[1] >= 0. else brake_deceleration
        state.speed_mps = max(0., state.speed_mps + result.action[1] * acceleration * dt)
        if step * dt > 10.:
            settled_actions.append(result.action[1])
            assert abs(state.speed_mps - target) < .01
    assert abs(state.speed_mps - target) < .001
    assert max(abs(action) for action in settled_actions) < .01

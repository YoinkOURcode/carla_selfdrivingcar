"""Signal approaches with varying brake response and delayed actuation."""

from dataclasses import replace

import numpy as np
import pytest

from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingState, SafetySupervisor


def signal_state(signal, speed, distance, position=0., limit=8.):
    return DrivingState(speed_mps=speed, speed_limit_mps=limit,
                        route_progress_m=position, yellow_light_confidence=1.,
                        **{signal + "_light": True, signal + "_light_distance_m": distance})


def signal_observation(state):
    tokens = np.zeros((4, 16), dtype=np.float32)
    tokens[0, [0, 1, 2, 9, 12, 15]] = [state.speed_mps / 30., state.speed_limit_mps / 30.,
                                         1., 1., float(state.red_light), 1.]
    tokens[1:4, 0], tokens[1:4, 15] = [.05, .15, .3], -1.
    if state.red_light:
        tokens[2, 3:5] = [state.red_light_distance_m / 100., 1.]
    if state.yellow_light:
        tokens[2, 5:8] = [1., state.yellow_light_distance_m / 100., 1.]
    return {"tokens": tokens, "mask": np.ones(4, dtype=bool)}


@pytest.mark.parametrize("signal", ["red", "yellow"])
@pytest.mark.parametrize("speed,distance", [
    (1.57219914898772, 2.9304215258973407),
    (2.2314595470049996, 3.935426413424562),
    (2.073717500758052, 3.7194992759000414),
])
def test_run12_near_line_approach_brakes_progressively(signal, speed, distance):
    # run12 ep2502/2507 used the comfort buffer in the full-brake threshold.
    # The stop-line emergency guard permits proportional correction here.
    controller, supervisor = RouteController(), SafetySupervisor()
    state = signal_state(signal, speed, distance, 20. - distance)
    carry = {"yellow_stop": True} if signal == "yellow" else None
    if signal == "yellow":
        supervisor.step(signal_state(signal, 3., 20.), (0., 1.), .05)
    action, _ = controller.act(signal_observation(state), carry)
    result = supervisor.step(state, (0., .049), .05)
    assert -1. < result.action[1] < 0.
    assert -1. < action[1] < 0.
    assert result.action[1] == pytest.approx(.5 * (result.target_speed_mps - speed))
    assert supervisor.step(state, (0., -.95), .05).action[1] == -.95
    for hazard in ({"front_hazard_m": 1.}, {"pedestrian_conflict": True}, {"crossing_occupied": True}):
        assert supervisor.step(replace(state, **hazard), (0., 1.), .05).action[1] == -1.


@pytest.mark.parametrize("signal", ["red", "yellow", "yellow_to_red"])
@pytest.mark.parametrize("use_controller", [False, True])
@pytest.mark.parametrize("initial_speed,line", [(3., 20.), (8., 40.), (14., 100.)])
@pytest.mark.parametrize("brake_deceleration", [1.5, 6., 10., 20.])
@pytest.mark.parametrize("dt", [.05, .1])
@pytest.mark.parametrize("delay_ticks", [0, 1])
def test_signal_first_stop_reaches_wait_area_with_delayed_brakes(
        signal, use_controller, initial_speed, line, brake_deceleration, dt, delay_ticks):
    controller, supervisor = RouteController(), SafetySupervisor()
    speed, position, previous_action, carry = initial_speed, 0., 0., None
    first_stop_distance = None
    for step in range(int(60. / dt)):
        phase = ("red" if step * dt >= 2. else "yellow") if signal == "yellow_to_red" else signal
        state = signal_state(phase, speed, line - position, position, initial_speed)
        requested = (0., 1.)
        if use_controller:
            requested, carry = controller.act(signal_observation(state), carry)
        result = supervisor.step(state, requested, dt)
        applied = previous_action if delay_ticks else result.action[1]
        previous_action = result.action[1]
        acceleration = 3. if applied >= 0. else brake_deceleration
        next_speed = max(0., speed + applied * acceleration * dt)
        position += .5 * (speed + next_speed) * dt
        speed = next_speed
        assert position < line
        if first_stop_distance is None and speed < supervisor.config.stop_speed_mps:
            first_stop_distance = line - position
    assert first_stop_distance is not None and first_stop_distance <= 2.5
    assert speed == 0.
    assert 1.5 <= line - position <= 2.5
    assert result.action[1] == -1. and result.target_speed_mps == 0.
    green = DrivingState(speed_limit_mps=initial_speed, route_progress_m=position)
    assert supervisor.step(green, (0., 1.), dt).action[1] > 0.


@pytest.mark.parametrize("signal", ["red", "yellow"])
def test_signal_imminent_crossing_full_brake_and_green_release(signal):
    supervisor = SafetySupervisor()
    # Commit to stopping while yellow still has enough distance.
    supervisor.step(signal_state(signal, 4., 30.), (0., 1.), .1)
    state = signal_state(signal, 2., 1.5, 28.5)
    result = supervisor.step(state, (0., 1.), .1)
    assert result.action[1] == -1.
    stopped = replace(state, speed_mps=0., **{signal + "_light_distance_m": 2.4})
    assert supervisor.step(stopped, (0., 1.), .1).action[1] == -1.
    green = replace(stopped, red_light=False, yellow_light=False)
    assert supervisor.step(green, (0., 1.), .1).action[1] > 0.


@pytest.mark.parametrize("signal", ["red", "yellow"])
@pytest.mark.parametrize("speed", [3., 5., 8.])
def test_no_forced_brake_outside_stop_area(signal, speed):
    # Full brake is reserved for the 2.5 m stop area; farther out the
    # speed-target feedback brakes proportionally.
    supervisor = SafetySupervisor()
    supervisor.step(signal_state(signal, speed, 60.), (0., 1.), .05)
    for distance in (2.6, 3.5, 5., 10.):
        state = signal_state(signal, speed, distance, 60. - distance)
        result = supervisor.step(state, (0., 1.), .05)
        assert signal + "_light_stop" not in result.intervention_reasons
        assert result.action[1] == pytest.approx(max(-1., .5 * (result.target_speed_mps - speed)))

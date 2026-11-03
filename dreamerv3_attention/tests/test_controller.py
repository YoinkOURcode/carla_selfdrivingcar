import numpy as np
import pytest

from dreamerv3_attention.controller import RouteController
from dreamerv3_attention.driving import DrivingConfig, DrivingState, SafetySupervisor, compute_target_speed
from dreamerv3_attention.navigation import Route


def observation(speed=5.0, limit=8.333, confidence=1.0, rain=0.0, visibility=100.0,
                risk=0.0, wetness=0.0, night=0.0, child_risk=0.0, crossing_distance=100.0,
                occupied=0.0, red=0.0, red_distance=None):
    tokens = np.zeros((5, 16), dtype=np.float32)
    tokens[0, [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] = [speed/30, limit/30, confidence,
        rain, wetness, night, visibility/100, crossing_distance/100, occupied, red, risk, child_risk, 1]
    tokens[1, :2], tokens[1, 15] = [.05, .0067], -1
    tokens[2, :2], tokens[2, 15] = [.15, .02], -1
    if red_distance is not None:
        tokens[2, 3:5] = [red_distance / 100., 1.]
    return {"tokens": tokens, "mask": np.ones(5, dtype=bool)}


def route_observation(route, position, yaw=0., speed=4.):
    obs = observation(speed=speed)
    progress, _, _ = route.project(position)
    rotation = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    for row, ahead in enumerate((5., 15., 30.), 1):
        distance = min(route.length, progress + ahead)
        index = min(np.searchsorted(route.cumulative, distance, side="right") - 1, len(route.delta) - 1)
        point = route.points[index] + (distance-route.cumulative[index])/route.lengths[index]*route.delta[index]
        obs["tokens"][row, :2] = rotation @ (point - position) / 100.
        obs["tokens"][row, 15] = -1.
    return obs


@pytest.mark.parametrize("direction", [-1., 1.])
def test_controller_stays_straight_until_near_corner(direction):
    route = Route([(0., 0.), (20., 0.), (20., direction*30.)])
    controller = RouteController()
    for x in (0., 5., 10., 14.):
        action, _ = controller.act(route_observation(route, (x, 0.)), None)
        assert abs(action[0]) < 1e-6
    action, _ = controller.act(route_observation(route, (18., 0.)), None)
    assert direction*action[0] > .1


@pytest.mark.parametrize("direction", [-1., 1.])
@pytest.mark.parametrize("speed", [3., 5., 8.])
def test_controller_tracks_corner_in_offline_bicycle_plant(direction, speed):
    # Kinematic geometry check only; CARLA tyre response is not modelled.
    route = Route([(0., 0.), (30., 0.), (30., direction*50.)])
    controller = RouteController()
    position, yaw, max_error = np.zeros(2), 0., 0.
    for _ in range(1000):
        action, _ = controller.act(route_observation(route, position, yaw, speed), None)
        position += speed*.05*np.array([np.cos(yaw), np.sin(yaw)])
        yaw += speed/2.8*np.tan(float(action[0])*.7)*.05
        progress, error, _ = route.project(position)
        max_error = max(max_error, abs(error))
        if progress > 75.:
            break
    assert progress > 75.
    assert max_error < 1.3


@pytest.mark.parametrize("failure", ["masked", "nonfinite", "wrong_type"])
def test_invalid_near_navigation_fails_safe(failure):
    obs = observation()
    if failure == "masked":
        obs["mask"][1] = False
    elif failure == "nonfinite":
        obs["tokens"][1, 0] = np.nan
    else:
        obs["tokens"][1, 15] = 0.
    np.testing.assert_array_equal(RouteController().act(obs, None)[0], [0., -1.])


def test_speed_limit_and_fallback_and_cap():
    controller = RouteController()
    assert controller.act(observation(), None)[0][1] > 0
    low = controller.act(observation(speed=5, limit=4), None)[0][1]
    assert low < 0
    custom = RouteController(config=DrivingConfig(default_speed_mps=6, min_speed_limit_confidence=.8))
    assert custom.act(observation(speed=5, limit=8, confidence=.2), None)[0][1] > 0
    assert RouteController(config=DrivingConfig(default_speed_mps=4)).act(
        observation(speed=5, limit=0), None)[0][1] < 0
    assert RouteController(config=DrivingConfig(default_speed_mps=6, min_speed_limit_confidence=.8)).act(
        observation(speed=5, limit=8, confidence=.7), None)[0][1] > 0
    capped = RouteController(4).act(observation(speed=5, limit=8), None)[0][1]
    assert capped < 0
    for cap in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            RouteController(cap)


def test_weather_visibility_and_risk_match_shared_target():
    controller = RouteController()
    for obs, kwargs in ((observation(rain=.5), {"rain": .5}),
                        (observation(wetness=.5), {"wetness": .5}),
                        (observation(night=.5), {"night": .5}),
                        (observation(visibility=20), {"visibility_m": 20}),
                        (observation(risk=.5), {"pedestrian_risk": .5}),
                        (observation(child_risk=.5), {"child_pedestrian_risk": .5})):
        state = DrivingState(speed_mps=1, speed_limit_mps=8.333, **kwargs)
        target = compute_target_speed(state, controller.config)
        action = controller.act(obs, None)[0][1]
        np.testing.assert_allclose(action, np.clip(.5 * (target - 5), -1, 1), atol=1e-5)


def test_nonfinite_context_or_navigation_fails_safe():
    for idx in (0, 1, 2, 6, 7, 8, 9, 11, 12, 13, 14, 15):
        obs = observation()
        obs["tokens"][0, idx] = np.nan
        np.testing.assert_array_equal(RouteController().act(obs, None)[0], [0, -1])
    obs = observation()
    obs["tokens"][2, 0] = np.inf
    np.testing.assert_array_equal(RouteController().act(obs, None)[0], [0, -1])
    obs = observation()
    obs["mask"] = np.asarray([1.0])
    np.testing.assert_array_equal(RouteController().act(obs, None)[0], [0, -1])


def requested_and_applied(controller, supervisor, speed, *, red=False, front=None,
                           crossing_id=None, crossing_distance=None, occupied=False, limit=8.333, dt=.1):
    obs = observation(speed=speed, limit=limit, red=float(red), occupied=float(occupied),
                      crossing_distance=100 if crossing_distance is None else crossing_distance)
    requested = controller.act(obs, None)[0]
    state = DrivingState(speed_mps=speed, speed_limit_mps=limit,
        red_light=red, red_light_distance_m=1 if red else None,
        front_hazard_m=front, crossing_id=crossing_id,
        crossing_distance_m=crossing_distance, crossing_occupied=occupied)
    return requested, supervisor.step(state, requested, dt)


def test_controller_and_supervisor_hold_hazards_and_release_crosswalk():
    controller, supervisor = RouteController(), SafetySupervisor()
    _, red = requested_and_applied(controller, supervisor, 0, red=True)
    assert red.action[1] < 0 and "red_light_stop" in red.intervention_reasons
    _, hazard = requested_and_applied(controller, supervisor, 0, front=1.5)
    assert hazard.action[1] < 0 and "front_hazard_brake" in hazard.intervention_reasons
    _, occupied = requested_and_applied(controller, supervisor, 0, crossing_id="crossing",
        crossing_distance=2, occupied=True)
    assert occupied.action[1] < 0
    result = occupied
    for _ in range(30):
        _, result = requested_and_applied(controller, supervisor, 0, crossing_id="crossing",
            crossing_distance=2, occupied=False)
    assert result.crossing_phase in ("resume", "passed")
    _, released = requested_and_applied(controller, supervisor, 0, crossing_id="crossing",
        crossing_distance=2, occupied=False)
    assert released.action[1] > 0
    assert released.crossing_phase in ("passed", "clear")


def test_offline_longitudinal_fake_plant_tracks_limit_brakes_and_restarts():
    # Simplified offline plant: signed action maps to acceleration with mild drag.
    controller, supervisor = RouteController(), SafetySupervisor()
    speed = 0.0
    trace = []
    for _ in range(400):
        requested, safety = requested_and_applied(controller, supervisor, speed, dt=.05)
        speed = max(0.0, speed + float(safety.action[1]) * 1.5 * .05 - .01 * speed * .05)
        trace.append(speed)
    assert trace[5] > 0
    assert max(trace[-50:]) >= 8.333 * .9
    assert max(trace) <= 8.333
    assert requested[1] <= 0.1
    high_speed = speed
    lowered = []
    for _ in range(200):
        requested, safety = requested_and_applied(controller, supervisor, speed, limit=5.0, dt=.05)
        speed = max(0.0, speed + float(safety.action[1]) * 1.5 * .05 - .01 * speed * .05)
        lowered.append(speed)
    assert min(lowered) < high_speed - 1.0
    assert lowered[-1] <= 5.5
    requested, safety = requested_and_applied(controller, supervisor, speed, crossing_id="x",
        crossing_distance=2, dt=.05)
    assert safety.action[1] < 0
    speed = 0.0
    for _ in range(30):
        requested, safety = requested_and_applied(controller, supervisor, speed,
            crossing_id="x", crossing_distance=2, dt=.1)
    assert safety.crossing_phase in ("resume", "passed")
    requested, safety = requested_and_applied(controller, supervisor, speed,
        crossing_id="x", crossing_distance=2, dt=.1)
    assert safety.action[1] > 0


@pytest.mark.parametrize("initial_speed,line", [(3., 20.), (8., 40.), (14., 100.)])
@pytest.mark.parametrize("dt", [.05, .1])
def test_controller_supervisor_trajectory_stops_holds_and_releases_red(initial_speed, line, dt):
    controller, supervisor = RouteController(), SafetySupervisor()
    speed, position = initial_speed, 0.
    stopped_position = None
    for _ in range(int(40./dt)):
        distance = line-position
        # The detector has a minimum 30 m lookahead and extends at higher speed.
        predicted = speed + supervisor.config.max_acceleration_mps2*dt
        lookahead = max(30., 2. + predicted**2/3. + speed*dt + 10.)
        red = distance <= lookahead
        state = DrivingState(speed_mps=speed, speed_limit_mps=initial_speed,
                             route_progress_m=position, red_light=red,
                             red_light_distance_m=distance if red else None)
        request = controller.act(observation(speed=speed, limit=initial_speed, red=float(red),
                                             red_distance=distance if red else None), None)[0]
        result = supervisor.step(state, request, dt)
        # Full brake gives only 1.5 m/s²; integrate average speed through a tick.
        next_speed = max(0., speed + float(result.action[1])*1.5*dt)
        position += .5*(speed+next_speed)*dt
        speed = next_speed
        assert position < line
        if speed == 0.:
            if stopped_position is None:
                stopped_position = position
            assert result.action[1] == -1.
            assert position == stopped_position
    assert stopped_position is not None
    assert 1.5 < line-position < 7.  # Stops near the buffer, without creeping.
    released = supervisor.step(DrivingState(speed_mps=0., speed_limit_mps=initial_speed,
        route_progress_m=position), controller.act(observation(speed=0., limit=initial_speed), None)[0], dt)
    assert released.action[1] > 0.


def test_controller_uses_red_light_distance():
    controller = RouteController()
    near = observation(speed=4., red=1., red_distance=2.)
    far = observation(speed=4., red=1., red_distance=30.)
    assert controller.act(near, None)[0][1] < 0.
    assert controller.act(far, None)[0][1] > 0.


def test_controller_slows_for_yellow_and_holds_only_near_stop_buffer():
    controller = RouteController()
    obs = observation(speed=6., limit=10.)
    obs["tokens"][2, 5:8] = [1., .30, .8]
    action, carry = controller.act(obs, None)
    assert action[1] < 0.
    obs = observation(speed=3., limit=10.)
    obs["tokens"][2, 5:8] = [1., .04, .8]
    action, carry = controller.act(obs, carry)
    assert action[1] == -1.
    obs = observation(speed=0., limit=10.)
    obs["tokens"][2, 5:8] = [1., .04, .8]
    action, carry = controller.act(obs, carry)
    assert action[1] > 0.  # A stop 4 m away is outside the accepted area.
    obs["tokens"][2, 6] = .02
    action, carry = controller.act(obs, carry)
    assert action[1] == -1.
    assert controller.act(observation(speed=0., limit=10.), carry)[0][1] > 0.


def test_yellow_controller_and_supervisor_release_recorded_far_stop():
    controller, supervisor = RouteController(), SafetySupervisor()
    carry = None
    for speed, distance in [(8.3, 26.), (0., 18.85)]:
        obs = observation(speed=speed, limit=10.)
        obs["tokens"][2, 5:8] = [1., distance / 100., 1.]
        action, carry = controller.act(obs, carry)
        state = DrivingState(speed_mps=speed, speed_limit_mps=10., route_progress_m=26. - distance,
                             yellow_light=True, yellow_light_distance_m=distance, yellow_light_confidence=1.)
        result = supervisor.step(state, action, .05)
        if speed:
            assert action[1] == -1. and result.action[1] == -1.
        else:
            assert action[1] > 0. and result.action[1] > 0.
            assert result.target_speed_mps > 0.
        assert carry["yellow_stop"] is True


@pytest.mark.parametrize("dt", [.05, .1])
@pytest.mark.parametrize("brake_deceleration", [1.5, 6., 10.])
@pytest.mark.parametrize("red_after_s", [None, 2.])
def test_yellow_controller_supervisor_trajectory_with_persistent_carry(dt, brake_deceleration, red_after_s):
    controller, supervisor = RouteController(), SafetySupervisor()
    speed, position, line = 8., 0., 40.
    carry = None
    for step in range(int(60. / dt)):
        distance = line - position
        red = red_after_s is not None and step * dt >= red_after_s
        obs = observation(speed=speed, limit=8., red=float(red), red_distance=distance if red else None)
        fields = {"red_light": True, "red_light_distance_m": distance}
        if not red:
            obs["tokens"][2, 5:8] = [1., distance / 100., 1.]
            fields = {"yellow_light": True, "yellow_light_distance_m": distance, "yellow_light_confidence": 1.}
        action, carry = controller.act(obs, carry)
        state = DrivingState(speed_mps=speed, speed_limit_mps=8., route_progress_m=position, **fields)
        result = supervisor.step(state, action, dt)
        acceleration = (supervisor.config.max_acceleration_mps2 if result.action[1] >= 0.
                        else brake_deceleration)
        next_speed = max(0., speed + result.action[1] * acceleration * dt)
        position += .5 * (speed + next_speed) * dt
        speed = next_speed
        assert position < line
    assert speed == 0.
    assert 1.5 <= line - position <= 2.5
    assert result.action[1] == -1. and result.target_speed_mps == 0.
    action, _ = controller.act(observation(speed=0., limit=8.), carry)
    assert supervisor.step(DrivingState(speed_limit_mps=8., route_progress_m=position), action, dt).action[1] > 0.


def test_controller_latches_late_yellow_decision():
    controller = RouteController()
    obs = observation(speed=8., limit=10.)
    obs["tokens"][2, 5:8] = [1., .03, .8]
    action, carry = controller.act(obs, None)
    assert action[1] > 0.
    obs = observation(speed=0., limit=10.)
    obs["tokens"][2, 5:8] = [1., .03, .8]
    action, _ = controller.act(obs, carry)
    assert action[1] > 0.


def test_child_speed_cap_persists_briefly_after_risk_flag_clears():
    supervisor = SafetySupervisor()
    base = dict(speed_mps=1.0, speed_limit_mps=8.0, speed_limit_confidence=1.0, rain=.3, night=1.)
    capped = supervisor.step(DrivingState(**base, pedestrian_risk=1., child_pedestrian_risk=1.), (0., 1.), .05)
    # The flag clears at the window edge: the cap must not jump up immediately.
    held = supervisor.step(DrivingState(**base), (0., 1.), .05)
    assert held.target_speed_mps == capped.target_speed_mps < 2.
    for _ in range(40):
        released = supervisor.step(DrivingState(**base), (0., 1.), .05)
    assert released.target_speed_mps > 4.
    supervisor.step(DrivingState(**base, pedestrian_risk=1., child_pedestrian_risk=1.), (0., 1.), .05)
    supervisor.reset()
    assert supervisor.step(DrivingState(**base), (0., 1.), .05).target_speed_mps > 4.


def test_creep_lets_a_stopped_car_leave_a_static_hazard_but_not_a_vehicle_or_pedestrian():
    base = dict(speed_mps=0., speed_limit_mps=8., speed_limit_confidence=1., front_hazard_m=2.5)
    off, on = SafetySupervisor(), SafetySupervisor(DrivingConfig(creep_speed_mps=1.))
    blocked = off.step(DrivingState(**base), (.5, 1.), .05)
    assert blocked.target_speed_mps == 0. and "front_hazard_brake" in blocked.intervention_reasons
    wall = on.step(DrivingState(**base), (.5, 1.), .05)
    assert wall.target_speed_mps == 1. and wall.action[1] > 0
    assert "front_hazard_brake" not in wall.intervention_reasons
    # Too close, already faster than the creep, a vehicle, or a pedestrian: unchanged.
    for override in (dict(front_hazard_m=1.0), dict(speed_mps=2.), dict(vehicle_ahead=True),
                     dict(pedestrian_conflict=True, pedestrian_risk=1.)):
        result = SafetySupervisor(DrivingConfig(creep_speed_mps=1.)).step(
            DrivingState(**{**base, **override}), (.5, 1.), .05)
        assert result.action[1] < 0, override

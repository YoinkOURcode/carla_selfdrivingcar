"""Small, deterministic safety supervisor and driving reward helpers.

Distances are metres, speed is metres/second, angles are radians, and actions
are normalized to [-1, 1]. Longitudinal action is positive for throttle and
negative for brake. Crossing distance is supplied route-filtered by the caller.
"""

from dataclasses import dataclass, field, replace
import copy
import math
from typing import Dict, Optional, Tuple


WAYPOINT_SPACING_M = 15.0
MIN_EPISODE_RETURN = -500.0
LANE_CENTER_TOLERANCE_M = 0.5
SIGNAL_STOP_AREA_M = 2.5


@dataclass
class DrivingState:
    speed_mps: float = 0.0
    speed_limit_mps: Optional[float] = None
    speed_limit_confidence: float = 1.0
    lane_error_m: float = 0.0
    lane_width_m: float = 3.5
    heading_error_rad: float = 0.0
    route_progress_m: float = 0.0
    front_hazard_m: Optional[float] = None
    vehicle_ahead: bool = False
    side_hazard_m: Optional[float] = None
    rear_hazard_m: Optional[float] = None
    pedestrian_risk: float = 0.0
    child_pedestrian_risk: float = 0.0
    rain: float = 0.0
    visibility_m: float = 100.0
    wetness: float = 0.0
    night: float = 0.0
    crossing_id: Optional[str] = None
    crossing_distance_m: Optional[float] = None
    crossing_occupied: bool = False
    pedestrian_conflict: bool = False
    red_light_distance_m: Optional[float] = None
    red_light: bool = False
    traffic_light_id: Optional[str] = None
    traffic_light_phase: Optional[str] = None
    traffic_light_distance_m: Optional[float] = None
    yellow_light: bool = False
    yellow_light_distance_m: Optional[float] = None
    yellow_light_confidence: float = 0.0
    red_light_queue: bool = False
    observations_valid: bool = True
    lane_departure: bool = False
    offroad: bool = False
    collision: bool = False
    pedestrian_collision: bool = False
    red_light_violation: bool = False
    crossing_violation: bool = False
    left_hazard_m: Optional[float] = None
    right_hazard_m: Optional[float] = None


@dataclass
class DrivingConfig:
    default_speed_mps: float = 3.0
    max_speed_mps: float = 0.0
    min_speed_limit_confidence: float = 0.6
    following_time_s: float = 2.0
    following_min_gap_m: float = 4.0
    crossing_buffer_m: float = 2.0
    stop_dwell_s: float = 0.5
    clear_dwell_s: float = 0.75
    stop_speed_mps: float = 0.2
    max_steer_when_side_hazard: float = 0.25
    max_acceleration_mps2: float = 3.0
    red_light_deceleration_mps2: float = 1.5
    restart_assist_enabled: bool = False
    restart_stop_distance_m: float = 6.0
    restart_stop_confirm_s: float = 0.5
    restart_green_grace_s: float = 0.5
    restart_target_speed_mps: float = 1.5
    restart_max_throttle: float = 0.4
    restart_timeout_s: float = 5.0
    restart_handover_s: float = 0.5
    pedestrian_speed_factor: float = 0.8
    child_pedestrian_speed_factor: float = 0.15
    # Keep the pedestrian speed cap this long after the risk flag clears, so a
    # flag flickering at the edge of its window cannot launch and re-brake.
    pedestrian_risk_hold_s: float = 1.5
    vehicle_speed_factor: float = 0.8
    # Let the car creep away from a static obstacle (not a vehicle) that is at
    # least creep_min_gap_m ahead, instead of holding it stopped at the
    # following gap. Without this a car facing a wall inside the gap can never
    # turn out. 0 disables it.
    creep_speed_mps: float = 0.0
    creep_min_gap_m: float = 1.5
    # The action space has no reverse, so a car stopped facing a wall inside the
    # following gap can never turn out (the steering does nothing at 0 m/s).
    # After reverse_after_s stopped in front of a static hazard, back up in a
    # straight line until the gap reaches reverse_clear_gap_m. 0 disables it.
    reverse_after_s: float = 0.0
    reverse_throttle: float = 0.4
    reverse_max_s: float = 3.0
    reverse_clear_gap_m: float = 6.0
    reverse_min_rear_m: float = 1.5
    reverse_max_attempts: int = 5
    # Backing up flips the steering effect, so positive steer reduces a positive
    # heading error. Steer by the heading error (capped) so the car turns toward
    # the route instead of reversing straight back along a wrong heading.
    reverse_max_steer: float = 0.5


@dataclass
class SafetyResult:
    action: Tuple[float, float]
    target_speed_mps: float
    intervention_reasons: Tuple[str, ...] = ()
    crossing_phase: str = "clear"
    restart_assist_phase: str = "disabled"
    restart_assist_event: Optional[str] = None
    restart_assist_elapsed_s: float = 0.0
    reverse: bool = False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _valid_nonnegative(value: Optional[float]) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            return None
    except (TypeError, ValueError):
        return None
    return value


def action_to_controls(action: Tuple[float, float]) -> Tuple[float, float, float]:
    """Convert (steer, signed longitudinal) into (steer, throttle, brake)."""
    if (len(action) != 2 or not all(math.isfinite(float(value)) for value in action)):
        return 0.0, 0.0, 1.0
    steer = _clamp(float(action[0]), -1.0, 1.0)
    longitudinal = _clamp(float(action[1]), -1.0, 1.0)
    return steer, max(0.0, longitudinal), max(0.0, -longitudinal)


def yellow_can_stop(state: DrivingState, config: DrivingConfig, dt_s: float = 0.1) -> bool:
    """Decide whether a yellow signal leaves enough distance for a controlled stop."""
    distance = _valid_nonnegative(state.yellow_light_distance_m)
    if distance is None:
        return True  # Unknown distance: do not accelerate into an identified signal.
    speed = max(0., state.speed_mps)
    reaction = max(0.1, dt_s)
    return distance >= (config.crossing_buffer_m + speed * reaction +
                        speed**2 / (2. * config.red_light_deceleration_mps2))


def signal_wait(state: DrivingState, config: DrivingConfig = DrivingConfig()) -> bool:
    """Accept a stationary signal stop at the line, crossing or connected queue."""
    if (not math.isfinite(state.speed_mps) or
            not 0.0 <= state.speed_mps < config.stop_speed_mps or
            not (state.red_light or state.yellow_light)):
        return False
    distance = _valid_nonnegative(state.red_light_distance_m if state.red_light
                                  else state.yellow_light_distance_m)
    crossing = _valid_nonnegative(state.crossing_distance_m)
    return bool(state.red_light_queue or
                (distance is not None and (distance <= SIGNAL_STOP_AREA_M or
                 (crossing is not None and crossing <= 5.0 and
                  0.0 <= distance - crossing <= 15.0))))


def signal_target_speed(distance: Optional[float], config: DrivingConfig,
                        dt_s: float = 0.1) -> float:
    """Approach the comfort buffer with gradual speed feedback."""
    distance = _valid_nonnegative(distance)
    if distance is None:
        return 0.0
    remaining = max(0.0, distance - config.crossing_buffer_m)
    deceleration = config.red_light_deceleration_mps2
    # For feedback gain .5 and the minimum assumed brake response, this
    # distance/time cap gives a critically damped approach. Two ticks allow
    # for delayed control. A low creep target reaches the hold area.
    approach_time = 4.0 / (0.5 * deceleration) + 2.0 * dt_s
    return min(math.sqrt(2.0 * deceleration * remaining),
               max(2.0 * config.stop_speed_mps, remaining / approach_time))


def signal_stop_required(state: DrivingState, distance: Optional[float],
                         config: DrivingConfig, dt_s: float = 0.1) -> bool:
    """Reserve full brake for holding or an imminent stop-line crossing."""
    distance = _valid_nonnegative(distance)
    if distance is None or signal_wait(state, config):
        return True
    speed = max(0.0, state.speed_mps)
    deceleration = config.red_light_deceleration_mps2
    hold_distance = (config.crossing_buffer_m + config.stop_speed_mps * dt_s +
                     (config.stop_speed_mps + config.max_acceleration_mps2 * dt_s)**2 /
                     (2.0 * deceleration))
    # The comfort buffer belongs to the gradual approach target. Including it
    # here causes full-brake pulses metres before the accepted waiting area.
    emergency_distance = (speed * max(0.1, dt_s) +
                          (speed + config.max_acceleration_mps2 * dt_s)**2 /
                          (2.0 * deceleration))
    # Full brake is only forced inside the stop area. Farther out, the
    # proportional speed-target feedback does the braking.
    return distance <= min(SIGNAL_STOP_AREA_M, max(hold_distance, emergency_distance))


def compute_target_speed(state: DrivingState,
                         config: DrivingConfig = DrivingConfig(), dt_s: float = 0.1) -> float:
    """Return a conservative speed target, never above a trusted limit."""
    if not state.observations_valid or not math.isfinite(state.speed_mps):
        return 0.0
    if signal_wait(state, config):
        return 0.0
    limit = state.speed_limit_mps
    trusted_limit = (limit if limit is not None and math.isfinite(limit)
                     and limit > 0 and state.speed_limit_confidence >=
                     config.min_speed_limit_confidence else None)
    target = (trusted_limit if trusted_limit is not None
              else config.default_speed_mps)
    if config.max_speed_mps > 0:
        target = min(target, config.max_speed_mps)
    base_speed = target
    # Each soft hazard caps the same base speed. Use the strongest reduction
    # once so overlapping weather and actor awareness cannot compound.
    # Child labels are only trusted when authored.
    pedestrian_risk = max(_clamp(state.pedestrian_risk, 0.0, 1.0),
                          _clamp(state.child_pedestrian_risk, 0.0, 1.0))
    child_risk = _clamp(state.child_pedestrian_risk, 0.0, 1.0)
    target = base_speed * min(
        1.0 - 0.35 * _clamp(state.rain, 0.0, 1.0),
        1.0 - 0.30 * _clamp(state.wetness, 0.0, 1.0),
        1.0 - 0.30 * _clamp(state.night, 0.0, 1.0),
        _clamp(state.visibility_m / 80.0, 0.25, 1.0),
        1.0 - (1.0 - config.pedestrian_speed_factor) * pedestrian_risk,
        1.0 - (1.0 - config.child_pedestrian_speed_factor) * child_risk,
        config.vehicle_speed_factor if state.vehicle_ahead else 1.0)
    if state.front_hazard_m is not None:
        front = _valid_nonnegative(state.front_hazard_m)
        if front is not None:
            following = max(0.0, (front - config.following_min_gap_m) /
                            max(config.following_time_s, 0.1))
            if _can_creep(state, config):
                following = max(following, config.creep_speed_mps)
            target = min(target, following)
    if state.crossing_distance_m is not None and state.crossing_distance_m >= 0:
        target = min(target, math.sqrt(max(0.0, 2.0 * 3.0 *
                                           (state.crossing_distance_m - config.crossing_buffer_m))))
    if state.red_light:
        target = min(target, signal_target_speed(state.red_light_distance_m, config, dt_s))
    if state.yellow_light and yellow_can_stop(state, config):
        target = min(target, base_speed * 0.65, max(2.0, state.speed_mps * 0.8),
                     signal_target_speed(state.yellow_light_distance_m, config, dt_s))
    if state.pedestrian_conflict or state.crossing_occupied:
        target = 0.0
    return max(0.0, target)


def _can_creep(state: DrivingState, config: DrivingConfig) -> bool:
    front = _valid_nonnegative(state.front_hazard_m)
    return (config.creep_speed_mps > 0 and front is not None and
            front >= config.creep_min_gap_m and not state.vehicle_ahead)


class SafetySupervisor:
    """Apply fail-safe actions and explicit crosswalk stop/dwell/clear phases."""

    def __init__(self, config: DrivingConfig = DrivingConfig()):
        self.config = config
        self.intervention_count = 0
        self.crossing_phase = "clear"
        self._crossing_id: Optional[str] = None
        self._stopped_s = 0.0
        self._clear_s = 0.0
        self._red_stopping = False
        self._yellow_stop: Optional[bool] = None
        self._yellow_position: Optional[float] = None
        self._yellow_braking = False
        self._restart_light_id: Optional[str] = None
        self._restart_stop_s = 0.0
        self._restart_green_s = 0.0
        self._restart_elapsed_s = 0.0
        self._restart_handover_s = 0.0
        self._restart_start_progress_m = 0.0
        self._restart_spent = False
        self._restart_phase = "disabled" if not config.restart_assist_enabled else "idle"
        self._risk_hold_s = 0.0
        self._held_risk = (0.0, 0.0)
        self._front_hold_s = self._stuck_s = self._reverse_left_s = 0.0
        self._reverse_attempts = 0

    def reset(self) -> None:
        self._risk_hold_s = 0.0
        self._held_risk = (0.0, 0.0)
        self._front_hold_s = self._stuck_s = self._reverse_left_s = 0.0
        self._reverse_attempts = 0
        self.crossing_phase = "clear"
        self._crossing_id = None
        self._stopped_s = self._clear_s = 0.0
        self._red_stopping = False
        self._yellow_stop = self._yellow_position = None
        self._yellow_braking = False
        self._restart_light_id = None
        self._restart_stop_s = self._restart_green_s = 0.0
        self._restart_elapsed_s = self._restart_handover_s = 0.0
        self._restart_start_progress_m = 0.0
        self._restart_spent = False
        self._restart_phase = "disabled" if not self.config.restart_assist_enabled else "idle"

    def longitudinal_reference(self, state: DrivingState, dt_s: float) -> float:
        """Speed feedback with the same stops, without advancing live state."""
        supervisor = copy.copy(self)
        supervisor.config = replace(self.config, restart_assist_enabled=False, reverse_after_s=0.0)
        result = supervisor.step(state, (0., 1.), dt_s)
        if result.target_speed_mps <= 0:
            return -1.
        feedback = _clamp(0.5 * (result.target_speed_mps - state.speed_mps), -1., 1.)
        return min(result.action[1], feedback)

    def step(self, state: DrivingState, requested_action: Tuple[float, float],
             dt_s: float) -> SafetyResult:
        event = None
        dt = max(0.0, dt_s if math.isfinite(dt_s) else 0.0)
        if not state.observations_valid:
            self.intervention_count += 1
            event = self._cancel_restart("invalid_observation")
            return SafetyResult((0.0, -1.0), 0.0, ("invalid_observation",), self.crossing_phase,
                                self._restart_phase, event, self._restart_elapsed_s)
        if (len(requested_action) != 2 or
                not all(math.isfinite(float(value)) for value in requested_action)):
            self.intervention_count += 1
            event = self._cancel_restart("invalid_action")
            return SafetyResult((0.0, -1.0), 0.0, ("invalid_action",), self.crossing_phase,
                                self._restart_phase, event, self._restart_elapsed_s)
        steer, longitudinal = (_clamp(float(requested_action[0]), -1, 1),
                               _clamp(float(requested_action[1]), -1, 1))
        reasons = []
        if state.pedestrian_risk > 0 or state.child_pedestrian_risk > 0:
            self._risk_hold_s = self.config.pedestrian_risk_hold_s
            self._held_risk = (state.pedestrian_risk, state.child_pedestrian_risk)
        elif self._risk_hold_s > 0:
            self._risk_hold_s -= dt
            state = replace(state, pedestrian_risk=self._held_risk[0],
                            child_pedestrian_risk=self._held_risk[1])
        if self._reverse_recovery(state, dt):
            self.intervention_count += 1
            return SafetyResult((self._reverse_steer(state), -self.config.reverse_throttle), 0.0,
                                ("reverse_recovery",),
                                self.crossing_phase, self._restart_phase, None,
                                self._restart_elapsed_s, reverse=True)

        # Side hazards constrain steering, but do not cause blind braking.
        if state.side_hazard_m is not None and 0 <= state.side_hazard_m < 3.0:
            left = _valid_nonnegative(state.left_hazard_m)
            right = _valid_nonnegative(state.right_hazard_m)
            known = [value for value in (left, right) if value is not None]
            # Unknown hazards and tight gaps constrain both directions: turning
            # can swing the rear body toward a very close obstacle.
            if (not known or state.side_hazard_m < .5 or state.side_hazard_m + 1e-6 < min(known) or
                    (state.left_hazard_m is not None and left is None) or
                    (state.right_hazard_m is not None and right is None)):
                left = right = state.side_hazard_m
            lower = -self.config.max_steer_when_side_hazard if left is not None and left < 3. else -1.
            upper = self.config.max_steer_when_side_hazard if right is not None and right < 3. else 1.
            steer = _clamp(steer, lower, upper)
            reasons.append("side_hazard_steering_limit")
        if state.crossing_id != self._crossing_id:
            self._crossing_id = state.crossing_id
            ahead = state.crossing_distance_m is not None and state.crossing_distance_m >= 0
            self.crossing_phase = "approach" if ahead else "clear"
            self._stopped_s = self._clear_s = 0.0
        distance = state.crossing_distance_m
        if distance is not None and distance >= 0 and self.crossing_phase == "clear":
            self.crossing_phase = "approach"
        stop_distance = self.config.crossing_buffer_m + max(0.0, state.speed_mps) ** 2 / 6.0
        if (self.crossing_phase == "approach" and distance is not None
                and distance <= stop_distance):
            self.crossing_phase = "stopped"
            self._stopped_s = self._clear_s = 0.0
        if self.crossing_phase == "stopped":
            longitudinal = -1.0
            reasons.append("crosswalk_stop")
            if state.speed_mps <= self.config.stop_speed_mps:
                self._stopped_s += dt
            conflict = state.crossing_occupied or state.pedestrian_conflict
            if self._stopped_s >= self.config.stop_dwell_s and not conflict:
                self._clear_s += dt
                if self._clear_s >= self.config.clear_dwell_s:
                    self.crossing_phase = "resume"
            else:
                self._clear_s = 0.0
        elif self.crossing_phase == "resume":
            self.crossing_phase = "passed"
        elif self.crossing_phase == "passed" and (distance is None or distance < 0):
            self.crossing_phase = "clear"

        if not state.yellow_light:
            self._yellow_stop = self._yellow_position = None
            self._yellow_braking = False
        else:
            yellow_distance = _valid_nonnegative(state.yellow_light_distance_m)
            yellow_position = (None if yellow_distance is None else
                               state.route_progress_m + yellow_distance)
            if (self._yellow_stop is None or
                    (yellow_position is not None and self._yellow_position is not None and
                     abs(yellow_position - self._yellow_position) > 1.0)):
                self._yellow_stop = yellow_can_stop(state, self.config, dt)
                self._yellow_braking = False
            self._yellow_position = yellow_position

        target_state = state if self._yellow_stop is not False else replace(state, yellow_light=False)
        if self.crossing_phase in {"resume", "passed"}:
            target_state = replace(target_state, crossing_id=None, crossing_distance_m=None)
        target = compute_target_speed(target_state, self.config, dt)
        if state.yellow_light and self._yellow_stop:
            yellow_limit = signal_target_speed(state.yellow_light_distance_m, self.config, dt)
            target = min(target, max(2., state.speed_mps * .8), yellow_limit)
        if self.crossing_phase == "stopped":
            target = 0.0

        longitudinal, event = self._restart_action(state, longitudinal, target, dt)
        if event and event not in {"active", "natural_restart"}:
            reasons.append("restart_assist_" + event)
        if self._restart_phase == "assist":
            reasons.append("restart_assist_active")

        if not state.red_light:
            self._red_stopping = False
        else:
            d = _valid_nonnegative(state.red_light_distance_m)
            self._red_stopping = signal_stop_required(target_state, d, self.config, dt)
            if self._red_stopping:
                if signal_wait(target_state, self.config) or d is None or d <= self.config.crossing_buffer_m:
                    target = 0.0
                longitudinal = -1.0
                reasons.append("red_light_stop")
        if state.yellow_light and self._yellow_stop:
            d = _valid_nonnegative(state.yellow_light_distance_m)
            self._yellow_braking = signal_stop_required(target_state, d, self.config, dt)
            if self._yellow_braking:
                if signal_wait(target_state, self.config) or d is None or d <= self.config.crossing_buffer_m:
                    target = 0.0
                longitudinal = -1.0
                reasons.append("yellow_light_stop")
            else:
                reasons.append("yellow_light_approach")
        creeping = _can_creep(state, self.config) and state.speed_mps <= self.config.creep_speed_mps
        if (state.front_hazard_m is not None and 0 <= state.front_hazard_m < max(3.0, state.speed_mps)
                and not creeping):
            longitudinal = min(longitudinal, -1.0 if state.front_hazard_m < 2.0 else -0.5)
            reasons.append("front_hazard_brake")
        if state.pedestrian_conflict or state.crossing_occupied:
            longitudinal = -1.0
            reasons.append("pedestrian_conflict_stop")
        speed_error = target - max(0.0, state.speed_mps)
        feedback = _clamp(0.5 * speed_error, -1.0, 1.0)
        if speed_error < 0:
            longitudinal = min(longitudinal, feedback)
            reasons.append("speed_target_brake")
        # Bound acceleration by the remaining speed margin without suppressing
        # all throttle at small positive targets. Stronger brakes retain priority.
        if longitudinal > 0:
            throttle_limit = min(max(0.0, feedback), max(0.0, speed_error) /
                                 max(self.config.max_acceleration_mps2 * dt, 1e-6))
            if longitudinal > throttle_limit:
                longitudinal = throttle_limit
                reasons.append("speed_target_throttle_limit")
        if reasons:
            self.intervention_count += 1
        return SafetyResult((steer, longitudinal), target, tuple(reasons), self.crossing_phase,
                            self._restart_phase, event, self._restart_elapsed_s)

    def _reverse_recovery(self, state: DrivingState, dt: float) -> bool:
        """True while backing away from a static hazard the car is stopped facing."""
        config = self.config
        if config.reverse_after_s <= 0:
            return False
        front = _valid_nonnegative(state.front_hazard_m)
        rear = _valid_nonnegative(state.rear_hazard_m)
        if self._reverse_left_s > 0:
            self._reverse_left_s -= dt
            if (self._reverse_left_s > 0 and not (front is not None and front >= config.reverse_clear_gap_m) and
                    not (rear is not None and rear < config.reverse_min_rear_m)):
                return True
            self._reverse_left_s = self._stuck_s = 0.0
        # The front return flickers in and out while stopped, so remember it briefly.
        self._front_hold_s = 1.0 if front is not None else max(0.0, self._front_hold_s - dt)
        waiting = (state.red_light or state.yellow_light or state.pedestrian_conflict or
                   state.crossing_occupied or state.vehicle_ahead or self.crossing_phase == "stopped" or
                   state.pedestrian_risk > 0 or state.child_pedestrian_risk > 0)
        blocked = (self._front_hold_s > 0 and state.speed_mps <= config.stop_speed_mps and not waiting)
        self._stuck_s = self._stuck_s + dt if blocked else 0.0
        if (self._stuck_s >= config.reverse_after_s and self._reverse_attempts < config.reverse_max_attempts and
                (rear is None or rear >= config.reverse_min_rear_m)):
            self._reverse_attempts += 1
            self._reverse_left_s = config.reverse_max_s
            return True
        return False

    def _reverse_steer(self, state: DrivingState) -> float:
        side = _valid_nonnegative(state.side_hazard_m)
        heading = state.heading_error_rad
        if (side is not None and side < 2.0) or not math.isfinite(heading):
            return 0.0  # the swinging rear could hit what is beside the car
        limit = self.config.reverse_max_steer
        return _clamp(heading, -limit, limit)

    def _cancel_restart(self, reason: str) -> Optional[str]:
        """Clear transient restart timing while retaining one-attempt state."""
        if self._restart_phase in {"assist", "grace", "stopped"}:
            self._restart_phase = "spent" if self._restart_spent else "idle"
            self._restart_stop_s = self._restart_green_s = 0.0
            self._restart_handover_s = 0.0
            return "cancel_" + reason
        self._restart_stop_s = self._restart_green_s = 0.0
        return None

    def _restart_action(self, state: DrivingState, longitudinal: float,
                        target: float, dt: float) -> Tuple[float, Optional[str]]:
        """Bounded speed assist after a confirmed red-stop becomes green."""
        event = None
        if not self.config.restart_assist_enabled:
            self._restart_phase = "disabled"
            return longitudinal, None

        light_id = state.traffic_light_id
        phase = state.traffic_light_phase
        distance = state.traffic_light_distance_m
        if (distance is None or isinstance(distance, bool) or
                not isinstance(distance, (int, float)) or not math.isfinite(distance)):
            distance = None
        near_stop = (distance is not None and
                     -3. * self.config.restart_stop_distance_m <= distance <=
                     self.config.restart_stop_distance_m)
        stopped = (math.isfinite(state.speed_mps) and
                   0. <= state.speed_mps <= self.config.stop_speed_mps)

        if (phase not in ("Red", "Yellow", "Green") or
                not isinstance(light_id, str) or not light_id or not near_stop):
            event = self._cancel_restart("signal_unknown")
            if self._restart_phase == "spent":
                return longitudinal, event
            if not self._restart_spent:
                self._restart_light_id = None
            self._restart_phase = "spent" if self._restart_spent else "idle"
            return longitudinal, event

        if self._restart_light_id != light_id:
            event = self._cancel_restart("signal_changed")
            self._restart_phase = "idle"
            self._restart_spent = False
            self._restart_stop_s = self._restart_green_s = 0.0
            self._restart_handover_s = 0.0
            if event is None:
                self._restart_elapsed_s = 0.0
            self._restart_light_id = light_id

        if phase in {"Red", "Yellow"}:
            if self._restart_phase in {"assist", "grace"}:
                event = self._cancel_restart("signal_changed")
            if phase == "Red" and self._restart_phase == "spent":
                self._restart_stop_s = self._restart_green_s = self._restart_handover_s = 0.0
                self._restart_phase = "idle"
            if phase == "Red" and distance is not None and distance >= 0 and stopped:
                was_confirmed = self._restart_stop_s >= self.config.restart_stop_confirm_s
                self._restart_stop_s += dt
                self._restart_phase = "stopped"
                if self._restart_stop_s >= self.config.restart_stop_confirm_s:
                    self._restart_spent = False
                    if not was_confirmed:
                        self._restart_start_progress_m = state.route_progress_m
                        self._restart_handover_s = 0.0
            else:
                self._restart_stop_s = 0.0
                self._restart_phase = "spent" if self._restart_spent else "idle"
            self._restart_green_s = 0.0
            return longitudinal, event

        if self._restart_phase == "spent":
            return longitudinal, None
        if self._restart_phase == "assist":
            pass
        elif self._restart_phase in {"stopped", "grace"} and (
                self._restart_stop_s >= self.config.restart_stop_confirm_s):
            self._restart_green_s += dt
            self._restart_phase = "grace"
            actor_maintaining = (longitudinal >= 0.1 and state.speed_mps >= 0.5 and
                                 state.route_progress_m - self._restart_start_progress_m >= 0.25)
            self._restart_handover_s = (self._restart_handover_s + dt
                                        if actor_maintaining else 0.0)
            if self._restart_handover_s >= self.config.restart_handover_s:
                self._restart_phase, self._restart_spent = "spent", True
                return longitudinal, "natural_restart"
            if self._restart_green_s < self.config.restart_green_grace_s or actor_maintaining:
                return longitudinal, None
            self._restart_phase = "assist"
            self._restart_spent = True
            self._restart_elapsed_s = self._restart_handover_s = 0.0
            self._restart_start_progress_m = state.route_progress_m
            event = "start"
        else:
            self._restart_phase = "spent" if self._restart_spent else "idle"
            return longitudinal, event

        # All required positive conditions are explicit. Unknown signal metadata
        # never enters this branch because phase and identity were checked above.
        hazards_clear = (math.isfinite(state.speed_mps) and state.speed_mps >= 0. and
                         math.isfinite(state.route_progress_m) and math.isfinite(target) and
                         not state.pedestrian_conflict and not state.crossing_occupied and
                         not state.red_light and not state.yellow_light and target > 0.05 and
                         not state.offroad and not state.lane_departure and
                         not state.collision and not state.pedestrian_collision and
                         not (state.crossing_distance_m is not None and
                              self.crossing_phase == "stopped") and
                         (state.front_hazard_m is None or
                          (math.isfinite(state.front_hazard_m) and
                           state.front_hazard_m > max(6., state.speed_mps))) and
                         math.isfinite(state.pedestrian_risk) and state.pedestrian_risk < 0.1 and
                         math.isfinite(state.child_pedestrian_risk) and state.child_pedestrian_risk < 0.1 and
                         math.isfinite(state.lane_width_m) and state.lane_width_m > 0. and
                         math.isfinite(state.lane_error_m) and
                         abs(state.lane_error_m) <= max(0.5, state.lane_width_m * 0.25) and
                         math.isfinite(state.heading_error_rad) and
                         abs(state.heading_error_rad) <= 0.35 and
                         (state.side_hazard_m is None or
                          (math.isfinite(state.side_hazard_m) and state.side_hazard_m >= 3.0)))
        if not hazards_clear:
            return longitudinal, self._cancel_restart("hazard_or_alignment")

        self._restart_elapsed_s += dt
        if self._restart_elapsed_s >= self.config.restart_timeout_s:
            self._restart_phase = "spent"
            return longitudinal, "timeout"
        assist_target = min(target, self.config.restart_target_speed_mps)
        assisted = _clamp(0.5 * (assist_target - max(0., state.speed_mps)), -0.4,
                          self.config.restart_max_throttle)
        actor_maintaining = longitudinal >= 0.1
        if (state.speed_mps >= 1.0 and actor_maintaining and
                state.route_progress_m - self._restart_start_progress_m >= 0.5):
            self._restart_handover_s += dt
            self._restart_phase = "assist" if self._restart_handover_s < self.config.restart_handover_s else "spent"
            if self._restart_phase == "spent":
                return longitudinal, "handover"
        else:
            self._restart_handover_s = 0.0
        return assisted, event or "active"


@dataclass
class RewardConfig:
    progress: float = 3.0
    lane: float = 0.2
    collision: float = 100.0
    offroad: float = 100.0
    waypoint: float = 50.0
    lane_centering: float = 1.0
    pedestrian_collision: float = 20.0
    red_light_violation: float = 8.0
    crossing_violation: float = 8.0
    speeding: float = 2.0
    target_tracking: float = 0.05
    steering_change: float = 0.01
    acceleration_change: float = 0.01
    jerk: float = 0.002
    time: float = 0.01
    idle: float = 1
    idle_speed_fraction: float = 0.5
    success: float = 300.0


@dataclass
class RewardResult:
    reward: float
    components: Dict[str, float] = field(default_factory=dict)


def compute_reward(state: DrivingState, action: Tuple[float, float],
                   next_state: DrivingState, dt_s: float = 1.0,
                   *, collision: bool = False,
                   pedestrian_collision: bool = False,
                   red_light_violation: bool = False,
                   crossing_violation: bool = False,
                   terminated: bool = False, truncated: bool = False,
                   success: bool = False,
                   previous_action: Tuple[float, float] = (0.0, 0.0),
                   previous_acceleration: float = 0.0,
                   target_speed_mps: Optional[float] = None,
                   mandated_stop: Optional[bool] = None,
                   pause_time: bool = False,
                   waypoints_reached: int = 0,
                   new_progress_m: Optional[float] = None,
                   config: RewardConfig = RewardConfig()) -> RewardResult:
    """Compute auditable rewards; route checkpoints use fixed 15 m spacing."""
    del truncated  # A time limit is not failure or success by itself.
    if (isinstance(waypoints_reached, bool) or not isinstance(waypoints_reached, int)
            or waypoints_reached < 0):
        raise ValueError("waypoints_reached must be a nonnegative integer")
    if new_progress_m is not None and (isinstance(new_progress_m, bool) or
                                        not math.isfinite(new_progress_m) or new_progress_m < 0):
        raise ValueError("new_progress_m must be finite and nonnegative")
    dt = max(0.0, dt_s)
    limit = next_state.speed_limit_mps
    if (limit is None or not math.isfinite(limit) or limit <= 0 or
            next_state.speed_limit_confidence < DrivingConfig().min_speed_limit_confidence):
        limit = config_default_speed()
    speed_violation = next_state.speed_mps > limit
    collision = collision or next_state.collision
    pedestrian_collision = pedestrian_collision or next_state.pedestrian_collision
    collision = collision or pedestrian_collision
    red_light_violation = red_light_violation or next_state.red_light_violation
    crossing_violation = crossing_violation or next_state.crossing_violation
    unsafe = (collision or pedestrian_collision or red_light_violation or
              crossing_violation or next_state.lane_departure or next_state.offroad or speed_violation)
    progress_m = next_state.route_progress_m - state.route_progress_m
    advancing_m = max(0.0, progress_m) if new_progress_m is None else new_progress_m
    lane_width = (next_state.lane_width_m if math.isfinite(next_state.lane_width_m)
                  and next_state.lane_width_m > 0 else 3.5)
    lateral_error = max(0.0, abs(next_state.lane_error_m) - LANE_CENTER_TOLERANCE_M)
    lane_center_score = max(0.0, 1.0 - lateral_error /
                            max(1e-6, lane_width / 2.0 - LANE_CENTER_TOLERANCE_M))
    valid_observations = state.observations_valid and next_state.observations_valid
    components = {
        "progress": config.progress * (0.0 if unsafe else progress_m),
        "lane": -config.lane * (lateral_error +
                                0.5 * abs(next_state.heading_error_rad)),
        "lane_departure": -config.lane * 2.0 * float(next_state.lane_departure),
        "collision": -config.collision * float(collision),
        "offroad": -config.offroad * float(next_state.offroad),
        "waypoint": config.waypoint * waypoints_reached * float(not unsafe and valid_observations),
        "lane_centering": (config.lane_centering * lane_center_score * advancing_m *
                           float(not unsafe and valid_observations)),
        "pedestrian_collision": -config.pedestrian_collision * float(pedestrian_collision),
        "red_light_violation": -config.red_light_violation * float(red_light_violation),
        "crossing_violation": -config.crossing_violation * float(crossing_violation),
        "speeding": -config.speeding * max(0.0, next_state.speed_mps - limit) * dt,
        "target_tracking": 0.0,
        "steering_change": -config.steering_change * abs(action[0] - previous_action[0]),
        "acceleration_change": -config.acceleration_change * abs(action[1] - previous_action[1]),
        "jerk": -config.jerk * abs((next_state.speed_mps - state.speed_mps) /
                                   max(dt, 1e-6) - previous_acceleration),
        "time": -config.time * dt * float(not pause_time),
        "idle": 0.0,
        "success": config.success * float(success and not unsafe and valid_observations),
    }
    safe_target = (compute_target_speed(next_state) if target_speed_mps is None
                   else max(0.0, float(target_speed_mps)))
    # Accepted signal waits also waive a pre-step approach target on arrival.
    if not unsafe and not pause_time:
        components["target_tracking"] = -config.target_tracking * abs(next_state.speed_mps - safe_target)
    stop_range = 2.0 + max(0.0, next_state.speed_mps) ** 2 / 6.0
    if mandated_stop is None:
        mandated_stop = ((next_state.crossing_id is not None and
                          next_state.crossing_distance_m is not None and
                          0.0 <= next_state.crossing_distance_m <= stop_range) or signal_wait(next_state))
    if (not unsafe and valid_observations and not mandated_stop and
            next_state.speed_mps < min(0.5, config.idle_speed_fraction * safe_target)):
        components["idle"] = -config.idle * dt
    if terminated and not success and not unsafe:
        components["termination"] = -1.0
    return RewardResult(sum(components.values()), components)


def config_default_speed() -> float:
    return DrivingConfig().default_speed_mps

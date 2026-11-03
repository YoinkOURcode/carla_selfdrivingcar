"""Simulated-time watchdogs for stalled CARLA episodes."""

from collections import deque
import math


class InactivityWatchdog:
    """Detect route stagnation without treating a required stop as failure.

    The short window runs only while movement is safe and sensing is valid.
    The independent long window runs across blocked periods. Both use net
    route progress over a rolling interval, so reversing or small oscillations
    do not keep restarting a timer.
    """

    def __init__(self, timeout_s=15., min_progress_m=.5,
                 blocked_timeout_s=60., grace_s=2.):
        self.timeout_s = self._positive_finite(timeout_s, "timeout_s")
        self.min_progress_m = self._positive_finite(min_progress_m, "min_progress_m")
        self.blocked_timeout_s = self._positive_finite(blocked_timeout_s, "blocked_timeout_s")
        if self.blocked_timeout_s < self.timeout_s:
            raise ValueError("blocked_timeout_s must be at least timeout_s")
        self.grace_s = self._finite(grace_s, "grace_s")
        if self.grace_s < 0.:
            raise ValueError("grace_s must be non-negative")
        self.reset(0.)

    @staticmethod
    def _finite(value, name):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _positive_finite(cls, value, name):
        result = cls._finite(value, name)
        if result <= 0.:
            raise ValueError(f"{name} must be positive")
        return result

    def reset(self, progress_m):
        """Start a fresh episode at the supplied route progress."""
        progress = self._finite(progress_m, "progress_m")
        self._elapsed_s = 0.
        self._tracking = self.grace_s == 0.
        self._long_history = deque([(0., progress)] if self._tracking else ())
        self._short_history = deque([(0., progress)] if self._tracking else ())
        self._short_active = self._tracking

    @staticmethod
    def _window_progress(history, now_s, window_s):
        """Return net progress across a full window, or None if too young."""
        target_s = now_s - window_s
        if len(history) < 2 or target_s < history[0][0]:
            return None
        if target_s == history[0][0]:
            return history[-1][1] - history[0][1]
        previous_t, previous_p = history[0]
        current_t, current_p = history[1]
        fraction = (target_s - previous_t) / (current_t - previous_t)
        start_p = previous_p + fraction * (current_p - previous_p)
        return history[-1][1] - start_p

    @staticmethod
    def _trim(history, cutoff_s):
        while len(history) > 2 and history[1][0] <= cutoff_s:
            history.popleft()

    def update(self, progress_m, dt_s, *, can_move):
        """Advance simulated time and return a truncation reason if stalled."""
        progress = self._finite(progress_m, "progress_m")
        dt = self._positive_finite(dt_s, "dt_s")
        self._elapsed_s += dt
        point = (self._elapsed_s, progress)

        if not self._tracking:
            if self._elapsed_s < self.grace_s:
                return None
            self._tracking = True
            self._long_history.append(point)
            self._short_history.append(point)
            self._short_active = bool(can_move)
            return None

        self._long_history.append(point)

        if can_move:
            if not self._short_active:
                self._short_history.clear()
                self._short_history.append(point)
                self._short_active = True
            else:
                self._short_history.append(point)
        else:
            self._short_history.clear()
            self._short_history.append(point)
            self._short_active = False

        cutoff = self._elapsed_s - max(self.timeout_s, self.blocked_timeout_s)
        self._trim(self._long_history, cutoff)
        self._trim(self._short_history, self._elapsed_s - self.timeout_s)

        long_progress = self._window_progress(
            self._long_history, self._elapsed_s, self.blocked_timeout_s)
        if (long_progress is not None and
                long_progress + 1e-9 < self.min_progress_m):
            return "blocked_timeout"
        if can_move and self._short_active:
            short_progress = self._window_progress(
                self._short_history, self._elapsed_s, self.timeout_s)
            if (short_progress is not None and
                    short_progress + 1e-9 < self.min_progress_m):
                return "inactivity_timeout"
        return None

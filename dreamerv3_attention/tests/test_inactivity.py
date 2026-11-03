import math

import pytest

from dreamerv3_attention.inactivity import InactivityWatchdog


def advance(watchdog, progress, seconds, *, can_move=True, dt=1.):
    reason = None
    elapsed = 0.
    while elapsed < seconds:
        step = min(dt, seconds - elapsed)
        reason = watchdog.update(progress, step, can_move=can_move)
        elapsed += step
        if reason:
            break
    return reason


def test_normal_advance_and_stationary_clear_road_timeout():
    watchdog = InactivityWatchdog(timeout_s=5., blocked_timeout_s=20., grace_s=0.)
    watchdog.reset(0.)
    for index in range(1, 6):
        assert watchdog.update(index * .1, 1., can_move=True) is None

    watchdog.reset(0.)
    assert advance(watchdog, 0., 4.9) is None
    assert watchdog.update(0., .1, can_move=True) == "inactivity_timeout"


def test_small_creep_does_not_keep_restarting_short_timer():
    watchdog = InactivityWatchdog(timeout_s=5., min_progress_m=.5,
                                  blocked_timeout_s=20., grace_s=0.)
    watchdog.reset(0.)
    assert advance(watchdog, .04, 4.9) is None
    assert watchdog.update(.04, .1, can_move=True) == "inactivity_timeout"


def test_back_and_forth_motion_uses_net_progress():
    watchdog = InactivityWatchdog(timeout_s=6., min_progress_m=.5,
                                  blocked_timeout_s=24., grace_s=0.)
    watchdog.reset(0.)
    for progress in (.3, 0., .3, 0., .3, 0.):
        reason = watchdog.update(progress, 1., can_move=True)
    assert reason == "inactivity_timeout"


def test_continuous_meaningful_progress_prevents_timeout():
    watchdog = InactivityWatchdog(timeout_s=5., min_progress_m=.5,
                                  blocked_timeout_s=20., grace_s=0.)
    watchdog.reset(0.)
    for index in range(1, 16):
        assert watchdog.update(index * .2, 1., can_move=True) is None


def test_brief_allowed_stop_grants_full_clear_road_window_after_release():
    watchdog = InactivityWatchdog(timeout_s=5., blocked_timeout_s=30., grace_s=0.)
    watchdog.reset(0.)
    assert advance(watchdog, 0., 4., can_move=True) is None
    assert advance(watchdog, 0., 2., can_move=False) is None
    assert watchdog.update(0., .1, can_move=True) is None
    assert advance(watchdog, 0., 4.9, can_move=True) is None
    assert watchdog.update(0., .1, can_move=True) == "inactivity_timeout"


def test_external_blockage_uses_independent_long_window():
    watchdog = InactivityWatchdog(timeout_s=5., blocked_timeout_s=10., grace_s=0.)
    watchdog.reset(0.)
    assert advance(watchdog, 0., 9.9, can_move=False) is None
    assert watchdog.update(0., .1, can_move=False) == "blocked_timeout"


def test_blocked_window_allows_net_progress_and_does_not_restart_on_oscillation():
    watchdog = InactivityWatchdog(timeout_s=5., min_progress_m=.5,
                                  blocked_timeout_s=10., grace_s=0.)
    watchdog.reset(0.)
    for progress in (.3, 0., .3, 0., .3, 0., .3, 0., .3, 0.):
        reason = watchdog.update(progress, 1., can_move=False)
    assert reason == "blocked_timeout"

    watchdog.reset(0.)
    for progress in (.1, .2, .3, .4, .5, .6):
        reason = watchdog.update(progress, 1., can_move=False)
    assert reason is None


def test_variable_dt_uses_simulated_elapsed_time_and_interpolates_window():
    watchdog = InactivityWatchdog(timeout_s=5., min_progress_m=.5,
                                  blocked_timeout_s=20., grace_s=0.)
    watchdog.reset(0.)
    for progress, dt in ((.1, .5), (.2, 1.5), (.3, .5), (.4, 1.5), (.4, 1.)):
        reason = watchdog.update(progress, dt, can_move=True)
    assert reason == "inactivity_timeout"


def test_startup_grace_suppresses_timeout_then_watchdog_runs():
    watchdog = InactivityWatchdog(timeout_s=2., blocked_timeout_s=10., grace_s=3.)
    watchdog.reset(0.)
    assert advance(watchdog, 0., 4.9) is None
    assert watchdog.update(0., .1, can_move=True) == "inactivity_timeout"


def test_reset_isolates_episode_time_and_route_history():
    watchdog = InactivityWatchdog(timeout_s=3., blocked_timeout_s=10., grace_s=0.)
    watchdog.reset(25.)
    assert advance(watchdog, 25., 2.9) is None
    watchdog.reset(1000.)
    assert advance(watchdog, 1000., 2.9) is None
    assert watchdog.update(1000., .1, can_move=True) == "inactivity_timeout"


def test_progress_validation_and_temporal_history_survives_small_dt():
    with pytest.raises(ValueError):
        InactivityWatchdog(timeout_s=0.)
    with pytest.raises(ValueError, match="at least"):
        InactivityWatchdog(timeout_s=10., blocked_timeout_s=9.)
    for value in (-1., math.inf, math.nan):
        with pytest.raises(ValueError):
            InactivityWatchdog(min_progress_m=value)
    with pytest.raises(ValueError):
        InactivityWatchdog(grace_s=-.1)

    watchdog = InactivityWatchdog(grace_s=0.)
    with pytest.raises(ValueError, match="dt_s"):
        watchdog.update(0., 0., can_move=True)
    with pytest.raises(ValueError, match="progress_m"):
        watchdog.update(math.nan, .1, can_move=True)

    watchdog = InactivityWatchdog(timeout_s=5., blocked_timeout_s=10., grace_s=0.)
    watchdog.reset(0.)
    for index in range(5001):
        reason = watchdog.update(index * .0002, .001, can_move=True)
        assert reason is None
    assert len(watchdog._short_history) > 4096
    assert advance(watchdog, 1.0002, 3.) == "inactivity_timeout"


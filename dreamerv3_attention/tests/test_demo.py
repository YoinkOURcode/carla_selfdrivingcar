"""Single-drive demonstration configuration and encounter scheduling."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.environment import CarlaEnv
from dreamerv3_attention.tests.test_motion_pilot import route_env, waypoints


def test_demo_config_matches_requested_checkpoint_and_scenario():
    cfg = Config.load('runs/experiment_26/curriculum/town03_demo.json').resolved()
    assert cfg.town == 'Town03'
    assert cfg.weather == 'SoftRainNoon'
    assert Path(cfg.viewer_checkpoint).name == 'latest.pt'
    assert cfg.viewer_checkpoint.endswith('stage5_run_3/latest.pt')
    assert [event['kind'] for event in cfg.demo['encounters']] == ['vehicle', 'adult', 'child']


def test_encounters_fire_once_in_order_and_finish_crossing():
    env = object.__new__(CarlaEnv)
    env.demo_pending = [{'kind': 'adult', 'trigger_m': 10}, {'kind': 'child', 'trigger_m': 20}]
    env.demo_walkers = []
    env._spawn_demo_actor = Mock()
    env._demo_encounters(9)
    env._demo_encounters(10)
    env._demo_encounters(10)
    env._demo_encounters(21)
    assert [call.args[0]['kind'] for call in env._spawn_demo_actor.call_args_list] == ['adult', 'child']
    actor = Mock(is_alive=True)
    actor.get_location.return_value = SimpleNamespace(x=9, y=0)
    direction = SimpleNamespace(x=1, y=0)
    env.carla = SimpleNamespace(WalkerControl=lambda **kwargs: kwargs)
    env.demo_walkers = [(actor, SimpleNamespace(x=0, y=0), direction, 9)]
    env._demo_encounters(22)
    actor.apply_control.assert_called_once_with({'direction': direction, 'speed': 0.})
    assert env.demo_walkers == []


def test_demo_rejects_unordered_or_unreachable_triggers():
    cfg = Config(demo={'minimum_route_length_m': 100, 'encounters': [
        {'kind': 'adult', 'trigger_m': 40}, {'kind': 'child', 'trigger_m': 39}]})
    with pytest.raises(ValueError, match='ordered route triggers'):
        cfg.validate()


def test_fixed_demo_route_does_not_search_for_other_endpoints(monkeypatch):
    env, spawns = route_env(monkeypatch)
    env.scenario = {"fixed_route": True, "requires_traffic_light": True,
                    "minimum_route_length_m": 250.}
    trace = Mock(return_value=waypoints(300.))
    env.route_planner.trace_route = trace
    env._route_geometry = Mock(return_value=(SimpleNamespace(length=300.), []))
    selected = env._select_route(spawns, 2, 4)
    assert selected[:2] == (2, 4)
    trace.assert_called_once_with(spawns[2].location, spawns[4].location)
    assert env.route_pools == {}
    env._route_geometry.return_value = (SimpleNamespace(length=100.), [])
    with pytest.raises(RuntimeError, match="selected route is shorter"):
        env._select_route(spawns, 2, 4)
    env._route_geometry.side_effect = RuntimeError("selected route has no usable traffic-light stop")
    with pytest.raises(RuntimeError, match="no usable traffic-light"):
        env._select_route(spawns, 2, 4)

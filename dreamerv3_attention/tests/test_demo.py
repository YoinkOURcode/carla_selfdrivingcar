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


def held_walker(trigger_m):
    actor = Mock(is_alive=True)
    direction = SimpleNamespace(x=1, y=0)
    return {'kind': 'adult', 'trigger_m': trigger_m, 'actor': actor,
            'origin': SimpleNamespace(x=0, y=0), 'direction': direction, 'crossing_m': 9.}


def test_encounters_release_once_in_order_and_finish_crossing():
    env = object.__new__(CarlaEnv)
    env.carla = SimpleNamespace(WalkerControl=lambda **kwargs: kwargs)
    env.cfg = Config()
    first, second = held_walker(10), held_walker(20)
    env.demo_pending, env.demo_walkers = [first, second], []
    env._demo_encounters(9)
    first['actor'].apply_control.assert_not_called()
    first['actor'].get_location.return_value = SimpleNamespace(x=0, y=0)
    second['actor'].get_location.return_value = SimpleNamespace(x=0, y=0)
    env._demo_encounters(10)
    env._demo_encounters(10)
    first['actor'].apply_control.assert_called_once_with({'direction': first['direction'], 'speed': 1.})
    second['actor'].apply_control.assert_not_called()
    env._demo_encounters(21)
    second['actor'].apply_control.assert_called_once()
    first['actor'].get_location.return_value = SimpleNamespace(x=9, y=0)
    second['actor'].get_location.return_value = SimpleNamespace(x=.5, y=0)
    env._demo_encounters(22)
    first['actor'].destroy.assert_called_once()  # crossed: removed, never left standing
    second['actor'].destroy.assert_not_called()
    assert [w[0] for w in env.demo_walkers] == [second['actor']]


def test_vehicle_stays_braked_until_trigger_then_drives_with_release_step():
    env = object.__new__(CarlaEnv)
    manager = Mock()
    manager.get_port.return_value = 8000
    env.carla = SimpleNamespace(VehicleControl=lambda **kwargs: kwargs)
    env.cfg, env.steps, env.moving, env.traffic_manager = Config(), 100, [], manager
    actor = Mock(is_alive=True)
    env.demo_pending = [{'kind': 'vehicle', 'trigger_m': 60., 'release_s': 20., 'actor': actor}]
    env.demo_walkers = []
    env._demo_encounters(59.)
    actor.set_autopilot.assert_not_called()
    env._demo_encounters(60.)
    actor.set_autopilot.assert_called_once_with(True, 8000)
    assert env.moving == [(actor, {'speed_mps': 3., 'encounter': True,
                                   'release_step': 100 + round(20. / env.cfg.dt)})]
    manager.update_vehicle_lights.assert_not_called()


def test_night_headlights_lets_traffic_manager_light_released_vehicle():
    env = object.__new__(CarlaEnv)
    manager = Mock()
    env.carla = SimpleNamespace(VehicleControl=lambda **kwargs: kwargs)
    env.cfg, env.steps, env.moving, env.traffic_manager = Config(night_headlights=True), 0, [], manager
    actor = Mock(is_alive=True)
    env.demo_pending = [{'kind': 'vehicle', 'trigger_m': 10., 'actor': actor}]
    env.demo_walkers = []
    env._demo_encounters(10.)
    manager.update_vehicle_lights.assert_called_once_with(actor, True)


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


@pytest.mark.parametrize('kind', ['vehicle', 'adult', 'child'])
@pytest.mark.parametrize('obstruction', ['junction', 'sidewalk', 'occupied'])
def test_encounter_is_preloaded_held_and_searches_beyond_unsuitable_geometry(kind, obstruction):
    import numpy as np
    from dreamerv3_attention.navigation import Route
    from dreamerv3_attention.tests.test_environment import spawn_test_environment

    if kind == 'vehicle' and obstruction == 'sidewalk':
        pytest.skip('vehicles need no sidewalk')
    actor = Mock(is_alive=True)
    env = spawn_test_environment(7, lambda _bp, _transform: actor)
    env.route = Route(np.array([[0., 0.], [300., 0.]]))
    env.scenario = {'encounters': [{'kind': kind, 'trigger_m': 0.}]}
    env.demo_pending, env.demo_walkers, env.moving = [], [], []
    env.ego.get_location = lambda: env.carla.Location()
    original = env.map.get_waypoint

    def waypoint(location, **kwargs):
        result = original(location, **kwargs)
        if location.x <= 60.:
            if obstruction == 'junction':
                result.is_junction = True
            elif obstruction == 'sidewalk' and kwargs['lane_type'] == env.carla.LaneType.Sidewalk:
                return None
        return result

    spawn_x = []
    def try_spawn(_bp, transform):
        spawn_x.append(transform.location.x)
        return None if obstruction == 'occupied' and transform.location.x <= 60. else actor

    env.map.get_waypoint = waypoint
    env.world.try_spawn_actor = Mock(side_effect=try_spawn)
    env._preload_encounters()
    assert env.actors == [actor]
    assert min(spawn_x) >= 40. and spawn_x[-1] > 60.
    assert env.demo_pending[0]['actor'] is actor
    assert env.demo_walkers == [] and env.moving == []
    if kind == 'vehicle':
        assert actor.apply_control.call_args.args[0].hand_brake
    else:
        actor.apply_control.assert_not_called()
    actor.set_autopilot.assert_not_called()


def test_unplaceable_required_encounter_fails_but_optional_is_skipped():
    import numpy as np
    from dreamerv3_attention.navigation import Route
    from dreamerv3_attention.tests.test_environment import spawn_test_environment

    env = spawn_test_environment(7, lambda _bp, _transform: None)
    env.route = Route(np.array([[0., 0.], [200., 0.]]))
    env.ego.get_location = lambda: env.carla.Location()
    env.map.get_waypoint = lambda *_args, **_kwargs: None
    env.scenario = {'encounters': [{'kind': 'adult', 'trigger_m': 0.}]}
    env.demo_pending = []
    with pytest.raises(RuntimeError, match='Required adult encounter.*could not be placed'):
        env._preload_encounters()
    env.scenario['optional_encounters'] = True
    env._preload_encounters()
    assert env.demo_pending == []


@pytest.mark.parametrize('kind', ['adult', 'child'])
def test_walker_crossing_spans_its_kerb_offset_plus_clearance(kind):
    import numpy as np
    from dreamerv3_attention.environment import WALKER_CLEAR_M
    from dreamerv3_attention.navigation import Route
    from dreamerv3_attention.tests.test_environment import spawn_test_environment

    transforms = []
    def try_spawn(_bp, transform):
        transforms.append(transform)
        return Mock(is_alive=True)

    env = spawn_test_environment(7, try_spawn)
    env.route = Route(np.array([[0., 0.], [300., 0.]]))  # along +x, so the kerb offset is |y|
    env.scenario = {'encounters': [{'kind': kind, 'trigger_m': 0.}]}
    env.demo_pending, env.demo_walkers, env.moving = [], [], []
    env.ego.get_location = lambda: env.carla.Location()
    env._preload_encounters()
    entry = env.demo_pending[0]
    assert entry['crossing_m'] == pytest.approx(abs(transforms[-1].location.y) + WALKER_CLEAR_M)


def walking_env():
    env = object.__new__(CarlaEnv)
    env.carla = SimpleNamespace(WalkerControl=lambda **kwargs: kwargs)
    env.cfg, env.demo_pending = Config(), []
    return env


def test_far_kerb_walker_keeps_walking_past_ego_lane_then_is_removed():
    # Regression (episode 6296): a walker starting 7 m from the route on the far kerb
    # stopped after a fixed 9 m, 2 m into the ego lane, and blocked the car until timeout.
    env, actor = walking_env(), Mock(is_alive=True)
    origin, direction = SimpleNamespace(x=0, y=0), SimpleNamespace(x=1, y=0)
    env.demo_walkers = [(actor, origin, direction, 7. + 4.5, 0., 0)]
    for x in (3., 6., 9.):
        actor.get_location.return_value = SimpleNamespace(x=x, y=0)
        env._demo_encounters(0.)
    actor.destroy.assert_not_called()
    actor.apply_control.assert_not_called()  # never told to stand still in the road
    actor.get_location.return_value = SimpleNamespace(x=11.5, y=0)
    env._demo_encounters(0.)
    actor.destroy.assert_called_once()
    assert env.demo_walkers == []


def test_stuck_walker_is_removed_before_ego_stationary_timeout():
    env, actor = walking_env(), Mock(is_alive=True)
    actor.get_location.return_value = SimpleNamespace(x=4., y=0)
    env.demo_walkers = [(actor, SimpleNamespace(x=0, y=0), SimpleNamespace(x=1, y=0), 11., 0., 0)]
    steps = 0
    while env.demo_walkers:
        env._demo_encounters(0.)
        steps += 1
    actor.destroy.assert_called_once()
    assert steps * env.cfg.dt <= 1.5 < env.cfg.inactivity_stationary_s


def npc_env(actor, spec):
    env = object.__new__(CarlaEnv)
    env.carla = SimpleNamespace(TrafficLightState=SimpleNamespace(Green='green', Red='red'))
    env.cfg, env.steps, env.traffic_manager = Config(), 0, Mock()
    env.moving = [(actor, spec)]
    return env


def stopped_vehicle(at_light=False, state='red'):
    actor = Mock(is_alive=True, type_id='vehicle.audi.a2')
    actor.get_speed_limit.return_value = 30.
    actor.get_velocity.return_value = SimpleNamespace(x=0., y=0., z=0.)
    actor.is_at_traffic_light.return_value = at_light
    actor.get_traffic_light_state.return_value = state
    return actor


def run_npcs(env, seconds):
    for _ in range(round(seconds / env.cfg.dt)):
        env._update_npcs()


def test_stalled_encounter_vehicle_is_removed_before_ego_stationary_timeout():
    actor = stopped_vehicle()
    env = npc_env(actor, {'speed_mps': 3., 'encounter': True})
    run_npcs(env, 4.9)
    actor.destroy.assert_not_called()
    run_npcs(env, .2)
    actor.destroy.assert_called_once()
    assert env.moving == []


def test_encounter_vehicle_waiting_at_red_and_background_traffic_are_kept():
    at_red = stopped_vehicle(at_light=True, state='red')
    env = npc_env(at_red, {'speed_mps': 3., 'encounter': True})
    run_npcs(env, 30.)
    at_red.destroy.assert_not_called()
    background = stopped_vehicle()
    env = npc_env(background, {'speed_mps': 3.})
    run_npcs(env, 30.)
    background.destroy.assert_not_called()


@pytest.mark.parametrize('length, accepted', [(390., False), (550., True), (710., False)])
def test_long_route_pool_keeps_only_routes_within_length_bounds(monkeypatch, length, accepted):
    env, spawns = route_env(monkeypatch)
    env.scenario = {'name': 'long', 'seed': 1, 'requires_traffic_light': True,
                    'minimum_route_length_m': 400., 'maximum_route_length_m': 700.}
    env.route_planner.trace_route = Mock(return_value=waypoints(length))
    env._route_geometry = Mock(return_value=(SimpleNamespace(length=length), []))
    if accepted:
        assert env._select_route(spawns, 0, 1)[3].length == length
    else:
        with pytest.raises(RuntimeError, match='no valid'):
            env._select_route(spawns, 0, 1)


@pytest.mark.parametrize('path', ['runs/experiment_28/training_configs/town01_final.json',
                                  'runs/experiment_28/evaluation/town02_final.json'])
def test_final_long_routes_are_400_to_700_m_with_encounters_that_fit(path):
    cfg = Config.load(path).resolved()
    records = {r['name']: r for r in __import__('json').load(open(cfg.scenario_file))['scenarios']}
    long = [records[name] for name in cfg.scenarios if 'long_mixed' in name]
    assert len(long) == 6
    for scenario in long:
        assert (scenario['minimum_route_length_m'], scenario['maximum_route_length_m']) == (400., 700.)
        triggers = [e['trigger_m'] for e in scenario['encounters']]
        # Actors start 40 m past their trigger and at least 5 m before the destination.
        assert triggers == sorted(triggers) and triggers[-1] + 40. <= 400. - 5.

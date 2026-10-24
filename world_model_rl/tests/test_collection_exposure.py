"""Collection exposure preserves replay labels, splits, and real actor ownership."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from world_model_rl import runner
from world_model_rl.__main__ import main
from world_model_rl.collection import CollectionOptions, ControlPerturbations, StagedHazard
from world_model_rl.control import apply_action
from world_model_rl.data import Dataset, write_json
from world_model_rl.environment import CarlaDrivingEnv
from world_model_rl.geometry import Route
from world_model_rl.observations import OracleExtractor
from world_model_rl.schema import EVENTS, OBS_KEYS, Observation
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_exposure_is_seeded_independent_and_combined():
    options = CollectionOptions(True, True)
    choices = [options.for_episode(seed) for seed in range(1000)]
    assert choices[400:] == [options.for_episode(seed) for seed in range(400, 1000)]
    assert 750 < sum(plan['exposed'] for plan in choices) < 850
    assert all(plan['perturbations'] and plan['staged_hazards'] for plan in choices)
    assert not CollectionOptions(True, True, 0).for_episode(1)['exposed']
    assert CollectionOptions(True, True, 1).for_episode(1)['exposed']
    choices[0]['disturbances'].append('independent')
    assert choices[1]['disturbances'] == []


@pytest.mark.parametrize('extra', [
    ['--exposure-fraction', '-.1'], ['--exposure-fraction', '1.1'],
    ['--exposure-fraction', 'nan'], ['--exposure-fraction', 'inf'],
    ['--perturbations', '--policy-checkpoint', 'agent.pt'],
    ['--staged-hazards', '--policy-checkpoint', 'agent.pt'],
])
def test_cli_rejects_exposure_before_creating_outputs(cfg, tmp_path, extra):
    target = tmp_path / 'new-data'
    config = tmp_path / 'config.json'
    write_json(config, cfg.to_dict())
    with pytest.raises(SystemExit) as stopped:
        main(['collect', '--config', str(config), '--data', str(target), *extra])
    assert stopped.value.code == 2
    assert not target.exists()


def test_direct_collection_checks_options_before_dataset(cfg, tmp_path):
    with pytest.raises(ValueError, match='policy-checkpoint'):
        runner.collect(cfg, tmp_path / 'data', 1, policy_checkpoint='missing.pt', perturbations=True)
    assert not (tmp_path / 'data').exists()


@pytest.mark.parametrize('perturbations,hazards', [(True, False), (False, True), (True, True)])
def test_cli_forwards_independent_options(cfg, tmp_path, monkeypatch, perturbations, hazards):
    config = tmp_path / 'config.json'
    write_json(config, cfg.to_dict())
    received = []
    monkeypatch.setattr(runner, 'collect', lambda *args, **kwargs: received.append(kwargs))
    flags = (['--perturbations'] if perturbations else []) + (['--staged-hazards'] if hazards else [])
    main(['collect', '--config', str(config), '--episodes', '2', *flags])
    assert received == [{'perturbations': perturbations, 'staged_hazards': hazards, 'exposure_fraction': .8}]


def test_disturbance_timing_recovery_and_action_limits(cfg):
    records = []
    perturb = ControlPerturbations(123, .1, records)
    baseline = np.array([.1, 0, .9], np.float32)
    original = baseline.copy()
    applied = np.zeros(3, np.float32)
    actions = []
    for step in range(1000):
        requested = perturb.action(baseline, step, 5)
        actions.append(requested)
        after = apply_action(requested, applied, cfg.environment)
        assert abs(after[0] - applied[0]) <= cfg.environment.steering_rate * .1 + 1e-6
        assert after[1] * after[2] == 0
        applied = after
    assert 30 <= records[0]['start_step'] <= 80
    assert {record['profile'] for record in records} == {'left', 'right', 'acceleration'}
    for index, record in enumerate(records):
        assert 20 <= record['duration_steps'] <= 50
        assert record['realized_steps'] <= record['duration_steps']
        assert .5 <= record['throttle'] <= 1
        if record['steering'] is not None:
            assert .35 <= abs(record['steering']) <= .9
        start, duration = record['start_step'], record['realized_steps']
        assert all(action[2] == 0 for action in actions[start:start + duration])
        if index:
            previous = records[index - 1]
            end = previous['start_step'] + previous['duration_steps']
            assert 30 <= start - end <= 80
            np.testing.assert_array_equal(actions[end:start], np.tile(baseline, (start - end, 1)))
    np.testing.assert_array_equal(baseline, original)
    repeated_records = []
    repeated = ControlPerturbations(123, .1, repeated_records)
    repeated_actions = [repeated.action(baseline, step, 5) for step in range(1000)]
    np.testing.assert_array_equal(actions, repeated_actions)
    assert repeated_records == records


def test_stationary_exposure_accelerates_instead_of_starting_steering():
    records = []
    perturb = ControlPerturbations(5, .1, records)
    for step in range(200):
        perturb.action([0, 0, 1], step, 0)
    assert records and all(record['profile'] == 'acceleration' for record in records)


class ReplayEnv:
    """Supply complete replay-compatible transitions with predetermined real endings."""

    def __init__(self, cfg, terminal=False):
        self.cfg = cfg
        self.terminal = terminal
        self.ego = None
        self.hazard_seeds = []

    def __enter__(self):
        """Mirror the collection environment context manager."""
        return self

    def __exit__(self, *args):
        """No real actors need cleanup in this replay fixture."""

    def reset(self, identity=0, seed=None, hazard_seed=None):
        """Install a new independent episode and optional staged provenance."""
        self.source = make_episode(self.cfg, identity, steps=120, terminal=self.terminal)
        self.metadata = self.source.metadata.copy()
        self.index = 0
        self.previous = np.zeros(3, np.float32)
        if hazard_seed is not None:
            self.hazard_seeds.append(hazard_seed)
            self.staged_hazard = SimpleNamespace(metadata={'kind': 'adult', 'triggered': False})
        return self.observation(), {'frame': 0, 'sim_time': 0}

    def observation(self):
        """Expose applied action history at the correct T+1 observation index."""
        obs = Observation(**{key: self.source.arrays[key][self.index].copy() for key in OBS_KEYS})
        obs.continuous[5:8] = self.previous
        return obs

    def expert_action(self, rng=None):
        """Keep baseline controls distinguishable from sustained perturbations."""
        return np.array([.1, .2, 0], np.float32)

    def finish_episode(self):
        """Mirror the event-settlement contract without synthesizing events."""

    def step(self, requested):
        """Limit real actions and end only at the fixture's predetermined boundary."""
        self.previous = apply_action(requested, self.previous, self.cfg.environment)
        self.index += 1
        ended = self.index == len(self.source)
        info = {'frame': self.index * self.cfg.environment.action_repeat,
                'sim_time': self.index * self.cfg.environment.decision_dt,
                'applied_action': self.previous.copy(), 'events': dict.fromkeys(EVENTS, 0),
                'components': {'progress': 1.},
                'terminal_reason': ('success' if self.terminal else 'stuck') if ended else 'running',
                'route_completion': 1. if ended else 0., 'distance_metres': .2,
                'lane_offset': 0., 'heading_error': 0., 'acceleration_mps2': 0.,
                'jerk_mps3': 0., 'steering_rate_per_s': 0.}
        return self.observation(), 1., ended and self.terminal, ended and not self.terminal, info


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize('perturbations,hazards', [(False, False), (True, False), (False, True), (True, True)])
def test_exposure_preserves_labels_and_action_alignment(cfg, terminal, perturbations, hazards):
    env = ReplayEnv(cfg, terminal)
    exposure = CollectionOptions(perturbations, hazards, 1).for_episode(cfg.training.seed)
    episode, summary = runner.drive_episode(env, 0, 'autopilot', exposure=exposure)
    episode.validate()
    assert episode.arrays['terminated'].sum() == int(terminal)
    assert episode.arrays['truncated'].sum() == int(not terminal)
    assert not episode.arrays['terminated'][:-1].any()
    assert summary['terminal_reason'] == ('success' if terminal else 'stuck')
    assert bool(exposure['disturbances']) == perturbations
    assert bool(env.hazard_seeds) == hazards
    assert summary['collection_exposure'] == episode.metadata['collection_exposure']
    if perturbations:
        assert not np.array_equal(episode.arrays['requested_actions'], episode.arrays['applied_actions'])


def test_append_keeps_baseline_episodes_splits_and_metadata(cfg, tmp_path, monkeypatch):
    env = ReplayEnv(cfg)
    monkeypatch.setattr(runner, 'CarlaDrivingEnv', lambda _: env)
    target = tmp_path / 'data'
    runner.collect(cfg, target, 1, 'autopilot')
    originals = {path: path.read_bytes() for path in target.glob('episode_*')}
    runner.collect(cfg, target, 2, 'autopilot', perturbations=True, staged_hazards=True, exposure_fraction=0)
    runner.collect(cfg, target, 1, 'autopilot', perturbations=True, staged_hazards=True, exposure_fraction=1)
    episodes = Dataset(target, cfg).load()
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert 'collection_exposure' not in episodes[0].metadata
    assert not episodes[1].metadata['collection_exposure']['exposed']
    assert episodes[3].metadata['collection_exposure']['exposed']
    assert env.hazard_seeds == [cfg.training.seed + 3]
    assert [episode.metadata['episode_id'] for episode in episodes] == [0, 1, 2, 3]


def test_reports_full_splits_and_capped_validation_suite(cfg, tmp_path):
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for identity in range(30):
        ep = make_episode(cfg, identity, steps=24, terminal=True)
        ep.metadata['terminal_reason'] = 'collision'
        dataset.save(ep)
    cfg.training.validation_episodes = 1
    report = runner.validate_dataset(cfg, dataset.path)
    all_validation = report['termination_support']['validation']
    fitted = report['fitting_coverage']['validation']
    assert len(all_validation['episode_ids']) == 3
    assert len(fitted['episode_ids']) == 1
    assert fitted['positive_count'] == 1
    assert fitted['terminal_density'] == 1 / 24
    assert fitted['ending_reason_counts'] == {'collision': 1}
    assert all_validation['ending_reason_counts'] == {'collision': 3}


class Vector:
    """Minimal CARLA vector/location stand-in for server-free hazard tests."""

    def __init__(self, x=0, y=0, z=0):
        """Store a three-dimensional location or direction."""
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        """Match CARLA's Euclidean location distance."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


class Transform:
    """Straight-lane transform with enough geometry for staging and visibility."""

    def __init__(self, location=None, rotation=None):
        """Default to a flat lane pointing along positive x."""
        self.location = location or Vector()
        self.rotation = rotation or SimpleNamespace(yaw=0)

    def get_forward_vector(self):
        """Return the fixed lane direction."""
        return Vector(x=1)

    def get_right_vector(self):
        """Return the fixed lane lateral direction."""
        return Vector(y=1)

    def transform(self, location):
        """Translate local fixture coordinates into world coordinates."""
        return Vector(self.location.x + location.x, self.location.y + location.y, self.location.z + location.z)


class Blueprint:
    """Stable vehicle and adult/child assets used by the fake simulator."""

    def __init__(self, identity):
        """Give actors a four-wheel count or explicit pedestrian age."""
        self.id = identity
        self.attrs = {'number_of_wheels': '4'} if identity.startswith('vehicle.') else {
            'age': 'child' if identity.endswith('0009') else 'adult', 'is_invincible': 'true'}

    def has_attribute(self, key):
        """Report only the fixture's actual blueprint attributes."""
        return key in self.attrs

    def get_attribute(self, key):
        """Mirror CARLA string and integer blueprint accessors."""
        return SimpleNamespace(as_str=lambda: self.attrs[key], as_int=lambda: int(self.attrs[key]))

    def set_attribute(self, key, value):
        """Record invincibility and role changes for assertions."""
        self.attrs[key] = value


def hazard_env(cfg, fail_spawns=0, ground=True, junction=False):
    """Build a fake world that exercises real placement and crossing code."""
    actors = []
    library = SimpleNamespace(filter=lambda pattern: [Blueprint('vehicle.test')] if pattern == 'vehicle.*' else
                              [Blueprint('walker.pedestrian.0001'), Blueprint('walker.pedestrian.0009')])
    world = SimpleNamespace(get_blueprint_library=lambda: library, get_actors=lambda: actors.copy(), attempts=0)
    world.cast_ray = lambda start, end: [SimpleNamespace(label='road', location=Vector(start.x, start.y))] if ground else []
    def spawn(bp, transform):
        """Simulate occupied candidates before accepting one real owned actor."""
        world.attempts += 1
        if world.attempts <= fail_spawns:
            return None
        actor = SimpleNamespace(id=42, is_alive=True, type_id=bp.id, attributes=bp.attrs,
                                bounding_box=SimpleNamespace(extent=Vector(.4, .3), location=Vector()),
                                get_location=lambda: transform.location, get_transform=lambda: transform,
                                controls=[])
        actor.apply_control = actor.controls.append
        actor.destroy = lambda: setattr(actor, 'is_alive', False)
        actors.append(actor)
        return actor
    world.try_spawn_actor = spawn
    carla = SimpleNamespace(Location=Vector, Vector3D=Vector, Transform=Transform,
                            Rotation=lambda **kw: SimpleNamespace(**kw),
                            VehicleControl=lambda **kw: SimpleNamespace(**kw),
                            WalkerControl=lambda **kw: SimpleNamespace(**kw),
                            CityObjectLabel=SimpleNamespace(Roads='road', Sidewalks='sidewalk', Ground='ground'))
    route_plan = [(SimpleNamespace(transform=Transform(Vector(x=x)), is_junction=junction, lane_width=3.5), None)
                  for x in range(0, 101, 2)]
    return SimpleNamespace(cfg=cfg, carla=carla, world=world, route=Route([[0, 0], [100, 0]]),
                           route_plan=route_plan, actors=[], sensors=[], controllers=[], provider=None,
                           frame=100, ego=SimpleNamespace(get_location=lambda: Vector()))


@pytest.mark.parametrize('kind', ['vehicle', 'adult', 'child'])
def test_hazard_placement_visibility_trigger_and_cleanup(cfg, kind):
    seed = next(seed for seed in range(100) if
                str(np.random.default_rng([seed, 733]).choice(('vehicle', 'adult', 'child'))) == kind)
    env = hazard_env(cfg, fail_spawns=2)
    hazard = StagedHazard(env, seed)
    env.staged_hazard = hazard
    assert env.actors == [hazard.actor]
    assert env.world.attempts == 3
    assert 20 <= hazard.metadata['route_metres'] <= 50
    assert hazard.metadata['kind'] == kind
    # The normal visibility path can see the real staged actor, without special slots.
    env.world.cast_ray = lambda start, end: []
    assert OracleExtractor(env).visible(hazard.actor, Transform(), 70)
    if kind == 'vehicle':
        assert hazard.actor.controls[-1].hand_brake
    else:
        assert hazard.actor.attributes['is_invincible'] == 'false'
        hazard.tick()
        assert not hazard.metadata['triggered']
        env.ego.get_location = lambda: Vector(hazard.centre.x - 5)
        hazard.tick()
        assert hazard.metadata['triggered']
        assert hazard.metadata['trigger_frame'] == 101
        assert hazard.actor.controls[-1].speed > 0
        for _ in range(400):
            hazard.tick()
        assert hazard.metadata['finished']
        assert hazard.actor.controls[-1].speed == 0
    CarlaDrivingEnv._destroy_actors(env)
    assert not hazard.actor.is_alive
    assert env.staged_hazard is None


@pytest.mark.parametrize('kwargs', [{'fail_spawns': 20}, {'ground': False}, {'junction': True}])
def test_hazard_placement_failure_is_bounded_and_explicit(cfg, kwargs):
    env = hazard_env(cfg, **kwargs)
    with pytest.raises(RuntimeError, match='20 candidates'):
        StagedHazard(env, 0)
    assert not env.actors
    assert env.world.attempts <= 20


def test_hazard_short_route_fails_without_actor_creation(cfg):
    env = hazard_env(cfg)
    env.route = Route([[0, 0], [25, 0]])
    with pytest.raises(RuntimeError, match='30 metres'):
        StagedHazard(env, 0)
    assert not env.actors


def test_hazard_clearance_rejects_existing_actor(cfg):
    env = hazard_env(cfg)
    hazard = StagedHazard(env, 0)
    assert not hazard._clear(hazard.actor.get_location())
    assert hazard._clear(Vector(x=500))

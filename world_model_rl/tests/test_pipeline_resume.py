"""Adopt an existing resumed trainer without starting a competing policy job."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from world_model_rl import pipeline
from world_model_rl.config import Config
from world_model_rl.data import write_json


def command(root):
    return [b'python', b'-m', b'world_model_rl', b'train', b'--config', str(root/'config.json').encode(),
        b'--resume', str(root/'agent.pt').encode(), b'--output', str(root).encode(),
        b'--training-clock', str(root/'training_clock.json').encode()]


@pytest.mark.parametrize('config_dir', ['.', 'config'])
def test_adoption_requires_this_experiments_resumed_training_and_clock(tmp_path, config_dir):
    config_path = tmp_path / config_dir / 'config.json'
    write_json(config_path, Config.load().to_dict())
    good = command(tmp_path)
    pipeline.check_adopted_command(good,tmp_path,'train')
    relocated = good.copy()
    relocated[relocated.index(b'--config') + 1] = str(config_path).encode()
    pipeline.check_adopted_command(relocated,tmp_path,'train')
    for flag in (b'--resume',b'--output',b'--training-clock',b'--config'):
        bad=good.copy();bad[bad.index(flag)+1]=b'another-experiment'
        with pytest.raises(ValueError):pipeline.check_adopted_command(bad,tmp_path,'train')
    fresh=good.copy();fresh[fresh.index(b'--resume')]=b'--model'
    with pytest.raises(ValueError):pipeline.check_adopted_command(fresh,tmp_path,'train')
    pipeline.check_adopted_command([b'python',b'-m',b'world_model_rl',b'fit',b'--config',
        str(tmp_path/'config.json').encode()],tmp_path,'fit')


def prepare(tmp_path,monkeypatch,deadline,config_dir='.'):
    write_json(tmp_path/config_dir/'config.json',Config.load().to_dict())
    write_json(tmp_path/'training_clock.json',{'deadline_unix':deadline})
    (tmp_path/'agent.pt').write_bytes(b'saved agent placeholder')
    identities=iter(['live','live',None])
    monkeypatch.setattr(pipeline,'process_identity',lambda pid:next(identities))
    original=Path.read_bytes
    def read_bytes(path):
        return b'\0'.join(command(tmp_path)) if str(path)=='/proc/123/cmdline' else original(path)
    monkeypatch.setattr(Path,'read_bytes',read_bytes)
    monkeypatch.setattr(pipeline.time,'sleep',lambda seconds:None)
    def forbidden(*args,**kwargs):raise AssertionError('Adoption cannot start a new fit or policy training command')
    monkeypatch.setattr(pipeline.subprocess,'Popen',forbidden)


@pytest.mark.parametrize('config_dir', ['.', 'config'])
def test_adopted_training_early_exit_stops_without_starting_another_job(tmp_path,monkeypatch,config_dir):
    prepare(tmp_path,monkeypatch,time.time()+3600,config_dir)
    with pytest.raises(RuntimeError,match='before the deadline'):
        pipeline.run(tmp_path,train_pid=123)
    state=json.loads((tmp_path/'pipeline_state.json').read_text())
    assert state['stage']=='training_stopped_early'
    assert not (tmp_path/'STAGE_COMMANDS.md').exists()


@pytest.mark.parametrize('config_dir', ['.', 'config'])
def test_adopted_training_reaches_client_check_only_after_deadline(tmp_path,monkeypatch,config_dir):
    prepare(tmp_path,monkeypatch,0,config_dir)
    class ReachedClientCheck(Exception):pass
    def client(*args,**kwargs):raise ReachedClientCheck
    monkeypatch.setitem(sys.modules,'carla',SimpleNamespace(Client=client))
    with pytest.raises(ReachedClientCheck):pipeline.run(tmp_path,train_pid=123)
    assert not (tmp_path/'STAGE_COMMANDS.md').exists()

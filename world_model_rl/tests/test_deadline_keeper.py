"""A file-only backup remains independent of learning success and model processes."""

import hashlib

import pytest

from world_model_rl.deadline_keeper import archive_latest
from world_model_rl.pipeline import process_identity


def test_deadline_backup_prefers_complete_agent_and_preserves_source(tmp_path):
    (tmp_path/'world_model.pt').write_bytes(b'world')
    (tmp_path/'agent.pt').write_bytes(b'latest complete agent')
    report=archive_latest(tmp_path,0)
    assert (tmp_path/'deadline_checkpoint.pt').read_bytes()==b'latest complete agent'
    assert report['sha256']==hashlib.sha256(b'latest complete agent').hexdigest()
    assert (tmp_path/'agent.pt').read_bytes()==b'latest complete agent'
    with pytest.raises(FileExistsError):archive_latest(tmp_path,0)


def test_deadline_backup_saves_model_even_without_a_policy(tmp_path):
    (tmp_path/'world_model.pt').write_bytes(b'fitted or partially fitted model')
    report=archive_latest(tmp_path,0)
    assert report['source'].endswith('world_model.pt')
    assert (tmp_path/'deadline_checkpoint.pt').read_bytes()==(tmp_path/'world_model.pt').read_bytes()


def test_process_identity_verifies_current_process_and_missing_handle():
    import os
    assert process_identity(os.getpid()) is not None
    assert process_identity(999999999) is None

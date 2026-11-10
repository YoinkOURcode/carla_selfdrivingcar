"""Experiment registration must preserve a single valid JSON document."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.training import register_run


REGISTRY = Path("world_model_rl/orchestration/experiment_registry.json")


def test_multiple_registrations_keep_one_json_document(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    register_run("runs/first", Config(), "train", 600)
    first = json.loads(REGISTRY.read_text())["experiments"][0]
    register_run("runs/second", Config(seed=8), "train", 120)
    data = json.loads(REGISTRY.read_text())
    assert [record["id"] for record in data["experiments"]] == ["first", "second"]
    assert data["experiments"][0] == first


def test_existing_registry_metadata_and_records_are_preserved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    REGISTRY.parent.mkdir(parents=True)
    previous = {"owner": "research", "experiments": [{"id": "historical", "notes": "preserve exactly"}]}
    REGISTRY.write_text(json.dumps(previous))
    inode = REGISTRY.stat().st_ino
    register_run("runs/new", Config(), "collect", 30)
    data = json.loads(REGISTRY.read_text())
    assert data["owner"] == "research"
    assert data["experiments"][0] == previous["experiments"][0]
    assert data["experiments"][1]["id"] == "new"
    assert REGISTRY.stat().st_ino == inode


def test_retry_of_identical_registration_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    register_run("runs/retry", Config(), "train", 600)
    register_run("runs/retry", Config(), "train", 600)
    assert len(json.loads(REGISTRY.read_text())["experiments"]) == 1


def test_concurrent_registrations_preserve_every_record(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(register_run, f"runs/run_{index}", Config(), "collect", 30)
                   for index in range(8)]
        for future in futures:
            future.result()
    records = json.loads(REGISTRY.read_text())["experiments"]
    assert sorted(record["id"] for record in records) == [f"run_{index}" for index in range(8)]


@pytest.mark.parametrize("content", ['{"experiments": []}\n{"experiments": []}', '{"experiments":', '[]'])
def test_invalid_registry_is_not_overwritten(monkeypatch, tmp_path, content):
    monkeypatch.chdir(tmp_path)
    REGISTRY.parent.mkdir(parents=True)
    REGISTRY.write_text(content)
    with pytest.raises(ValueError):
        register_run("runs/new", Config(), "train", 600)
    assert REGISTRY.read_text() == content

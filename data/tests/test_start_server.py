"""Launcher regressions: occupied ports fail before Unreal can crash."""

import errno
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest


DATA = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("server_preflight", DATA / "server_preflight.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def ports():
    sockets = [socket.socket() for _ in range(3)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        result = dict(zip(("rpc", "streaming", "secondary"),
                          (item.getsockname()[1] for item in sockets)))
    finally:
        for item in sockets:
            item.close()
    return result


@pytest.mark.parametrize("occupied", ["rpc", "streaming", "secondary"])
def test_active_listener_on_any_carla_port_is_rejected(ports, occupied):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", ports[occupied]))
        server.listen()
        with pytest.raises(RuntimeError, match=f"{occupied}={ports[occupied]}"):
            launcher.check_ports(ports)


def test_free_port_probe_releases_all_sockets(ports):
    launcher.check_ports(ports)
    for port in ports.values():
        with socket.socket() as server:
            server.bind(("0.0.0.0", port))
            server.listen()


def test_runtime_port_check_errors_never_launch_unreal(monkeypatch):
    def denied(*args):
        raise PermissionError(errno.EPERM, "socket access denied")
    monkeypatch.setattr(launcher.socket, "socket", denied)
    monkeypatch.setattr(launcher, "port_owners", lambda ports: "")
    monkeypatch.setattr(launcher.os, "execv", lambda *args: pytest.fail("Must not start Unreal"))
    assert launcher.main(["editor", "project"]) == 2


def test_custom_port_replaces_default_instead_of_appending_ignored_override():
    command, ports = launcher.launch_command("editor", "project with spaces",
        ["-carla-rpc-port=2010", "-carla-streaming-port=3011", "-nosplash"])
    assert ports == {"rpc": 2010, "streaming": 3011, "secondary": 2012}
    assert "-carla-rpc-port=2000" not in command
    assert command.count("-carla-rpc-port=2010") == 1
    assert command[1] == "project with spaces" and command[-1] == "-nosplash"


@pytest.mark.parametrize("arguments", [["-carla-rpc-port=65535"],
    ["-carla-rpc-port=-1"], ["-carla-rpc-port"], ["-carla-rpc-port=no"],
    ["-carla-rpc-port=2010", "-world-port=2011"], ["-carla-secondary-port=2000"]])
def test_invalid_or_conflicting_ports_fail_before_launch(arguments):
    with pytest.raises(ValueError):
        launcher.launch_command("editor", "project", arguments)


def fake_project(tmp_path):
    project = tmp_path / "project with spaces"
    data = project / "data"
    data.mkdir(parents=True)
    for name in ("start_server.sh", "server_preflight.py"):
        (data / name).write_text((DATA / name).read_text())
    editor = tmp_path / "engine with spaces" / "Engine" / "Binaries" / "Linux" / "UE4Editor"
    editor.parent.mkdir(parents=True)
    editor.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                      'open(os.environ["LAUNCH_RECORD"],"w").write(json.dumps(sys.argv))\n'
                      'sys.exit(7)\n')
    editor.chmod(0o755)
    record = tmp_path / "launched.json"
    environment = dict(os.environ, UE4_ROOT=str(editor.parents[3]), LAUNCH_RECORD=str(record))
    return data / "start_server.sh", record, environment


def test_launcher_refuses_busy_port_before_starting_editor(tmp_path, ports):
    script, record, environment = fake_project(tmp_path)
    with socket.socket() as server:
        server.bind(("127.0.0.1", ports["streaming"]))
        server.listen()
        result = subprocess.run(["bash", str(script),
            *(f"-carla-{name}-port={port}" for name, port in ports.items())],
            env=environment, capture_output=True, text=True)
    assert result.returncode == 2 and "ports already in use" in result.stderr
    assert not record.exists()


def test_launcher_exec_preserves_arguments_and_editor_failure(tmp_path, ports):
    script, record, environment = fake_project(tmp_path)
    result = subprocess.run(["bash", str(script),
        *(f"-carla-{name}-port={port}" for name, port in ports.items()), "-nosplash"],
        env=environment, capture_output=True, text=True)
    assert result.returncode == 7
    command = json.loads(record.read_text())
    assert "-game" in command and "-carla-rgb-only" in command and command[-1] == "-nosplash"
    assert all(f"-carla-{name}-port={port}" in command for name, port in ports.items())

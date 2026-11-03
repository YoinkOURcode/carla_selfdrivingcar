"""GPU preflight classification uses synthetic process lists, never a GPU."""

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dreamerv3_attention import training
from dreamerv3_attention.config import Config


PROJECT = str(Path(__file__).resolve().parents[2] / "Unreal/CarlaUE4/CarlaUE4.uproject")
SERVER = "/home/kevinkoh/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor"


def mock_gpu(monkeypatch, output, arguments=None, returncode=0):
    result = SimpleNamespace(stdout=output, stderr="driver unavailable", returncode=returncode)
    monkeypatch.setattr(training.subprocess, "run", lambda *args, **kwargs: result)
    monkeypatch.setattr(training, "_process_arguments", lambda pid: (arguments or {}).get(pid, []))


def server_arguments(port=None, project=PROJECT):
    return [SERVER, project, "/Game/Carla/Maps/Town01", "-game", *(
        [f"-carla-rpc-port={port}"] if port is not None else [])]


def test_reported_desktop_and_configured_carla_server_are_permitted(monkeypatch):
    output = f"2868, /usr/bin/kwin_wayland, 158 MiB\n33369, {SERVER}, 4430 MiB\n"
    mock_gpu(monkeypatch, output, {33369: server_arguments(2000)})
    training.check_gpu_workloads(Config())


def test_desktop_renderer_is_permitted_but_other_gpu_work_remains_blocked(monkeypatch):
    output = ("2868, /usr/bin/kwin_wayland, 158 MiB\n"
              "33, /usr/share/code/code --type=gpu-process, 162 MiB\n"
              "42, /usr/bin/python, 2000 MiB\n")
    mock_gpu(monkeypatch, output, {33: ["/usr/share/code/code", "--type=gpu-process"]})
    with pytest.raises(RuntimeError) as error:
        training.check_gpu_workloads(Config())
    assert "42, /usr/bin/python" in str(error.value)
    assert "2868" not in str(error.value) and "33," not in str(error.value)


def test_reported_desktop_renderer_command_works_without_host_proc_access(monkeypatch):
    mock_gpu(monkeypatch, "221123, /usr/share/code/code --type=gpu-process --ozone-platform=wayland, 162 MiB\n")
    training.check_gpu_workloads(Config())


def test_electron_combined_process_title_is_recognized(monkeypatch):
    title = "/usr/share/code/code --type=gpu-process --ozone-platform=wayland"
    mock_gpu(monkeypatch, f"221123, {title}, 162 MiB\n", {221123: [title]})
    training.check_gpu_workloads(Config())


def test_editor_executable_without_renderer_flag_is_not_a_desktop_exemption(monkeypatch):
    mock_gpu(monkeypatch, "221123, /usr/share/code/code, 162 MiB\n")
    with pytest.raises(RuntimeError, match="221123"):
        training.check_gpu_workloads(Config())


@pytest.mark.parametrize("arguments,cfg,launch_server", [
    (server_arguments(2010), Config(), False),
    (server_arguments(2000, "/other/project.uproject"), Config(), False),
    ([SERVER, PROJECT], Config(), False),  # editor UI, not a game server
    (server_arguments(2000), Config(host="remote.example"), False),
    (server_arguments(2000), Config(), True),  # cannot launch a second simulator
    ([], Config(), False),  # unreadable/missing process metadata
    (server_arguments(2000), None, False),  # no intended server supplied
])
def test_unrelated_or_unverified_unreal_process_still_blocks(monkeypatch, arguments, cfg, launch_server):
    mock_gpu(monkeypatch, f"33369, {SERVER}, 4430 MiB\n", {33369: arguments})
    with pytest.raises(RuntimeError, match="33369"):
        training.check_gpu_workloads(cfg, launch_server=launch_server)


@pytest.mark.parametrize("port", [None, 2000, 2010])
def test_default_and_custom_rpc_ports(monkeypatch, port):
    mock_gpu(monkeypatch, f"33369, {SERVER}, 4430 MiB\n", {33369: server_arguments(port)})
    training.check_gpu_workloads(Config(port=port or 2000))


def test_aliases_and_conflicting_rpc_flags(monkeypatch):
    args = server_arguments() + ["-world-port=2010"]
    mock_gpu(monkeypatch, f"33369, {SERVER}, 4430 MiB\n", {33369: args})
    training.check_gpu_workloads(Config(port=2010))
    args.append("-carla-rpc-port=2000")
    with pytest.raises(RuntimeError, match="33369"):
        training.check_gpu_workloads(Config(port=2010))


@pytest.mark.parametrize("output,returncode,message", [
    ("", 1, "Cannot inspect"),
    ("unparseable output", 0, "Cannot interpret"),
])
def test_failed_or_invalid_gpu_inspection_still_refuses_work(monkeypatch, output, returncode, message):
    mock_gpu(monkeypatch, output, returncode=returncode)
    with pytest.raises(RuntimeError, match=message):
        training.check_gpu_workloads(Config())


def test_lock_is_exclusive_and_not_deleted_or_replaced(monkeypatch, tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("old metadata is not an active lock")
    inode = path.stat().st_ino
    monkeypatch.setattr(training, "GPU_LOCK_PATH", str(path))
    mock_gpu(monkeypatch, "2868, /usr/bin/kwin_wayland, 158 MiB\n")
    with training.workload_lock(Config()):
        assert "pid" in json.loads(path.read_text())
        with path.open("a+") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="holds the machine lock"):
            with training.workload_lock(Config()):
                pytest.fail("A concurrent workload entered")
    assert path.exists() and path.stat().st_ino == inode
    with training.workload_lock(Config()):
        pass


def test_rejection_releases_lock_and_preserves_its_metadata(monkeypatch, tmp_path):
    path = tmp_path / "gpu.lock"
    path.write_text("existing metadata")
    monkeypatch.setattr(training, "GPU_LOCK_PATH", str(path))
    mock_gpu(monkeypatch, "42, /usr/bin/python, 2000 MiB\n")
    with pytest.raises(RuntimeError, match="42"):
        with training.workload_lock(Config()):
            pytest.fail("Unrelated compute job was allowed")
    assert path.read_text() == "existing metadata"
    mock_gpu(monkeypatch, "")
    with training.workload_lock(Config()):
        pass

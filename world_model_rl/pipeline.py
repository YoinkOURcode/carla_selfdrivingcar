"""Adopt a live fit or resumed SAC process; sequence the remaining stages."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import experiment_config_path


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # Linux stat field 22: process start ticks.
    except FileNotFoundError:
        return None


def check_adopted_command(command, root, mode):
    if mode not in ('fit', 'train') or b'world_model_rl' not in command or mode.encode() not in command:
        raise ValueError('The adopted process is not the expected world_model_rl stage')
    def option(flag):
        try:
            return command[command.index(flag) + 1]
        except (ValueError, IndexError):
            raise ValueError(f'The adopted process lacks {flag.decode()}') from None
    config_path = Path(option(b'--config').decode())
    if not config_path.exists() and config_path.name == 'config.json':
        config_path = experiment_config_path(config_path.parent)
    if config_path.resolve() != experiment_config_path(root).resolve():
        raise ValueError('The adopted process uses another experiment configuration')
    if mode == 'train':
        for flag, path in ((b'--resume', root / 'agent.pt'), (b'--output', root),
                           (b'--training-clock', root / 'training_clock.json')):
            if option(flag) != str(path).encode():
                raise ValueError('The resumed process belongs to another experiment or clock')


def run(experiment, fit_pid=None, train_pid=None):
    from .config import Config
    from .data import write_json
    from .training import imagination_gate

    root = Path(experiment)
    cfg = Config.load(experiment_config_path(root))
    expected = None
    if fit_pid is not None and train_pid is not None:
        raise ValueError('Adopt one process at a time')
    adopted_pid = fit_pid if fit_pid is not None else train_pid
    if adopted_pid is not None:
        expected = process_identity(adopted_pid)
        if expected is None:
            raise RuntimeError("The supplied process is not live; inspect its terminal result before proceeding")
        command = Path(f"/proc/{adopted_pid}/cmdline").read_bytes().split(b"\0")
        check_adopted_command(command, root, 'fit' if fit_pid is not None else 'train')
    elif (root / "agent.pt").exists():
        raise RuntimeError("A policy already exists; inspect its state instead of starting fresh again")
    state_path = root / "pipeline_state.json"
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": str(cfg.training.seed),
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    def state(stage, **extra):
        write_json(state_path, {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
                               "pipeline_pid": os.getpid(), **extra})
    def execute(stage, args, log_name):
        prefix = ["env", *[f"{k}={env[k]}" for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
                   "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")]]
        with (root / "STAGE_COMMANDS.md").open("a") as journal:
            journal.write(f"\n## {stage}\n\nPurpose: {stage.replace('_', ' ')}. Log: `logs/{log_name}`.\n\n```bash\n{shlex.join(prefix + args)}\n```\n")
        with (root / "logs" / log_name).open("x") as output:
            child = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT, env=env)
            state(stage, child_pid=child.pid, command=args)
            code = child.wait()
        with (root / "STAGE_COMMANDS.md").open("a") as journal:
            journal.write(f"\nExit status: {code}.\n")
        if code:
            state("stage_failed", failed_stage=stage, exit_code=code)
            raise RuntimeError(f"{stage} exited {code}; preserving artifacts, without automatic restart")
    if adopted_pid is not None:
        state("waiting_for_resumed_sac" if train_pid is not None else "waiting_for_existing_fresh_fit",
              adopted_pid=adopted_pid, adopted_start_ticks=expected)
        while process_identity(adopted_pid) == expected:
            time.sleep(5)
    clock = json.loads((root / "training_clock.json").read_text())
    if train_pid is None:
        report_path = root / "world_model.validation.json"
        if not report_path.exists():
            state("fitting_failed", reason="No final validation report from the adopted fit")
            raise RuntimeError("Fitting ended without its final report; inspect the saved fitting log")
        report = json.loads(report_path.read_text())
        passed, reason = imagination_gate(report, cfg)
        state("fresh_fit_finished", gate_passed=passed, gate_reason=reason)
        if not passed:
            state("blocked_model_gate", gate_reason=reason)
            raise RuntimeError(f"World-model gate failed: {reason}. No SAC fallback is run")
        execute("full_held_out_model_validation", [sys.executable, "-u", "-m", "world_model_rl", "evaluate-model",
            "--config", str(experiment_config_path(root)), "--data", str(root / "data"),
            "--checkpoint", str(root / "world_model.pt"), "--split", "validation",
            "--output", str(root / "world_model_full_validation.json")], "full_model_validation.log")
        if time.time() < clock["deadline_unix"]:
            execute("fresh_sac_until_shared_deadline", [sys.executable, "-u", "-m", "world_model_rl", "train",
                "--config", str(experiment_config_path(root)), "--data", str(root / "data"),
                "--model", str(root / "world_model.pt"), "--output", str(root),
                "--policy-updates", "1000000000", "--training-clock", str(root / "training_clock.json"),
                "--training-seconds", "18000", "--checkpoint-seconds", "300"], "policy_training.log")
    if time.time() < clock["deadline_unix"]:
        state('training_stopped_early', adopted_train_pid=train_pid)
        raise RuntimeError("Training stopped before the deadline; inspect the terminal result")
    # Read-only client check occurs after all offline updates have stopped.
    import carla
    client = carla.Client(cfg.environment.host, cfg.environment.port)
    client.set_timeout(cfg.environment.timeout)
    world = client.get_world()
    vehicles = [(v.id, v.attributes.get("role_name")) for v in world.get_actors().filter("vehicle.*")]
    write_json(root / "pre_evaluation_session.json", {"synchronous_mode": world.get_settings().synchronous_mode,
        "map": world.get_map().name, "vehicles": vehicles,
        "checked_utc": datetime.now(timezone.utc).isoformat()})
    if world.get_settings().synchronous_mode or vehicles:
        state("blocked_driving_client_check", vehicles=vehicles)
        raise RuntimeError("Another simulator owner or vehicle is present; no competing evaluation client is started")
    del world, client
    execute("development_selection_and_final50", [sys.executable, "-u", "-m", "world_model_rl.select_checkpoint",
        "--experiment", str(root)], "post_training_evaluation.log")
    metrics = json.loads((root / "final_metrics.json").read_text())
    state("evaluation_complete", acceptance_passed=metrics["prototype_acceptance"]["passed"], metrics=metrics)
    # Restore our explicit offline viewport preparation only while the world is idle.
    saved = json.loads((root / "provenance" / "server_settings_before_offline.json").read_text())
    client = carla.Client(cfg.environment.host, cfg.environment.port);client.set_timeout(cfg.environment.timeout)
    world = client.get_world();settings = world.get_settings()
    if not settings.synchronous_mode and not list(world.get_actors().filter("vehicle.*")):
        settings.no_rendering_mode = saved["no_rendering_mode"]
        world.apply_settings(settings)
        write_json(root / "server_settings_restored.json", {"no_rendering_mode": settings.no_rendering_mode})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--adopt-fit-pid", type=int)
    mode.add_argument("--adopt-train-pid", type=int,
                      help="Wait for this experiment's already-running resumed SAC; never initialize another policy")
    mode.add_argument("--from-fitted-model", action="store_true",
                      help="Continue after an explicitly completed readout recovery; never start another fit")
    args = parser.parse_args()
    run(args.experiment, args.adopt_fit_pid, args.adopt_train_pid)

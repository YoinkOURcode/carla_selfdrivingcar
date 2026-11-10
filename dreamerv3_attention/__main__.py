"""Executable smoke, collection, training, evaluation and comparison commands."""

import argparse
import json
import os
import subprocess
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from .config import Config
from .environment import CarlaEnv
from .evaluation import audit_trace, collection_scenarios, compare_results, evaluate
from .synthetic import SyntheticEnv
from .training import BudgetEnv, Trainer, register_run, seed_all, versions, workload_lock


@contextmanager
def server(cfg, engine_root, workload_deadline=None):
    """An optional owned server runs inside the same locked experiment."""
    process = None
    log = None
    try:
        if engine_root:
            import carla

            processes = subprocess.run(["ps", "-eo", "pid=,comm="], capture_output=True, text=True, check=True)
            if any("UE4Editor" in row or "CarlaUE4" in row for row in processes.stdout.splitlines()):
                raise RuntimeError("A CARLA/Unreal process already exists; connect to its idle dedicated server")
            engine = Path(engine_root) / "Engine/Binaries/Linux/UE4Editor"
            project = Path(__file__).resolve().parents[1] / "Unreal/CarlaUE4/CarlaUE4.uproject"
            if not engine.is_file():
                raise FileNotFoundError(engine)
            from data.server_preflight import check_ports

            check_ports({"rpc": cfg.port, "streaming": cfg.port+1, "secondary": cfg.port+2})
            log = open(Path("/tmp") / f"dreamerv3-server-{os.getpid()}.log", "w")
            process = subprocess.Popen([str(engine), str(project), f"/Game/Carla/Maps/{cfg.town}", "-game",
                                        "-vulkan", "-RenderOffScreen", "-nosound", "-unattended",
                                        f"-carla-rpc-port={cfg.port}", "-quality-level=Low"], stdout=log, stderr=log)
            deadline = min(time.monotonic()+120, workload_deadline or float("inf"))
            client = carla.Client(cfg.host, cfg.port)
            client.set_timeout(1.)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"CARLA exited; inspect {log.name}")
                try:
                    client.get_server_version()
                    break
                except RuntimeError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("CARLA startup exceeded the approved budget or 120 seconds")
                    time.sleep(0.5)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log:
            log.close()


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("smoke", "train", "collect", "validate", "evaluate",
                                         "infer", "compare", "preflight", "viewer", "baseline", "audit-controls"))
    cli.add_argument("--config", default="dreamerv3_attention/configs/attention.json")
    cli.add_argument("--output")
    cli.add_argument("--checkpoint")
    cli.add_argument("--world-model-only", action="store_true", help="Train the world model without actor/critic updates")
    cli.add_argument("--baseline")
    cli.add_argument("--candidate")
    cli.add_argument("--data-checkpoint", help="Validation/held-out collection checkpoint")
    cli.add_argument("--budget-seconds", type=float)
    cli.add_argument("--ue4-root", help="Launch and own a depth-capable server within the experiment lock")
    cli.add_argument("--web-port", type=int, default=8080, help="Local viewer HTTP port (default: 8080)")
    cli.add_argument("--training-run", help="Viewer: watch a training/evaluation run instead of auto-discovering runs/")
    cli.add_argument("--fixed-scenario", action="store_true",
                     help="Evaluate the collection scenario; training sanity only")
    cli.add_argument("--controller-speed", type=float, default=None,
                     help="Optional route-controller cruise cap in m/s")
    cli.add_argument("--controller", action="store_true", help="Evaluate RouteController on the same configured suite")
    cli.add_argument("--trace", help="JSONL trace input for audit-controls")
    cli.add_argument("--baseline-result", help="Require a successful matched controller result before training")
    return cli


def prepare_output(output):
    """Reuse an empty preflight-failure directory without touching run data."""
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Output already contains data; select a new run directory: {output}") from None


def check_baseline(cfg, result):
    """A controller must finish the same task before a gated pilot starts."""
    if (result.get("scope") != "fixed_training_sanity" or
            "controller_speed_mps" not in result or
            len(result.get("episodes", [])) != 1 or
            not result["episodes"][0].get("success") or
            result["episodes"][0].get("metrics", {}).get("collision_events", 1) != 0):
        raise ValueError("Controller baseline must complete one fixed route without collision")
    expected = json.loads(json.dumps(vars(cfg.resolved())))
    actual = dict(result.get("config", {}))
    for settings in (expected, actual):
        for key in ("steps", "checkpoint_every", "device", "warmup_frames", "warmup_policy",
                    "world_model_warmup_updates", "capacity", "actor_bc_weight", "expert_episode_interval",
                    "actor_steering_bc_weight", "actor_longitudinal_bc_weight"):
            settings.pop(key, None)
    if expected != actual:
        raise ValueError("Controller baseline configuration does not match the training task")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.world_model_only and args.command != "train":
        raise ValueError("--world-model-only is supported by train")
    if args.fixed_scenario and args.command not in ("evaluate", "infer", "baseline", "compare"):
        raise ValueError("--fixed-scenario is supported by evaluate, infer, baseline and compare")
    if args.baseline_result and args.command != "train":
        raise ValueError("--baseline-result is supported by train")
    if args.controller and args.command != "evaluate":
        raise ValueError("--controller is supported by evaluate")
    if args.controller and args.checkpoint:
        raise ValueError("Controller evaluation does not use --checkpoint")
    if args.command == "audit-controls":
        if not args.trace:
            raise ValueError("audit-controls requires --trace")
        source = Path(args.trace).resolve()
        report = audit_trace(source)
        if args.output:
            destination = Path(args.output).resolve()
            if destination == source or destination.exists():
                raise FileExistsError("Audit output must be a new path distinct from the trace")
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
            with destination.open("x", encoding="utf-8") as output_file:
                output_file.write(payload)
        print(json.dumps(report, indent=2, allow_nan=False))
        return report
    if args.command == "preflight":
        report = {"versions": versions(), "cuda_available": torch.cuda.is_available(),
                  "live_validation": "not executed"}
        print(json.dumps(report, indent=2))
        return report
    if args.command == "compare":
        if not args.baseline or not args.candidate:
            raise ValueError("compare requires --baseline and --candidate result JSON files")
        result = compare_results(json.loads(Path(args.baseline).read_text()),
                                 json.loads(Path(args.candidate).read_text()), fixed_scenario=args.fixed_scenario)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
        return result
    cfg = Config.load(args.config).resolved()
    if args.command == "viewer" and not args.checkpoint and cfg.viewer_checkpoint:
        args.checkpoint = cfg.viewer_checkpoint
    explicit_selection = cfg.scenario != "basic" or bool(cfg.scenarios)
    if args.command in ("evaluate", "infer") and explicit_selection:
        selected_evaluation = collection_scenarios(cfg)
        if cfg.split == "train" and len(selected_evaluation) == 1:
            args.fixed_scenario = True
    if args.command == "baseline" or args.controller:
        from .controller import RouteController
        from .driving import DrivingConfig
        controller = RouteController(args.controller_speed, DrivingConfig(**cfg.driving))
        cfg = Config(**{**vars(cfg), "device": "cpu"})
    if args.command == "viewer":
        from .viewer import serve

        torch.set_num_threads(1)
        return serve(cfg, args)
    seed_all(cfg.seed)
    torch.set_num_threads(1)
    if args.command == "validate":
        if not args.checkpoint or not args.data_checkpoint:
            raise ValueError("validate requires --checkpoint and --data-checkpoint")
        trainer = Trainer.restore(args.checkpoint, device="cpu")
        data = Trainer.restore(args.data_checkpoint, device="cpu")
        if data.cfg.split not in ("validation", "held_out"):
            raise ValueError("Validation requires separately collected validation/held_out data")
        if (data.cfg.image_size, data.cfg.max_objects) != (trainer.cfg.image_size, trainer.cfg.max_objects):
            raise ValueError("Validation observation contract differs")
        sample = data.replay.sample().reshape(data.cfg.batch, data.cfg.sequence)
        result = trainer.agent.validate(sample)
        print(json.dumps(result, indent=2))
        return result
    if args.command == "smoke":
        limits = {"steps": 64, "hidden": 32, "image_size": 32, "horizon": 5, "belief": 32,
                  "batch": 4, "sequence": 8, "categoricals": 4, "classes": 8,
                  "image_depth": 4, "bins": 31, "capacity": 256, "max_objects": 8}
        if cfg.device != "cpu" or any(getattr(cfg, key) > limit for key, limit in limits.items()):
            raise ValueError("smoke requires bounded CPU configuration; use configs/smoke.json")
    else:
        if args.budget_seconds is None or not 0 < args.budget_seconds <= 86400:
            raise ValueError("Live work requires --budget-seconds (0, 86400]")
        if args.command == "train" and cfg.split != "train":
            raise ValueError("Training and collection must use the train split")
        if (args.command == "evaluate" and cfg.split == "train" and
                not (args.fixed_scenario or explicit_selection)):
            raise ValueError("Evaluation must use validation or held_out split")
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; repair the NVIDIA driver before a GPU workload")
    if args.command in ("evaluate", "infer") and not args.checkpoint and not args.controller:
        raise ValueError("Evaluation/inference requires --checkpoint")
    selected = collection_scenarios(cfg) if args.command in ("train", "collect") else None
    if args.command == "baseline" or args.fixed_scenario:
        if len(collection_scenarios(cfg)) != 1:
            raise ValueError("Fixed evaluation requires exactly one scenario")
    if args.baseline_result:
        check_baseline(cfg, json.loads(Path(args.baseline_result).read_text()))
    output = Path(args.output or ("/tmp" if args.command == "smoke" else "runs"))
    if args.output is None:
        output = output / ("dreamerv3_"+args.command+"_"+time.strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}")
    if args.command == "smoke":
        prepare_output(output)
        if output.resolve().is_relative_to((Path.cwd()/"runs").resolve()):
            register_run(output, cfg, "cpu_synthetic_smoke", 30)
        env = SyntheticEnv(cfg)
        trainer = Trainer.restore(args.checkpoint, cfg) if args.checkpoint else Trainer(cfg, 4+cfg.max_objects+32)
        before = [p.detach().clone() for p in trainer.agent.loss.parameters()]
        result = trainer.run(env, output)
        changed = any(not torch.equal(a, b) for a, b in zip(before, trainer.agent.loss.parameters()))
        if not result["updates"] or not changed:
            raise RuntimeError("Synthetic smoke did not update model parameters")
        restored = Trainer.restore(output / "latest.pt")
        result.update(synthetic_only=True, checkpoint_verified=restored.frames == trainer.frames, output=str(output))
        (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
        return result
    with workload_lock(cfg, launch_server=bool(args.ue4_root)):
        prepare_output(output)
        register_run(output, cfg, "world_model" if args.world_model_only else args.command, args.budget_seconds)
        started = time.monotonic()
        deadline = started+args.budget_seconds
        with server(cfg, args.ue4_root, deadline):
            if args.command in ("train", "collect"):
                trainer = (Trainer.restore(args.checkpoint, cfg)
                           if args.checkpoint else
                           Trainer(cfg, 4+cfg.max_objects+32))
                if args.checkpoint:
                    print(f"[{args.command}] restored={Path(args.checkpoint).resolve()} "
                          f"frames={trainer.frames} updates={trainer.updates} "
                          f"completed_episodes={trainer.completed_episodes}", flush=True)
                print(f"[{args.command}] saving checkpoints to {output.resolve() / 'latest.pt'}", flush=True)
                with CarlaEnv(cfg, scenarios=selected, start_episode=trainer.completed_episodes) as env:
                    from .live import LiveTraining

                    mode = "collection" if args.command == "collect" else "world_model" if args.world_model_only else "dreamer"
                    with LiveTraining(output, cfg, mode) as observer:
                        result = trainer.run(BudgetEnv(env, deadline-time.monotonic()), output,
                                             collect_only=args.command == "collect", world_model_only=args.world_model_only,
                                             observer=observer)
                        observer.finish(result["stop_reason"])
            else:
                if args.command == "baseline" or args.controller:
                    agent = controller
                else:
                    trainer = Trainer.restore(args.checkpoint, cfg if args.fixed_scenario else None, device=cfg.device)
                    agent = trainer.agent
                    agent.checkpoint_path = str(Path(args.checkpoint).resolve())
                    if (trainer.cfg.image_size, trainer.cfg.max_objects) != (cfg.image_size, cfg.max_objects):
                        raise ValueError("Evaluation observation shape differs from checkpoint")
                    if trainer.cfg.attention != cfg.attention:
                        raise ValueError("Evaluation attention setting must match checkpoint")

                def factory(run_cfg, scenario):
                    env = CarlaEnv(run_cfg, scenario)
                    wrapped = BudgetEnv(env, deadline-time.monotonic())
                    wrapped.close = env.close
                    return wrapped

                from .live import LiveTraining

                publisher = (LiveTraining(output, cfg, "evaluation")
                             if args.command in ("evaluate", "infer") else nullcontext())
                with publisher as observer:
                    if observer is not None:
                        print("Live evaluation is available in the viewer's Live run tab.", flush=True)
                    result = evaluate(cfg, agent, factory, episodes=1 if args.command == "infer" else None,
                                      fixed_scenario=args.fixed_scenario or args.command == "baseline",
                                      trace_path=output / "trace.jsonl", observer=observer,
                                      action_source="controller" if args.command == "baseline" or args.controller
                                      else "actor")
                    if observer is not None:
                        observer.finish("completed")
                result["checkpoint"] = str(Path(args.checkpoint).resolve()) if args.checkpoint else None
                if args.command == "baseline":
                    result["controller_speed_mps"] = args.controller_speed
                if args.command == "baseline" or args.command == "evaluate":
                    if args.command == "baseline" or args.controller:
                        result["policy"] = ({"type": "route_controller", "speed_mode": "speed_limit"}
                                            if args.controller_speed is None else
                                            {"type": "route_controller", "target_speed_mps": args.controller_speed})
                    else:
                        result["policy"] = {"type": "actor"}
            result["elapsed_s"] = time.monotonic()-started
            result["versions"] = versions()
            (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            print(json.dumps(result, indent=2))
            return result


if __name__ == "__main__":
    main()

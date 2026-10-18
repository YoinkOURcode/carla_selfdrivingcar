"""CLI entry point. Importing the package never launches a simulator or training."""

import argparse
import json
from pathlib import Path

from .config import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CARLA attention/RSSM + model-assisted SAC")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "collect", "validate", "fit", "fit-termination", "evaluate-model", "train", "evaluate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
        if name in ("collect", "validate", "fit", "fit-termination", "evaluate-model", "train"):
            command.add_argument("--data", default="runs/default/data")
        if name in ("collect", "train", "evaluate", "run"):
            command.add_argument("--episodes", type=int)
        if name == "collect":
            command.add_argument("--controller", choices=("mixed", "autopilot", "noisy"), default="mixed")
            command.add_argument("--policy-checkpoint", help="Collect policy states with separate BasicAgent action labels")
            command.add_argument("--expert-probability", type=float, default=.5,
                                 help="Probability of executing the expert during labelled policy collection")
        if name == "validate":
            command.add_argument("--for-training", action="store_true",
                                 help="Exit with code 2 if training/validation event coverage is insufficient")
        if name in ("fit", "fit-termination", "evaluate-model", "evaluate"):
            command.add_argument("--checkpoint", required=True)
        if name == "fit":
            command.add_argument("--updates", type=int)
        if name == "fit-termination":
            command.add_argument("--updates", type=int)
            command.add_argument("--output", required=True, help="New world-model checkpoint path")
        if name == "evaluate-model":
            command.add_argument("--split", choices=("validation", "test"), default="test")
            command.add_argument("--output", default="runs/default/test_model.json")
        if name in ("train", "run"):
            command.add_argument("--output", default="runs/default")
            command.add_argument("--resume", action="store_true" if name == "run" else "store")
            command.add_argument("--policy-updates", type=int)
        if name == "train":
            command.add_argument("--model")
        if name == "evaluate":
            command.add_argument("--output", default="runs/default/evaluation.json")
            command.add_argument("--seed-offset", type=int, default=100000)
            command.add_argument("--plan", help="Fixed JSON scenario plan; overrides episode count and scenario seeds")
            command.add_argument("--record-traces", action="store_true", help="Save diagnostic traces explicitly excluded from training")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if getattr(args, "episodes", None) is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if getattr(args, "updates", None) is not None and args.updates < 1:
        parser.error("--updates must be positive")
    if getattr(args, "policy_updates", None) is not None:
        if args.policy_updates < 1 or args.episodes is not None:
            parser.error("Use a positive --policy-updates budget, separately from --episodes")
    # Heavy imports are deferred; --help/configuration do not need CARLA or CUDA.
    from . import runner
    def policy_stage(function, *arguments):
        try:
            return function(*arguments)
        except (runner.ImaginationGateError, runner.DataCoverageError) as exc:
            parser.exit(2, str(exc) + "\n")
    if args.command == "inspect":
        import carla
        client = carla.Client(cfg.environment.host, cfg.environment.port)
        client.set_timeout(cfg.environment.timeout)
        world = client.get_world()
        print(json.dumps({"client": client.get_client_version(), "server": client.get_server_version(),
                          "map": world.get_map().name, "available_maps": client.get_available_maps(),
                          "spawn_points": len(world.get_map().get_spawn_points())}, indent=2))
    elif args.command == "collect":
        runner.collect(cfg, args.data, args.episodes or cfg.training.bootstrap_episodes, args.controller,
                       args.policy_checkpoint, args.expert_probability)
    elif args.command == "validate":
        report = runner.validate_dataset(cfg, args.data)
        print(json.dumps(report, indent=2))
        if args.for_training:
            policy_stage(runner.require_fitting_coverage, report['fitting_coverage'])
    elif args.command == "fit":
        policy_stage(runner.fit, cfg, args.data, args.checkpoint, args.updates)
    elif args.command == "fit-termination":
        runner.fit_termination(cfg, args.data, args.checkpoint, args.output, args.updates)
    elif args.command == "evaluate-model":
        from .data import Dataset, write_json
        from .training import choose_device, evaluate_model, load_checkpoint
        device = choose_device(cfg.training.device)
        payload, model, normalizer, _ = load_checkpoint(args.checkpoint, cfg, device)
        report = evaluate_model(model, Dataset(args.data, cfg).load(args.split), normalizer, cfg, device,
                                termination_fit=payload.get("termination_fit"))
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        policy_stage(runner.train, cfg, args.data, args.output, args.episodes, args.model, args.resume, args.policy_updates)
    elif args.command == "evaluate":
        runner.evaluate(cfg, args.checkpoint, args.output, args.episodes or 10, args.seed_offset, args.plan, args.record_traces)
    elif args.command == "run":
        policy_stage(runner.run, cfg, args.output, args.episodes, args.resume, args.policy_updates)


if __name__ == "__main__":
    main()

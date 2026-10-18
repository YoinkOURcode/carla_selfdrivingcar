"""Post-deadline development selection and the complete reserved final evaluation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .config import Config, experiment_config_path
from .data import write_json
from .deadline import TrainingClock
from .evaluation import evaluation_metrics, validate_plan


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduled_candidates(clock, selection):
    """Select by preregistered times, never by favorable development outcomes."""
    eligible = [{**record, "archive_index": index} for index, record in enumerate(clock["checkpoints"])
                if Path(record["source"]).name == "agent.pt" and
                record.get("policy_updates", 0) > 0]
    chosen = {}
    first_sac = [r for r in eligible if r["stage"] == "first_sac"]
    if first_sac:
        chosen[first_sac[0]["archive_index"]] = first_sac[0]
    for seconds in selection["candidate_schedule_seconds"]:
        available = [r for r in eligible if r["saved_unix"] <= clock["started_unix"] + seconds]
        if seconds == clock["seconds"] and selection.get("last_cutoff_uses_final_deadline_snapshot"):
            stopped = [r for r in eligible if r["stage"] in ("deadline", "warmup_deadline")]
            if stopped:
                available = stopped
        if available:
            record = max(available, key=lambda r: r["archive_index"])
            chosen[record["archive_index"]] = record
    return [chosen[i] for i in sorted(chosen)]


def selection_score(metrics, archive_index):
    if not metrics["complete"]:
        raise ValueError("Incomplete development results cannot select a policy")
    return (metrics["arrivals_without_recorded_violations"], metrics["successes"],
            -metrics["collision_episodes"], -metrics["red_light_violations"],
            metrics["mean_route_completion"], -archive_index)


def verified_result(report, plan, cfg, checkpoint):
    if (report.get("plan") != plan or report.get("config") != cfg.to_dict() or
            report.get("checkpoint_sha256") != file_hash(checkpoint)):
        raise ValueError("Evaluation provenance differs from the frozen plan/config/checkpoint")
    metrics = evaluation_metrics(report["episodes"], plan, cfg)
    if not metrics["complete"]:
        raise RuntimeError("Evaluation is incomplete. Preserve its committed prefix; do not rerun favorable subsets")
    return metrics


def save_deadline_world_model(root, clock):
    import torch
    path = root / "world_model.pt"
    if not path.exists():
        raise FileNotFoundError("No fitted world model to save at the deadline")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "world_model":
        raise ValueError("The standalone model artifact is not a world-model checkpoint")
    agent_path = root / "agent.pt"
    if agent_path.exists():
        agent = torch.load(agent_path, map_location="cpu", weights_only=False)
        if agent.get("kind") != "agent":
            raise ValueError("The policy artifact is not an agent checkpoint")
        if payload["normalizer"] != agent["normalizer"] or any(
                not torch.equal(value, agent["model"][key]) for key, value in payload["model"].items()):
            raise ValueError("Frozen world-model tensors/normalizer differ from the final agent")
    payload.update(training_clock=clock.summary(), deadline_snapshot=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    clock.record_checkpoint(path, "deadline_world_model", {"model_updates": payload["model_updates"]}, force=True)


def finish(experiment):
    root = Path(experiment)
    if not (root / "training_clock.json").exists():
        raise RuntimeError("The training clock has not started")
    from . import runner
    cfg = Config.load(experiment_config_path(root))
    selection = json.loads(experiment_config_path(root, "selection_plan.json").read_text())
    if selection["ranking_descending"] != ["arrivals_without_recorded_violations", "successes",
            "negative_collision_episodes", "negative_red_light_violations",
            "mean_route_completion", "negative_archive_index"]:
        raise ValueError("Unsupported preregistered selection ranking")
    clock = TrainingClock(root / "training_clock.json", selection["training_seconds"], selection["checkpoint_seconds"])
    try:
        # The same lock proves the offline training command has released ownership.
        # It also prevents two post-training driving clients from this workflow.
        clock.start()
        if not clock.expired:
            raise RuntimeError("The training deadline has not passed; post-training evaluation is not started")
        plans = {}
        for kind in ("development", "final"):
            path = experiment_config_path(root, f"{kind}_plan.json")
            if file_hash(path) != selection[f"{kind}_plan_sha256"]:
                raise ValueError(f"Frozen {kind} plan was changed")
            plans[kind] = json.loads(path.read_text())
            validate_plan(plans[kind], cfg)
        save_deadline_world_model(root, clock)
        candidates = scheduled_candidates(clock.value, selection)
        if not candidates:
            raise RuntimeError("No learned policy candidate exists. World model saved at the deadline; no policy fallback is used")
        results = []
        for candidate in candidates:
            path = Path(candidate["path"])
            if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != candidate["sha256"]:
                raise ValueError("Candidate archive path/hash differs from its checkpoint record")
            output = root / "development" / f"candidate_{candidate['archive_index']:06d}.json"
            report = (json.loads(output.read_text()) if output.exists() else
                      runner.evaluate(cfg, path, output, plan_path=experiment_config_path(root, "development_plan.json"), record_traces=True))
            metrics = verified_result(report, plans["development"], cfg, path)
            results.append({"candidate": candidate, "metrics": metrics, "report": str(output)})
            write_json(root / "selection_progress.json", {"complete": False, "evaluated": results})
            print(f"Development candidate {candidate['archive_index']}: {metrics['successes']} arrivals, "
                  f"{metrics['collision_episodes']} collisions, {metrics['red_light_violations']} red violations", flush=True)
        winner = max(results, key=lambda r: selection_score(r["metrics"], r["candidate"]["archive_index"]))
        # The latest agent already has an immutable archive. Keep the selected
        # artifact at the required root path and retain every other candidate.
        temporary = root / "agent.pt.tmp"
        shutil.copyfile(winner["candidate"]["path"], temporary)
        temporary.replace(root / "agent.pt")
        choice = {"complete": True, "rule": selection, "evaluated": results, "selected": winner,
                  "selected_agent_sha256": file_hash(root / "agent.pt"),
                  "final_results_used_for_selection": False}
        write_json(root / "selection_report.json", choice)
        output = root / "final_evaluation.json"
        report = (json.loads(output.read_text()) if output.exists() else
                  runner.evaluate(cfg, root / "agent.pt", output, plan_path=experiment_config_path(root, "final_plan.json"), record_traces=True))
        metrics = verified_result(report, plans["final"], cfg, root / "agent.pt")
        if not metrics["prototype_acceptance"]["eligible"]:
            raise ValueError("Final evaluation does not cover the full balanced 50-episode plan")
        write_json(root / "final_metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    finally:
        clock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    finish(parser.parse_args().experiment)

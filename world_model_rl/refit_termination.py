"""Training-only, capacity-controlled recovery of a fitted RSSM event readout.

The dynamics, observation normalizer and reward model stay fixed. An internal
episode split of the training data selects capacity/regularization/update count;
the existing external validation is evaluated once, after the final refit.
"""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config, experiment_config_path
from .data import Dataset, write_json
from .deadline import TrainingClock
from .model import mlp
from .runner import validation_suite
from .training import (choose_device, clear_termination_moments, evaluate_model,
                       imagination_gate, load_checkpoint, rng_state, save_checkpoint,
                       seed_everything, termination_examples, termination_metrics)


# Committed before observing internal-split results. No external gate tuning.
CANDIDATES = (
    {"hidden": 8, "weight_decay": 0.0001, "noise": 0.05},
    {"hidden": 32, "weight_decay": 0.0001, "noise": 0.05},
    {"hidden": 32, "weight_decay": 0.001, "noise": 0.1},
    {"hidden": 128, "weight_decay": 0.0001, "noise": 0.05},
)
COUNTS = (500, 1000, 2000, 4000)
OBSERVATION_CANDIDATES = (
    {"hidden": 8, "weight_decay": 0.001, "noise": 1.0},
    {"hidden": 32, "weight_decay": 0.001, "noise": 1.0},
    {"hidden": 64, "weight_decay": 0.0001, "noise": 0.5},
    {"hidden": 128, "weight_decay": 0.0001, "noise": 0.5},
)


def internal_partition(episodes, seed):
    """Stratify by recorded ending reason; never move persisted dataset splits."""
    if not episodes or any(ep.metadata["split"] != "train" for ep in episodes):
        raise ValueError("Readout selection accepts training episodes only")
    groups = {}
    for index, episode in enumerate(episodes):
        groups.setdefault(episode.metadata.get("terminal_reason", "unspecified"), []).append(index)
    rng = np.random.default_rng(seed)
    holdout = set()
    for group in groups.values():
        shuffled = rng.permutation(group)
        if len(group) > 1:
            holdout.update(map(int, shuffled[:max(1, len(group) // 5)]))
    if not holdout:
        raise ValueError("Insufficient training episodes for an internal episode holdout")
    return np.concatenate([np.full(len(ep), i in holdout, dtype=bool)
                           for i, ep in enumerate(episodes)]), sorted(holdout)


def install_readout(destination, source, center, scale):
    """Fold input standardization and a smaller MLP into the existing plain MLP.

    Zero padding is exact (ELU(0)=0); inference/checkpoint architecture is unchanged.
    """
    with torch.no_grad():
        for parameter in destination.parameters():
            parameter.zero_()
        for index in (0, 2, 4):
            layer = source[index]
            destination[index].weight[:layer.out_features, :layer.in_features].copy_(layer.weight)
            destination[index].bias[:layer.out_features].copy_(layer.bias)
        width = source[0].out_features
        destination[0].weight[:width].copy_(source[0].weight / scale)
        destination[0].bias[:width].copy_(source[0].bias - (source[0].weight / scale) @ center)


@torch.no_grad()
def score(head, x, labels, offset):
    p = torch.cat([(head(chunk).squeeze(-1) + offset).sigmoid()
                   for chunk in x.split(4096)]).cpu().numpy()
    return termination_metrics(p, labels)


def rank(metrics, hidden, updates):
    precision, recall = metrics["precision"] or 0.0, metrics["recall"] or 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision >= 0.5 and recall >= 0.5, f1, precision, recall, -hidden, -updates)


def run(experiment, context_skip=False, observation_skip=False, device_override=None):
    root = Path(experiment)
    work = root / ("termination_observation_refit" if observation_skip else
                   "termination_context_refit" if context_skip else "termination_refit")
    work.mkdir(exist_ok=True)
    if (work / "plan.json").exists():
        raise FileExistsError("Readout recovery already started; inspect artifacts instead of restarting")
    cfg = Config.load(experiment_config_path(root))
    source_config = cfg.to_dict()
    if device_override is not None:
        cfg = copy.deepcopy(cfg)
        cfg.training.device = device_override
    clock = TrainingClock(root / "training_clock.json")
    clock.start()
    try:
        if clock.expired:
            raise RuntimeError("The shared training deadline has expired")
        seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
        device = choose_device(cfg.training.device)
        baseline = root / "world_model.pt"
        baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
        payload, model, normalizer, joint_optimizer = load_checkpoint(baseline, cfg, device)
        if model.termination_factorized:
            raise ValueError('Use factorized_termination to preserve separately labelled event semantics')
        if context_skip or observation_skip:
            cfg = copy.deepcopy(cfg)
            cfg.training.termination_context_skip = context_skip
            cfg.training.termination_observation_skip = observation_skip
            cfg.validate()
            model.termination_context_skip = context_skip
            model.termination_observation_skip = observation_skip
        physical_size = model.termination_physical_size
        candidates = OBSERVATION_CANDIDATES if observation_skip else CANDIDATES
        counts = (2000, 4000, 8000, 16000) if physical_size else COUNTS
        train = Dataset(root / "data", cfg).load("train")
        mask, held_indices = internal_partition(train, cfg.training.seed + 5101)
        plan = {"method": "causal-observation-train-internal-v1" if observation_skip else
                "causal-context-train-internal-v1" if context_skip else "capacity-controlled-train-internal-v1",
            "termination_context_skip": context_skip, "source_sha256": baseline_hash,
            "termination_observation_skip": observation_skip, "physical_input_size": physical_size,
            "device": str(device), "device_override": device_override,
            "source_model_updates": payload["model_updates"], "candidates": candidates,
            "counts": counts, "batch_size": 1024, "learning_rate": cfg.model.learning_rate,
            "noise_applies_to": "latent fields only" if physical_size else "all latent inputs",
            "holdout_episode_ids": [train[i].metadata["episode_id"] for i in held_indices],
            "fit_episode_ids": [ep.metadata["episode_id"] for i, ep in enumerate(train) if i not in held_indices],
            "selection": "passes precision/recall 0.5, F1, precision, recall, smaller hidden, fewer updates",
            "external_validation_used_for_selection": False,
            "limitation": "Frozen RSSM representation was fitted on the entire training split"}
        write_json(work / "plan.json", plan)
        write_json(work / "source_config.json", source_config)
        shutil.copyfile(baseline, work / "baseline.pt")
        shutil.copyfile(root / "world_model.validation.json", work / "baseline.validation.json")
        inputs, labels = termination_examples(model, train, normalizer, cfg, device)
        total_updates = 0
        records, best = [], None
        output = work / "world_model_refit.pt"
        def snapshot(head, center, scale, offset, stage, optimizer, generator):
            install_readout(model.termination_head, head, center, scale)
            model.termination_logit_offset.fill_(offset)
            save_checkpoint(output, cfg, model, normalizer, joint_optimizer,
                payload["model_version"] + 1, payload["model_updates"],
                source_checkpoint=str(work / "baseline.pt"), kind="world_model",
                termination_updates=payload.get("termination_updates", 0) + total_updates,
                readout_optimizer={"state": optimizer.state_dict(), "center": center.cpu(),
                    "scale": scale.cpu(), "hidden": head[0].out_features,
                    "parameterization": "standardized inputs before folding"},
                training_clock=clock.summary(), rng=rng_state(generator),
                training_episode_ids=[ep.metadata["episode_id"] for ep in train],
                termination_refit_plan=plan, fit_stage=stage)
            clock.record_checkpoint(output, stage, {"model_updates": payload["model_updates"],
                "termination_updates": payload.get("termination_updates", 0) + total_updates}, force=True)
        def fit_readout(indices, spec, updates, seed, evaluate_at=()):
            nonlocal total_updates, best
            local_rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            center = torch.as_tensor(inputs[indices].mean(0), device=device)
            scale = torch.as_tensor(np.maximum(inputs[indices].std(0), 1e-3), device=device)
            x = (torch.as_tensor(inputs, device=device) - center) / scale
            head = mlp(inputs.shape[1], 1, spec["hidden"]).to(device)
            optimizer = torch.optim.Adam(head.parameters(), lr=cfg.model.learning_rate,
                                        weight_decay=spec["weight_decay"])
            positive = indices[labels[indices]]
            negative = indices[~labels[indices]]
            if not len(positive) or not len(negative):
                raise ValueError("Internal fit lacks terminal/nonterminal training support")
            offset = float(np.log(len(positive) / len(negative)))
            for update in range(1, updates + 1):
                if clock.expired:
                    snapshot(head, center, scale, offset, "deadline", optimizer, local_rng)
                    raise RuntimeError("Shared deadline reached during termination recovery")
                sampled = np.concatenate([local_rng.choice(positive, 512), local_rng.choice(negative, 512)])
                batch = x[sampled]
                noise = torch.randn_like(batch) * spec["noise"]
                if physical_size:
                    noise[:, -physical_size:] = 0  # Exact current physical context is causally known.
                batch = batch + noise
                target = torch.as_tensor(labels[sampled], device=device, dtype=torch.float32)
                loss = nn.functional.binary_cross_entropy_with_logits(head(batch).squeeze(-1), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), cfg.model.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                total_updates += 1
                if clock.checkpoint_due:
                    snapshot(head, center, scale, offset, "termination_recovery", optimizer, local_rng)
                if update in evaluate_at:
                    metrics = score(head, x[mask], labels[mask], offset)
                    record = {"spec": spec, "updates": update, "metrics": metrics}
                    records.append(record)
                    key = rank(metrics, spec["hidden"], update)
                    if best is None or key > best[0]:
                        best = (key, copy.deepcopy(record))
                    write_json(work / "internal_results.json", {"records": records, "best": best[1]})
                    print(json.dumps(record), flush=True)
            return head, center, scale, offset, optimizer, local_rng
        for index, spec in enumerate(candidates):
            if spec["hidden"] > cfg.model.hidden:
                continue
            fit_readout(np.flatnonzero(~mask), spec, max(counts), cfg.training.seed + 5200 + index, counts)
        selected = best[1]
        write_json(work / "selected_before_external_validation.json", selected)
        head, center, scale, offset, optimizer, generator = fit_readout(np.arange(len(inputs)), selected["spec"],
            selected["updates"], cfg.training.seed + 5300)
        snapshot(head, center, scale, offset, "termination_refit_complete", optimizer, generator)
        clear_termination_moments(joint_optimizer, model)
        validation = validation_suite(Dataset(root / "data", cfg).load("validation"), cfg)
        stats = {"method": plan["method"], "source_sha256": baseline_hash,
            "internal_selection": selected, "updates": total_updates,
            "final_readout_updates": selected["updates"],
            "training_metrics": score(head, (torch.as_tensor(inputs, device=device) - center) / scale, labels, offset)}
        report = evaluate_model(model, validation, normalizer, cfg, device, termination_fit=stats)
        passed, reason = imagination_gate(report, cfg)
        save_checkpoint(output, cfg, model, normalizer, joint_optimizer,
            payload["model_version"] + 1, payload["model_updates"], kind="world_model",
            source_checkpoint=str(work / "baseline.pt"), validation=report, termination_fit=stats,
            termination_updates=payload.get("termination_updates", 0) + total_updates,
            training_episode_ids=[ep.metadata["episode_id"] for ep in train],
            validation_episode_ids=[ep.metadata["episode_id"] for ep in validation],
            training_clock=clock.summary(), termination_refit_plan=plan, rng=rng_state(generator))
        write_json(output.with_suffix(".validation.json"), report)
        write_json(work / "outcome.json", {"gate_passed": passed, "gate_reason": reason})
        clock.record_checkpoint(output, "validated_readout_refit", {"model_updates": payload["model_updates"],
            "termination_updates": payload.get("termination_updates", 0) + total_updates}, force=True)
        if passed:
            temporary = baseline.with_suffix(".pt.tmp")
            shutil.copyfile(output, temporary)
            temporary.replace(baseline)
            write_json(root / "world_model.validation.json", report)
            if physical_size:
                write_json(experiment_config_path(root), cfg.to_dict())
        print(f"Readout recovery gate: {passed} ({reason})", flush=True)
        return passed
    finally:
        clock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--context-skip", action="store_true",
                        help="Refit a learned causal physical-context event head; keep dynamics fixed")
    mode.add_argument("--observation-skip", action="store_true",
                      help="Include current visible object geometry, masks and light/control context")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        help="Explicit stage device override, recorded in the refit plan/config")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.experiment, args.context_skip, args.observation_skip, args.device) else 2)

"""Supervised fitting, held-out open-loop evaluation, gating, and imagination."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .control import apply_action_tensor
from .data import append_json, sample_sequences
from .model import RSSM
from .sac import HistoryCache, Replay
from .rewards import REWARD_VERSION
from .schema import (CONTEXT_FIELDS, OBS_KEYS, SCHEMA_VERSION, Normalizer,
                     continuous_valid, schema_description)


class ImaginationGateError(RuntimeError):
    """A quality check stopped policy learning; fitted model artifacts remain usable."""


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(rng):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "generator": rng.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value, rng):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    rng.bit_generator.state = value["generator"]
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and value["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def save_checkpoint(path, cfg, model, normalizer, optimizer, version, model_updates, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": 2, "kind": "world_model", "reward_version": REWARD_VERSION, "schema": schema_description(), "config": cfg.to_dict(),
               "model": model.state_dict(), "normalizer": normalizer.to_dict(),
               "model_optimizer": optimizer.state_dict(), "model_version": version,
               "model_updates": model_updates, **extra}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, cfg, device):
    from .config import require_compatible

    # Load only checkpoints you created/trust; optimizer and RNG state require pickle.
    payload = torch.load(path, map_location=device, weights_only=False)
    if (payload.get("format") != 2 or payload.get("reward_version") != REWARD_VERSION or
            payload["schema"]["version"] != SCHEMA_VERSION):
        raise ValueError("Incompatible checkpoint; train from scratch in a new experiment directory")
    if payload.get("kind") not in ("world_model", "agent"):
        raise ValueError("Unsupported checkpoint kind; expected a world model or SAC agent")
    require_compatible(payload["config"], cfg)
    model = RSSM(cfg.model).to(device)
    state = dict(payload["model"])
    # Format-2 checkpoints written before termination calibration used an implicit zero.
    state.setdefault("termination_logit_offset", model.termination_logit_offset.detach().clone())
    model.load_state_dict(state)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    optimizer.load_state_dict(payload["model_optimizer"])
    return payload, model, Normalizer.from_dict(payload["normalizer"]), optimizer


def fit_updates(model, optimizer, episodes, normalizer, cfg, updates, device, rng,
                log_path=None, previous_updates=0):
    model.train()
    # Joint sequence loss is an auxiliary event objective. Its biased window sampling
    # must not undo a saved class-prior correction; the final readout is refitted below.
    model.termination_logit_offset.zero_()
    metrics = {}
    started = time.monotonic()
    for update in range(updates):
        batch = sample_sequences(episodes, cfg.model.batch_size,
                                 cfg.model.sequence_length + cfg.model.burn_in, rng,
                                 cfg.training.failure_sequence_fraction)
        loss, metrics = model.sequence_loss(batch, normalizer, device)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite world-model loss; checkpoint was not overwritten")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        metrics.update(loss=float(loss.detach()), gradient_norm=float(norm),
                       model_update=previous_updates + update + 1)
        if (update + 1) % 100 == 0 or update + 1 == updates:
            elapsed = time.monotonic() - started
            metrics.update(seconds=elapsed, updates_per_second=(update + 1) / elapsed,
                           eta_seconds=(updates - update - 1) * elapsed / (update + 1), device=str(device))
            if log_path:
                append_json(log_path, metrics)
            print(f"World-model update {previous_updates + update + 1}: loss={metrics['loss']:.4f}", flush=True)
    model.eval()
    return metrics


def termination_metrics(probabilities, labels):
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels, dtype=bool)
    predicted = probabilities >= 0.5
    tp, fp = int(np.sum(predicted & labels)), int(np.sum(predicted & ~labels))
    fn, tn = int(np.sum(~predicted & labels)), int(np.sum(~predicted & ~labels))
    return {"positive_count": tp + fn, "negative_count": tn + fp,
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "threshold": 0.5, "brier": float(np.mean((probabilities - labels) ** 2)),
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None}


@torch.no_grad()
def termination_examples(model, episodes, normalizer, cfg, device):
    """Real-action prior means from full posterior histories, exactly as in validation."""
    model.eval()
    encoded = model.encode_episodes(episodes, normalizer, device, cfg.training.encoding_batch_episodes)
    h, z = cfg.model.deterministic, cfg.model.stochastic
    inputs, labels = [], []
    for episode, features in zip(episodes, encoded):
        for start in range(0, len(episode), 1024):
            end = min(start + 1024, len(episode))
            source = torch.as_tensor(features[start:end], device=device)
            applied = torch.as_tensor(episode.arrays["applied_actions"][start:end], device=device)
            state, _, _ = model.transition((source[:, :h], source[:, h:h + z]), applied, sample=False,
                                          weather=source[:, h + z:h + z + 4])
            inputs.append(torch.cat(state, -1).cpu().numpy())
        labels.append(episode.arrays["terminated"])
    if not inputs:
        raise ValueError("No training transitions for termination fitting")
    return np.concatenate(inputs), np.concatenate(labels)


def fit_termination_updates(model, episodes, normalizer, cfg, updates, device, rng, log_path=None):
    """Fit only the event readout; class-balanced draws get a training-prior correction."""
    if updates < 1:
        raise ValueError("Termination fitting requires a positive update budget")
    started = time.monotonic()
    inputs, labels = termination_examples(model, episodes, normalizer, cfg, device)
    encoding_seconds = time.monotonic() - started
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    support = {"positive_count": len(positive), "negative_count": len(negative)}
    if not len(positive) or not len(negative):
        return {"available": False, **support,
                "reason": "Need both task terminations and nonterminal real training transitions"}
    if cfg.training.batch_size < 2:
        raise ValueError("Termination fitting requires batch_size >= 2")
    x = torch.as_tensor(inputs, device=device)
    y = torch.as_tensor(labels, device=device).float()
    optimizer = torch.optim.Adam(model.termination_head.parameters(), lr=cfg.model.learning_rate)
    positive_draws = cfg.training.batch_size // 2
    sampled_prior = positive_draws / cfg.training.batch_size
    natural_prior = len(positive) / len(labels)
    # Bayes: balanced sampling preserves p(x|class), so restore the training event prior.
    offset = float(np.log(natural_prior / (1 - natural_prior)) -
                   np.log(sampled_prior / (1 - sampled_prior)))
    model.termination_head.train()
    started = time.monotonic()
    loss_value = None
    for update in range(updates):
        indices = np.concatenate([rng.choice(positive, positive_draws, replace=True),
            rng.choice(negative, cfg.training.batch_size - positive_draws, replace=True)])
        indices = torch.as_tensor(indices, device=device)
        # Optimize raw balanced logits; apply the prior correction only at inference.
        logits = model.termination_head(x[indices]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[indices])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite termination-head loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.termination_head.parameters(), cfg.model.gradient_clip,
                                       error_if_nonfinite=True)
        optimizer.step()
        if (update + 1) % 100 == 0 or update + 1 == updates:
            elapsed = time.monotonic() - started
            loss_value = float(loss.detach())
            metrics = {"stage": "termination_fit", "termination_update": update + 1,
                "balanced_loss": loss_value, "seconds": elapsed, "device": str(device),
                "updates_per_second": (update + 1) / elapsed,
                "eta_seconds": (updates - update - 1) * elapsed / (update + 1), **support}
            if log_path:
                append_json(log_path, metrics)
            print(f"Termination update {update + 1}/{updates}: balanced loss={loss_value:.5f}", flush=True)
    model.termination_logit_offset.fill_(offset)
    model.eval()
    fitting_seconds = time.monotonic() - started
    with torch.no_grad():
        probabilities = torch.cat([(model.termination_head(chunk).squeeze(-1) + model.termination_logit_offset).sigmoid()
                                   for chunk in x.split(4096)]).cpu().numpy()
    return {"available": True, **support, "updates": updates,
        "method": "balanced-full-history-prior-means-v1", "natural_training_prevalence": natural_prior,
        "sampled_training_prevalence": sampled_prior, "logit_offset": offset,
        "feature_encoding_seconds": encoding_seconds, "fitting_seconds": fitting_seconds,
        "balanced_loss": loss_value, "training_metrics": termination_metrics(probabilities, labels)}


def clear_termination_moments(optimizer, model):
    """Joint optimizer moments predate the separate readout fit and must be rebuilt."""
    for parameter in model.termination_head.parameters():
        optimizer.state.pop(parameter, None)


@torch.no_grad()
def evaluate_model(model, episodes, normalizer, cfg, device, cache=None, version=0, termination_fit=None):
    if not episodes:
        return {"available": False, "reason": "No held-out episodes"}
    model.eval()
    h, z = cfg.model.deterministic, cfg.model.stochastic
    encoded = (cache or HistoryCache()).encode(episodes, model, normalizer, device, version,
                                             cfg.training.encoding_batch_episodes)
    cache = list(zip(episodes, encoded))
    terminal_probabilities, terminal_labels = [], []
    # Event metrics use every held-out transition, retaining its natural class frequency.
    for ep, features in cache:
        for start in range(0, len(ep), 512):
            end = min(start + 512, len(ep))
            source = torch.as_tensor(features[start:end], device=device)
            applied = torch.as_tensor(ep.arrays["applied_actions"][start:end], device=device)
            state, _, _ = model.transition((source[:, :h], source[:, h:h + z]), applied, sample=False,
                                          weather=source[:, h + z:h + z + 4])
            terminal_probabilities.append(model.heads(state)["termination"].sigmoid().cpu().numpy())
            terminal_labels.append(ep.arrays["terminated"][start:end])
    probabilities, labels = np.concatenate(terminal_probabilities), np.concatenate(terminal_labels)
    terminal = termination_metrics(probabilities, labels)
    terminal["logit_offset"] = float(model.termination_logit_offset)
    terminal["latent_input"] = "action-conditioned prior means from full posterior histories"
    # Describe each real ending after prediction, without using its label/next observation
    # as a model input. Ranking against negatives distinguishes a threshold issue from
    # poor discrimination, while visible-object counts flag possible coverage gaps.
    negative_scores = np.sort(probabilities[~labels])
    terminal["endings"] = []
    terminal["by_episode"] = []
    terminal["false_positive_groups"] = []
    terminal["false_positive_group_count"] = 0
    terminal["false_positive_groups_limit"] = 20
    offset = 0
    for ep, _ in cache:
        episode_scores = probabilities[offset:offset + len(ep)]
        episode_labels = ep.arrays['terminated']
        false_positives = np.flatnonzero((episode_scores >= 0.5) & ~episode_labels)
        terminal['by_episode'].append({'episode_id': ep.metadata['episode_id'],
            'weather': ep.metadata.get('weather'), 'reason': ep.metadata.get('terminal_reason', 'unspecified'),
            **termination_metrics(episode_scores, episode_labels)})
        # Contiguous false alarms are described in time order, without changing the
        # metrics or fitting/selecting a threshold from these held-out outcomes.
        groups = np.split(false_positives, np.flatnonzero(np.diff(false_positives) > 1) + 1) if len(false_positives) else []
        terminal['false_positive_group_count'] += len(groups)
        for group in groups:
            if len(terminal['false_positive_groups']) >= terminal['false_positive_groups_limit']:
                break
            speed = ep.arrays['continuous'][group, 0]
            remaining = ep.arrays['continuous'][group, 13]
            visible = ep.arrays['present'][group].sum(-1)
            terminal['false_positive_groups'].append({
                'episode_id': ep.metadata['episode_id'], 'weather': ep.metadata.get('weather'),
                'first_transition': int(group[0]), 'last_transition': int(group[-1]),
                'count': len(group), 'decision_seconds': len(group) * cfg.environment.decision_dt,
                'probability_range': [float(episode_scores[group].min()), float(episode_scores[group].max())],
                'speed_range_mps': [float(speed.min()), float(speed.max())],
                'route_remaining_range_m': [float(remaining.min()), float(remaining.max())],
                'visible_objects_before_range': [int(visible.min()), int(visible.max())]})
        for step in np.flatnonzero(ep.arrays["terminated"]):
            probability = float(probabilities[offset + step])
            terminal["endings"].append({
                "episode_id": ep.metadata["episode_id"], "transition": int(step),
                "reason": ep.metadata.get("terminal_reason", "unspecified"),
                "weather": ep.metadata.get("weather"), "probability": probability,
                "visible_objects_before": int(ep.arrays["present"][step].sum()),
                "nonterminal_scores_at_or_above": int(len(negative_scores) -
                    np.searchsorted(negative_scores, probability, side="left"))})
        offset += len(ep)

    rng = np.random.default_rng(cfg.training.seed + 7919)
    candidates = [(i, t) for i, (ep, _) in enumerate(cache) for t in range(len(ep))]
    selected = rng.choice(len(candidates), min(cfg.training.validation_branches, len(candidates)), replace=False)
    horizons = {}
    for horizon in (1, 3, 5):
        errors, masks, reward_errors, class_correct, class_count, presence_errors, light_correct = [], [], [], 0, 0, [], []
        heading_errors = []
        for index in selected:
            episode_index, start = candidates[int(index)]
            ep, encoded = cache[episode_index]
            if start + horizon > len(ep):
                continue
            feature = torch.as_tensor(encoded[start:start + 1], device=device)
            state = feature[:, :h], feature[:, h:h + z]
            reward_sum = 0.0
            for step in range(horizon):
                applied = torch.as_tensor(ep.arrays["applied_actions"][start + step:start + step + 1], device=device)
                state, _, _ = model.transition(state, applied, sample=False, weather=feature[:, h + z:h + z + 4])
                prediction = model.heads(state)
                reward_sum += float(prediction["reward"].item())
            target_index = start + horizon
            target = ep.arrays["continuous"][target_index]
            physical = normalizer.denormalize(prediction["continuous"].cpu().numpy()[0])
            # Previous controls are causally known, not a learned rollout field.
            physical[5:8] = ep.arrays["applied_actions"][target_index - 1]
            physical[16:20] = ep.arrays["continuous"][start, 16:20]
            errors.append(physical - target)
            angle = np.arctan2(physical[9], physical[10]) - np.arctan2(target[9], target[10])
            heading_errors.append(abs(float(np.arctan2(np.sin(angle), np.cos(angle)))))
            mask = continuous_valid(ep.arrays["present"][target_index], ep.arrays["controls"][target_index])
            mask[5:8], mask[16:20] = False, False
            masks.append(mask)
            reward_errors.append(reward_sum - ep.arrays["rewards"][start:target_index].sum())
            actual_presence = ep.arrays["present"][target_index]
            presence_errors.append(float(np.mean((prediction["present"].sigmoid().cpu().numpy()[0] - actual_presence) ** 2)))
            classes = prediction["classes"].argmax(-1).cpu().numpy()[0]
            class_correct += int(np.sum((classes == ep.arrays["classes"][target_index]) & actual_presence))
            class_count += int(actual_presence.sum())
            light_correct.append(int(prediction["light"].argmax(-1).item() == ep.arrays["light"][target_index]))
        if errors:
            errors, masks = np.asarray(errors), np.asarray(masks)
            count = np.maximum(masks.sum(axis=0), 1)
            mae = (np.abs(errors) * masks).sum(axis=0) / count
            horizons[str(horizon)] = {
                "branches": len(errors), "seconds": horizon * cfg.environment.decision_dt,
                "normalized_rmse": float(np.sqrt(((errors / normalizer.scale) ** 2 * masks).sum() / masks.sum())),
                "context_mae": dict(zip(CONTEXT_FIELDS, mae[:len(CONTEXT_FIELDS)].tolist())),
                "heading_angle_mae_rad": float(np.mean(heading_errors)),
                "reward_sum_mae": float(np.mean(np.abs(reward_errors))),
                "reward_sum_bias": float(np.mean(reward_errors)),
                "presence_brier": float(np.mean(presence_errors)),
                "object_class_accuracy_when_present": class_correct / class_count if class_count else None,
                "light_accuracy": float(np.mean(light_correct))}
    collision_count = int(sum(np.sum(ep.arrays["events"][:, 0] > 0) for ep in episodes))
    report = {"available": True, "episodes": len(episodes), "transitions": len(labels),
            "episode_ids": [ep.metadata["episode_id"] for ep in episodes],
            "known_fields": "applied controls and episode-constant weather; excluded from prediction RMSE",
            "horizons": horizons, "termination": terminal, "collision_events": collision_count,
            "collision_modeling": "unvalidated" if collision_count == 0 else "aggregate termination head; inspect missed collisions"}
    if termination_fit is not None:
        report["termination_fit"] = termination_fit
    return report


def imagination_gate(report, cfg):
    if not report.get("available"):
        return False, report.get("reason", "No validation")
    horizon = report["horizons"].get(str(cfg.training.imagination_horizon))
    term, t = report["termination"], cfg.training
    reasons = []
    numbers = [term["brier"]]
    if horizon is not None:
        numbers.extend([horizon["normalized_rmse"], horizon["reward_sum_mae"], horizon["heading_angle_mae_rad"]])
    if not np.isfinite(numbers).all():
        reasons.append("nonfinite validation metrics")
    if horizon is None:
        reasons.append("insufficient sequence lengths")
    elif horizon["normalized_rmse"] > t.gate_normalized_rmse or horizon["reward_sum_mae"] > t.gate_reward_mae:
        reasons.append("prediction/reward tolerance failed")
    if horizon is not None and (horizon["context_mae"]["forward_speed_mps"] > t.gate_speed_mae or
            horizon["context_mae"]["lane_offset_left_m"] > t.gate_lane_mae or
            horizon["heading_angle_mae_rad"] > t.gate_heading_mae):
        reasons.append("speed/lane/heading tolerance failed")
    if term["positive_count"] < max(1, t.gate_min_terminal_positive) or term["negative_count"] < max(1, t.gate_min_terminal_negative):
        reasons.append("insufficient terminal/nonterminal labels")
    if term["brier"] > t.gate_terminal_brier:
        reasons.append("termination Brier tolerance failed")
    if term["positive_count"] and (term["recall"] is None or term["recall"] < t.gate_terminal_recall):
        reasons.append("termination recall tolerance failed")
    if ((term["positive_count"] and term["precision"] is None) or
            (term["precision"] is not None and term["precision"] < t.gate_terminal_precision)):
        reasons.append("termination precision tolerance failed")
    return not reasons, "; ".join(reasons) if reasons else "configured validation checks passed"


@torch.no_grad()
def imagine(model, sac, real, normalizer, cfg, device, rng):
    if not len(real):
        return Replay({}, real.version), {"generated": 0, "accepted": 0}
    model.eval()
    current = torch.as_tensor(real.sample(cfg.training.imagination_starts, rng)["features"], device=device)
    h, z = cfg.model.deterministic, cfg.model.stochastic
    chunks, generated = [], 0
    for _ in range(cfg.training.imagination_horizon):
        if not len(current):
            break
        requested, _ = sac.actor(current)
        applied = apply_action_tensor(requested, current[:, -3:], cfg.environment)
        weather = current[:, h + z:h + z + 4]
        state, prior, _ = model.transition((current[:, :h], current[:, h:h + z]), applied, sample=True, weather=weather)
        prediction = model.heads(state, prior_mean=prior.mean)
        next_features = model.features(state, applied, weather)
        if cfg.training.policy_observation_skip:
            from .policy import augment_policy_features, predicted_observation
            next_features = augment_policy_features(next_features,
                predicted_observation(prediction, applied, weather, normalizer))
        terminated = torch.rand(len(current), device=device) < prediction["termination"].sigmoid()
        physical = normalizer.denormalize(prediction["continuous"].cpu().numpy())
        valid = (torch.isfinite(next_features).all(-1) & torch.isfinite(prediction["reward"]) &
                 torch.isfinite(prediction["termination"]))
        # Numeric/physical sanity checks are not an epistemic uncertainty estimate.
        route_upper = max(cfg.environment.max_route_metres + 50,
                          float(normalizer.mean[13] + 6 * normalizer.scale[13] + 50))
        plausible = np.isfinite(physical).all(-1) & (np.abs(physical[:, 0]) <= 50) & (
            physical[:, 13] >= -5) & (physical[:, 13] <= route_upper)
        valid &= torch.as_tensor(plausible, device=device)
        generated += len(current)
        chunks.append({"features": current[valid].cpu().numpy(), "actions": requested[valid].cpu().numpy(),
                       "rewards": prediction["reward"][valid].cpu().numpy(),
                       "next_features": next_features[valid].cpu().numpy(),
                       "terminated": terminated[valid].cpu().numpy()})
        # A finite rollout horizon is not a task termination and does not suppress bootstrap.
        current = next_features[valid & ~terminated]
    if not chunks:
        return Replay({}, real.version), {"generated": generated, "accepted": 0}
    replay = Replay({key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}, real.version)
    return replay, {"generated": generated, "accepted": len(replay)}


def validation_suite(episodes, cfg, episode_ids=None):
    """Fixed whole episodes, in their natural distribution; never select by event labels."""
    if episode_ids is None:
        episode_ids = [ep.metadata["episode_id"] for ep in episodes[:cfg.training.validation_episodes]]
    selected = [ep for ep in episodes if ep.metadata["episode_id"] in episode_ids]
    if len(selected) != len(episode_ids):
        raise ValueError("A fixed validation episode is missing from the dataset")
    return selected

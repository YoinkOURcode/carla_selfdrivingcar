"""User-invoked collection, pretraining, mixed-replay learning and evaluation."""

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .config import experiment_config_path, scenario_key
from .collection import CollectionOptions, ControlPerturbations
from .data import Dataset, EpisodeBuilder, append_json, write_json
from .environment import CarlaDrivingEnv
from .model import RSSM, tensor_observation
from .policy import policy_feature_size, augment_policy_features
from .sac import HistoryCache, SAC, mixed_batch, real_replay, Demonstrations
from .schema import EVENTS, Normalizer
from .telemetry import EpisodeTelemetry
from .training import (choose_device, evaluate_model, fit_updates, imagination_gate,
                       imagine, load_checkpoint, restore_rng, rng_state, save_checkpoint,
                       seed_everything, validation_suite, fit_termination_updates,
                       clear_termination_moments, ImaginationGateError)


class DataCoverageError(ValueError):
    """Recorded data cannot support the configured fitting/validation checks yet."""


def termination_support(episodes):
    """Measure ending support without filtering or relabelling recorded episodes."""
    positive = sum(int(ep.arrays['terminated'].sum()) for ep in episodes)
    transitions = sum(map(len, episodes))
    return {'episode_ids': [ep.metadata['episode_id'] for ep in episodes],
            'transitions': transitions, 'positive_count': positive,
            'negative_count': transitions - positive,
            'terminal_density': positive / transitions if transitions else 0.0,
            'truncated_count': sum(int(ep.arrays['truncated'].sum()) for ep in episodes),
            'ending_reason_counts': dict(Counter(ep.metadata.get('terminal_reason', 'unspecified')
                                                for ep in episodes))}


def fitting_coverage(cfg, train, validation):
    """Inspect fixed splits before fitting; never select episodes by their outcomes."""
    result = {'train': termination_support(train), 'validation': termination_support(validation)}
    reasons = []
    if result['train']['transitions'] < max(1, cfg.training.minimum_real_steps):
        reasons.append(f"training split needs at least {max(1, cfg.training.minimum_real_steps)} transitions")
    for split, positives, negatives in (
            ('train', 1, 1),
            ('validation', max(1, cfg.training.gate_min_terminal_positive),
             max(1, cfg.training.gate_min_terminal_negative))):
        found = result[split]
        if found['positive_count'] < positives or found['negative_count'] < negatives:
            reasons.append(f"{split} has {found['positive_count']} task endings and "
                f"{found['negative_count']} nonterminal transitions; needs at least {positives} and {negatives}")
    if cfg.training.allow_scenario_variation:
        # A fixed old-town suite cannot establish quality in a new scene domain.
        domains = {scenario_key(ep.metadata) for ep in train}
        held_out = {scenario_key(ep.metadata) for ep in validation}
        for domain in sorted(domains - held_out):
            reasons.append(f"no validation episodes for training scenario {domain}")
    return {**result, 'ready': not reasons, 'reasons': reasons}


def require_fitting_coverage(report):
    if not report['ready']:
        raise DataCoverageError('Insufficient data before world-model fitting: ' + '; '.join(report['reasons']) +
            '. Active time limits are task endings; stuck endings are truncations. '
            'Collect a fixed additional batch of independent episodes, preserving the existing splits, '
            'then run validate --for-training. No gradient updates were performed.')


def one_observation(observation, normalizer, device):
    return tensor_observation({key: value[None] for key, value in observation.arrays().items()}, normalizer, device)


@torch.no_grad()
def drive_episode(env, episode_id, controller="mixed", model=None, normalizer=None, sac=None,
                  device=None, deterministic=False, seed=None, expert_probability=None, exposure=None):
    """Record a whole episode, preserving actions and the environment's ending labels."""
    if expert_probability is not None and (sac is None or not 0 <= expert_probability <= 1):
        raise ValueError("Expert-labelled collection requires a policy and a probability in [0, 1]")
    if exposure is not None and sac is not None:
        raise ValueError("Collection exposure requires a baseline controller")
    reset_started = time.monotonic()
    episode_seed = env.cfg.training.seed + episode_id if seed is None else seed
    reset_args = {}
    if exposure is not None and exposure["exposed"] and exposure["staged_hazards"]:
        reset_args["hazard_seed"] = episode_seed
    observation, info = env.reset(episode_id, seed, **reset_args)
    reset_seconds = time.monotonic() - reset_started
    noisy = controller == "noisy" or (controller == "mixed" and episode_id % 2 == 1)
    controller = "sac" if sac is not None else ("noisy" if noisy else "autopilot")
    if expert_probability is not None:
        controller = "dagger"
    metadata = {**env.metadata, "collector": controller,
                "policy_updates": 0 if sac is None else sac.updates}
    if expert_probability is not None:
        metadata.update(expert_probability=expert_probability, expert_source="BasicAgent")
    perturbations = None
    if exposure is not None:
        metadata["collection_exposure"] = exposure
        if exposure["exposed"]:
            if exposure["staged_hazards"]:
                exposure["hazard"] = env.staged_hazard.metadata
            if exposure["perturbations"]:
                perturbations = ControlPerturbations(episode_seed, env.cfg.environment.decision_dt,
                                                    exposure["disturbances"])
    builder = EpisodeBuilder(observation, info, metadata)
    rng = np.random.default_rng(metadata["seed"] + 17)
    telemetry = EpisodeTelemetry(env, metadata)
    def encode_live(observation):
        obs = one_observation(observation, normalizer, device)
        encoded, weights = model.encoder(obs, return_attention=True)
        return obs, encoded, weights[0].cpu().tolist()
    attention = None
    if sac is not None:
        obs, encoded, attention = encode_live(observation)
        state = model.observe_initial(obs, encoded=encoded)
    else:
        state = None
    telemetry.publish(observation, info, attention=attention)
    previous = torch.as_tensor(observation.continuous[None, 5:8], device=device) if sac else None
    weather = one_observation(observation, normalizer, device)["continuous"][:, 16:20] if sac else None
    summaries = []
    expert_labels = []
    started = time.monotonic()
    while True:
        if sac is None:
            requested = env.expert_action(rng if noisy else None)
            if perturbations is not None:
                requested = perturbations.action(requested, len(summaries),
                                                 float(np.linalg.norm(observation.continuous[:2])))
        else:
            features = model.features(state, previous, weather)
            if env.cfg.training.policy_observation_skip:
                features = augment_policy_features(features, one_observation(observation, normalizer, device))
            requested = sac.act(features, deterministic)[0]
            if expert_probability is not None:
                expert = env.expert_action()
                expert_labels.append(expert.copy())
                if rng.random() < expert_probability:
                    requested = expert
        observation, reward, terminated, truncated, info = env.step(requested)
        builder.append(requested, observation, reward, terminated, truncated, info)
        summaries.append(info)
        if sac is not None:
            previous = torch.as_tensor(info["applied_action"][None], device=device)
            obs, encoded, attention = encode_live(observation)
            state, _, _ = model.transition(state, previous,
                obs, sample=False, weather=weather, encoded=encoded)
        telemetry.publish(observation, info, reward, attention, active=not (terminated or truncated))
        if terminated or truncated:
            break
    env.finish_episode()
    episode = builder.finish(expert_labels) if expert_labels else builder.finish()
    a = episode.arrays
    distance = sum(item["distance_metres"] for item in summaries)
    events = {name: int(np.sum(a["events"][:, i] > 0)) for i, name in enumerate(EVENTS)}
    summary = {"episode_id": episode_id, "split": metadata["split"], "seed": metadata["seed"],
        "weather": metadata["weather"], "collector": controller, "return": float(a["rewards"].sum()),
        "steps": len(episode), "distance_metres": distance, "sim_seconds": len(episode) * env.cfg.environment.decision_dt,
        "wall_seconds": time.monotonic() - started, "route_completion": info["route_completion"],
        "reset_seconds": reset_seconds,
        "success": info["terminal_reason"] == "success", "terminal_reason": info["terminal_reason"],
        "event_decision_counts": events,
        "event_totals": {name: int(np.sum(a["events"][:, i])) for i, name in enumerate(EVENTS)},
        "terminal_count": int(a["terminated"].sum()),
        "terminal_density": float(a["terminated"].mean()),
        "truncated_count": int(a["truncated"].sum()),
        "collisions_per_km": events["collision"] / (distance / 1000) if distance > 0 else None,
        "mean_abs_lane_offset_m": float(np.mean([abs(item["lane_offset"]) for item in summaries])),
        "mean_abs_heading_error_rad": float(np.mean([abs(item["heading_error"]) for item in summaries]))}
    for key in ("acceleration_mps2", "jerk_mps3", "steering_rate_per_s"):
        summary["mean_" + key] = float(np.mean([item[key] for item in summaries]))
    if exposure is not None:
        summary["collection_exposure"] = exposure
    episode.metadata["summary"] = summary
    print(f"Episode {episode_id}: {controller}, {metadata['weather']}, {len(episode)} steps, "
          f"{info['route_completion']:.1%} route, {info['terminal_reason']}, return {summary['return']:.2f}", flush=True)
    return episode, summary


def collect(cfg, data_path, episodes, controller="mixed", policy_checkpoint=None, expert_probability=.5,
            *, perturbations=False, staged_hazards=False, exposure_fraction=.8):
    """Append all completed real episodes, with optional seeded failure exposure."""
    options = CollectionOptions(perturbations, staged_hazards, exposure_fraction)
    options.validate(policy_checkpoint)
    dataset = Dataset(data_path, cfg, create=True)
    policy_args = {}
    if policy_checkpoint is not None:
        if not 0 <= expert_probability <= 1:
            raise ValueError("Expert probability must be in [0, 1]")
        seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
        device = choose_device(cfg.training.device)
        payload, model, normalizer, _ = load_checkpoint(policy_checkpoint, cfg, device)
        if payload.get("kind") != "agent":
            raise ValueError("Expert-labelled collection requires an agent checkpoint")
        sac = SAC(policy_feature_size(model, cfg), cfg, device)
        sac.load_state_dict(payload["sac"])
        policy_args = dict(model=model, normalizer=normalizer, sac=sac, device=device,
                           deterministic=True, expert_probability=expert_probability)
    with CarlaDrivingEnv(cfg) as env:
        for _ in range(episodes):
            identity = dataset.next_id()
            exposure_args = {}
            if perturbations or staged_hazards:
                exposure_args["exposure"] = options.for_episode(cfg.training.seed + identity)
            episode, summary = drive_episode(env, identity, controller, **policy_args, **exposure_args)
            if policy_checkpoint is not None:
                episode.metadata["source_agent"] = str(policy_checkpoint)
            dataset.save(episode)
            append_json(dataset.path / "collection.jsonl", summary)


def validate_dataset(cfg, data_path):
    dataset = Dataset(data_path, cfg)
    episodes = dataset.load()
    for episode in episodes:
        episode.validate()
    return {"episodes": len(episodes), "transitions": sum(map(len, episodes)),
            "splits": {split: sum(ep.metadata["split"] == split for ep in episodes)
                       for split in ("train", "validation", "test")},
            "event_decision_counts": {name: int(sum(np.sum(ep.arrays["events"][:, i] > 0) for ep in episodes))
                                      for i, name in enumerate(EVENTS)},
            "termination_support": {split: termination_support(
                [ep for ep in episodes if ep.metadata['split'] == split])
                for split in ("train", "validation", "test")},
            "fitting_coverage": fitting_coverage(cfg,
                [ep for ep in episodes if ep.metadata['split'] == 'train'],
                validation_suite([ep for ep in episodes if ep.metadata['split'] == 'validation'], cfg))}


def fit(cfg, data_path, checkpoint, updates=None, warm_start=None, clock=None):
    if cfg.training.termination_factorized:
        raise ValueError('Fit the fresh RSSM with its source configuration, then use factorized_termination for readout recovery')
    dataset = Dataset(data_path, cfg)
    train, validation = dataset.load("train"), dataset.load("validation")
    for ep in [*train, *validation]:
        ep.validate()
    if Path(checkpoint).exists():
        raise FileExistsError(f"Choose a new checkpoint path; refusing to overwrite {checkpoint}")
    validation = validation_suite(validation, cfg)
    coverage = fitting_coverage(cfg, train, validation)
    write_json(Path(checkpoint).with_suffix('.coverage.json'), coverage)
    require_fitting_coverage(coverage)
    if clock is not None:
        clock.start()
    seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
    device = choose_device(cfg.training.device)
    source = {}
    if warm_start:
        source, model, normalizer, optimizer = load_checkpoint(warm_start, cfg, device)
        if source.get("kind") != "world_model":
            raise ValueError("--warm-start requires a world-model checkpoint")
        # Keep the meaning of the pretrained encoder inputs. New-domain data must
        # not silently replace its normalization statistics.
    else:
        normalizer = Normalizer.fit(train)
        model = RSSM(cfg.model, cfg.training.termination_context_skip, cfg.training.termination_observation_skip,
                     cfg.training.termination_factorized).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    rng = np.random.default_rng(cfg.training.seed)
    updates = cfg.training.pretrain_updates if updates is None else updates
    if updates < 1:
        raise ValueError("fit requires at least one update")
    joint_done = head_done = 0
    terminal_fit = head_optimizer = None
    report = {"available": False, "reason": "World-model fitting has not finished"}
    version = source.get("model_version", 0) + 1
    def checkpoint_fit(stage, force=False):
        if clock is None or not (force or clock.checkpoint_due):
            return
        save_checkpoint(checkpoint, cfg, model, normalizer, optimizer, version,
            source.get("model_updates", 0) + joint_done, kind="world_model",
            validation=report, rng=rng_state(rng), fit_stage=stage,
            training_clock=clock.summary(), termination_optimizer=head_optimizer,
            validation_episode_ids=[ep.metadata["episode_id"] for ep in validation],
            termination_fit=terminal_fit,
            termination_updates=source.get("termination_updates", 0) + head_done,
            source_checkpoint=str(warm_start) if warm_start else None,
            normalizer_frozen=bool(warm_start),
            training_episode_ids=[ep.metadata["episode_id"] for ep in train])
        clock.record_checkpoint(checkpoint, stage,
            {"model_updates": source.get("model_updates", 0) + joint_done,
             "termination_updates": source.get("termination_updates", 0) + head_done}, force=force)
    def joint_progress(count):
        nonlocal joint_done
        joint_done = count - source.get("model_updates", 0)
        checkpoint_fit("world_model")
        return not clock.expired
    def head_progress(count, head_opt):
        nonlocal head_done, head_optimizer
        head_done, head_optimizer = count, head_opt.state_dict()
        checkpoint_fit("termination")
        return not clock.expired
    if clock is not None:
        clock.start()
        checkpoint_fit("initial_model", force=True)
    started = time.monotonic()
    print(f"World model: device={device}, updates={updates}, train episodes={len(train)}", flush=True)
    stats = fit_updates(model, optimizer, train, normalizer, cfg, updates, device, rng,
                Path(checkpoint).parent / "world_model_training.jsonl", source.get("model_updates", 0),
                **({"progress": joint_progress} if clock is not None else {}))
    joint_done = stats["updates_completed"]
    fitting_seconds = time.monotonic() - started
    checkpoint_fit("joint_complete", force=True)
    clear_termination_moments(optimizer, model)
    if clock is not None and clock.expired:
        terminal_fit = {"available": False, "updates": 0, "reason": "Training deadline reached before readout fitting"}
    else:
        terminal_fit = fit_termination_updates(model, train, normalizer, cfg, cfg.training.termination_fit_updates,
            device, rng, Path(checkpoint).parent / "termination_training.jsonl",
            **({"progress": head_progress} if clock is not None else {}))
        head_done = terminal_fit.get("updates", 0)
    checkpoint_fit("deadline" if clock is not None and clock.expired else "fitted_model", force=True)
    started = time.monotonic()
    report = evaluate_model(model, validation, normalizer, cfg, device, termination_fit=terminal_fit)
    passed, reason = imagination_gate(report, cfg)
    save_checkpoint(checkpoint, cfg, model, normalizer, optimizer,
        version, source.get("model_updates", 0) + joint_done,
        kind="world_model", validation=report, rng=rng_state(rng),
        validation_episode_ids=[ep.metadata["episode_id"] for ep in validation],
        termination_fit=terminal_fit,
        termination_updates=source.get("termination_updates", 0) + terminal_fit.get("updates", 0),
        source_checkpoint=str(warm_start) if warm_start else None,
        normalizer_frozen=bool(warm_start),
        **({"training_clock": clock.summary(), "termination_optimizer": head_optimizer,
            "fit_stage": "deadline" if clock.expired else "complete"} if clock is not None else {}),
        timings={"model_fitting_seconds": fitting_seconds, "validation_seconds": time.monotonic() - started},
        training_episode_ids=[ep.metadata["episode_id"] for ep in train])
    if clock is not None:
        clock.record_checkpoint(checkpoint, "validated_model",
            {"model_updates": source.get("model_updates", 0) + joint_done,
             "termination_updates": source.get("termination_updates", 0) + head_done}, force=True)
    write_json(Path(checkpoint).with_suffix(".validation.json"), report)
    print(f"Saved {checkpoint}. Imagination gate: {passed} ({reason})", flush=True)
    return report


def fit_termination(cfg, data_path, checkpoint, output, updates=None):
    """Recover a failed ending head without repeating expensive dynamics fitting."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new checkpoint path; refusing to overwrite {output}")
    seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
    device = choose_device(cfg.training.device)
    payload, model, normalizer, optimizer = load_checkpoint(checkpoint, cfg, device)
    if payload.get("kind") != "world_model":
        raise ValueError("fit-termination requires a fitted world-model checkpoint")
    dataset = Dataset(data_path, cfg)
    train = dataset.load("train")
    validation = validation_suite(dataset.load("validation"), cfg)
    if not train or not validation:
        raise ValueError("Need real training and validation episodes")
    for episode in [*train, *validation]:
        episode.validate()
    rng = np.random.default_rng(cfg.training.seed)
    updates = cfg.training.termination_fit_updates if updates is None else updates
    print(f"Termination-only fit: device={device}; dynamics/reward/normalizer fixed; updates={updates}", flush=True)
    stats = fit_termination_updates(model, train, normalizer, cfg, updates, device, rng,
                                   output.parent / "termination_training.jsonl")
    if not stats["available"]:
        raise ValueError(stats["reason"])
    clear_termination_moments(optimizer, model)
    report = evaluate_model(model, validation, normalizer, cfg, device, termination_fit=stats)
    passed, reason = imagination_gate(report, cfg)
    save_checkpoint(output, cfg, model, normalizer, optimizer, payload["model_version"] + 1,
        payload["model_updates"], kind="world_model",
        termination_fit=stats, termination_updates=payload.get("termination_updates", 0) + updates,
        source_checkpoint=str(checkpoint), validation=report, rng=rng_state(rng),
        validation_episode_ids=[ep.metadata["episode_id"] for ep in validation],
        training_episode_ids=[ep.metadata["episode_id"] for ep in train])
    write_json(output.with_suffix(".validation.json"), report)
    print(f"Saved {output}. Imagination gate: {passed} ({reason})", flush=True)
    return report


def train(cfg, data_path, output, episodes=None, model_path=None, resume=None, policy_updates=None, clock=None):
    """Train SAC with a frozen world model; collected episodes only update real replay."""
    if episodes is not None and policy_updates is not None:
        raise ValueError("Choose --episodes or --policy-updates")
    if episodes is not None and cfg.training.collection_episodes < 1:
        raise ValueError("An episode budget requires collection_episodes > 0")
    if clock is not None and cfg.training.collection_episodes:
        raise ValueError("Timed policy training requires offline replay: set collection_episodes=0. "
                         "Collect fixed batches separately before training; no CARLA client is started by timed updates.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    agent_path = output / "agent.pt"
    if agent_path.exists() and not resume:
        raise FileExistsError(f"Use --resume {agent_path} or choose a new output")
    seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
    device = choose_device(cfg.training.device)
    dataset = Dataset(data_path, cfg)
    if bool(model_path) == bool(resume):
        raise ValueError("Supply exactly one of --model or --resume")
    payload, model, normalizer, optimizer = load_checkpoint(resume or model_path, cfg, device)
    if not resume and payload.get("kind") != "world_model":
        raise ValueError("--model requires a world-model checkpoint; a new SAC policy starts with fresh weights")
    # SAC trains on detached features. Keep every model head, buffer and the saved
    # normalizer fixed, including after collection and interrupted-run recovery.
    model.eval()
    model.requires_grad_(False)
    sac = SAC(policy_feature_size(model, cfg), cfg, device)
    rng = np.random.default_rng(cfg.training.seed)
    model_updates, version = payload["model_updates"], payload["model_version"]
    termination_updates = payload.get("termination_updates", 0)
    terminal_fit = payload.get("termination_fit")
    training_episodes = block_done = collection_remaining = 0
    replay_pending = False
    if resume:
        if payload.get("kind") != "agent":
            raise ValueError("--resume requires an agent checkpoint")
        ignored = ("device", "episodes", "policy_updates", "refresh_updates")
        old = {k: v for k, v in payload["config"]["training"].items() if k not in ignored}
        new = {k: v for k, v in cfg.to_dict()["training"].items() if k not in ignored}
        old.setdefault("termination_fit_updates", cfg.training.termination_fit_updates)
        old.setdefault("behavior_cloning_updates", 0)
        old.setdefault("behavior_cloning_weight", 0.0)
        old.setdefault("behavior_cloning_teacher", "recorded")
        old.setdefault("behavior_cloning_traffic_fraction", 0.0)
        old.setdefault("behavior_cloning_permute_objects", False)
        old.setdefault("policy_observation_skip", False)
        old.setdefault("allow_scenario_variation", False)
        old.setdefault("red_light_objective_multiplier", 1.0)
        old.setdefault("collision_objective_multiplier", 1.0)
        old.setdefault("behavior_cloning_applied_weight", 0.0)
        old.setdefault("deterministic_algorithms", False)
        old.setdefault("termination_context_skip", False)
        old.setdefault("termination_observation_skip", False)
        old.setdefault("termination_factorized", False)
        if old != new:
            raise ValueError("Resume training settings differ, apart from device/budgets")
        if payload["next_episode_id"] != dataset.next_id():
            raise ValueError("Checkpoint and dataset episode frontier differ; inspect interrupted collection before resuming")
        sac.load_state_dict(payload["sac"])
        restore_rng(payload["rng"], rng)
        training_episodes = payload["training_episodes"]
        block_done = payload["block_updates_done"]
        collection_remaining = payload["collection_remaining"]
        # Older checkpoints requested model fitting here. Only rebuild replay now.
        replay_pending = payload.get("replay_pending", payload.get("refresh_pending", False))
        if clock is not None and collection_remaining:
            raise ValueError("Timed offline training cannot resume pending collection")
    if clock is not None:
        clock.start()
    write_json(experiment_config_path(output), cfg.to_dict())
    suite = validation_suite(dataset.load("validation"), cfg, payload.get("validation_episode_ids"))
    if cfg.training.allow_scenario_variation:
        require_fitting_coverage(fitting_coverage(cfg, dataset.load("train"), suite))
    cache = HistoryCache()
    timings = {}
    def rebuild():
        started = time.monotonic()
        replay = real_replay(dataset.load("train", cfg.training.replay_steps), model, normalizer, device,
                             version, cache, cfg.training.encoding_batch_episodes, cfg.training.policy_observation_skip,
                             reward_cfg=cfg)
        timings["replay_encoding_seconds"] = time.monotonic() - started
        return replay
    def validate():
        started = time.monotonic()
        report = evaluate_model(model, suite, normalizer, cfg, device, cache, version,
                                termination_fit=terminal_fit)
        timings["validation_seconds"] = time.monotonic() - started
        write_json(output / "validation.json", report)
        return report
    def require_gate(report):
        enabled, reason = imagination_gate(report, cfg)
        if not enabled:
            terminal = report.get("termination")
            counts = (f" Endings: TP={terminal['true_positive']}, FP={terminal['false_positive']}, "
                      f"FN={terminal['false_negative']} at probability 0.5." if terminal else "")
            if terminal and terminal['positive_count'] == 0:
                counts = (f" Validation has 0 task endings and {terminal['negative_count']} nonterminal transitions; "
                    "termination recall is undefined, so this report cannot establish ending prediction quality.")
            fitted = (report.get("termination_fit") or {}).get("training_metrics")
            def ending_checks_pass(metrics):
                return (metrics is not None and metrics.get("recall") is not None and
                    metrics["recall"] >= cfg.training.gate_terminal_recall and
                    metrics.get("precision") is not None and
                    metrics["precision"] >= cfg.training.gate_terminal_precision and
                    metrics.get("brier", float("inf")) <= cfg.training.gate_terminal_brier)
            if terminal and (terminal["positive_count"] < cfg.training.gate_min_terminal_positive or
                    terminal["negative_count"] < cfg.training.gate_min_terminal_negative):
                recovery = ("Collect a fixed additional batch of independent real episodes, preserving split assignments. "
                    "Run validate --for-training before fitting a new model from scratch. "
                    "More head-fitting updates cannot create missing validation labels.")
            elif terminal and not ending_checks_pass(terminal) and ending_checks_pass(fitted):
                recovery = (f"The last head fit detected {fitted['true_positive']}/{fitted['positive_count']} "
                    f"training endings (precision {fitted['precision']:.3f}), but failed held-out data. "
                    "Repeating the same head-only fit is unlikely to help. ")
                if terminal.get('recall') is not None and terminal['recall'] >= cfg.training.gate_terminal_recall:
                    recovery += ("Recall passes, but false alarms reduce precision. Inspect termination.false_positive_groups "
                        "and termination.by_episode in the report to identify their contexts. Improve independent real "
                        "nonterminal coverage in those contexts, then use fit to train a new world model from scratch. "
                        "Do not use held-out transitions for gradient updates or choose a threshold just to pass this report.")
                else:
                    recovery += ("Inspect termination.endings in the report; improve real event and observation coverage, "
                        "then use fit to train a new world model from scratch. "
                        "An absent visible object does not identify the collision partner; static/out-of-view hazards are not recorded as object tokens.")
            elif terminal and not ending_checks_pass(terminal):
                recovery = ("Use fit-termination to refit only the ending head, then train --model with the new checkpoint. "
                    "If training endings fit but held-out endings fail, improve real event/observation coverage and jointly refit.")
            else:
                recovery = "Improve the real training data and use fit to train a new world model from scratch."
            preserved = agent_path if agent_path.exists() else resume or model_path
            raise ImaginationGateError("Imagination gate blocked SAC before further updates or CARLA collection: " + reason + "." +
                counts + f" Validation report: {output / 'validation.json'}. Fitted checkpoint preserved: {preserved}. " +
                recovery + " No real-only fallback is used.")
        return reason
    real = rebuild()
    demonstrations = None
    policy_initialization = payload.get("policy_initialization")
    def rebuild_demonstrations():
        if cfg.training.behavior_cloning_updates or cfg.training.behavior_cloning_weight:
            experts = [ep for ep in dataset.load("train") if cfg.training.behavior_cloning_teacher != "recorded"
                       or ep.metadata.get("collector") in ("autopilot", "dagger")]
            return Demonstrations(experts, model, normalizer, device, version, cache,
                                  cfg.training.encoding_batch_episodes, cfg.training.policy_observation_skip,
                                  cfg.training.behavior_cloning_teacher,
                                  cfg.training.behavior_cloning_traffic_fraction)
        return None
    if len(real) < cfg.training.minimum_real_steps:
        raise ValueError(f"Need at least {cfg.training.minimum_real_steps} real training transitions; found {len(real)}")
    real_steps = dataset.transition_count()
    report = validate()
    def refresh_replay():
        """Encode newly collected episodes with the unchanged model and normalizer."""
        nonlocal real, report, replay_pending, demonstrations
        real = rebuild()
        demonstrations = rebuild_demonstrations()
        report = validate()
        replay_pending = False
        passed, reason = imagination_gate(report, cfg)
        append_json(output / "training.jsonl", {"stage": "replay_refresh", "device": str(device),
            "model_updates": model_updates, "termination_updates": termination_updates,
            "termination_fit": terminal_fit, "model_version": version, "policy_updates": sac.updates,
            "imagination_enabled": passed, "imagination_gate": reason, "timings": dict(timings)})
    def checkpoint(stage="sac", force=False):
        started = time.monotonic()
        save_checkpoint(agent_path, cfg, model, normalizer, optimizer, version, model_updates,
            kind="agent", sac=sac.state_dict(), next_episode_id=dataset.next_id(),
            training_episodes=training_episodes, real_steps=real_steps, rng=rng_state(rng), validation=report,
            validation_episode_ids=[ep.metadata["episode_id"] for ep in suite],
            termination_updates=termination_updates, termination_fit=terminal_fit,
            block_updates_done=block_done, collection_remaining=collection_remaining,
            replay_pending=replay_pending, refresh_pending=False, world_model_frozen=True,
            policy_initialization=policy_initialization,
            **({"training_clock": clock.summary(), "training_stage": stage} if clock is not None else {}))
        if clock is not None:
            clock.record_checkpoint(agent_path, stage,
                {"policy_updates": sac.updates, "model_updates": model_updates,
                 "warmup_updates": (policy_initialization or {}).get("updates", 0)}, force=force)
        timings["checkpoint_seconds"] = time.monotonic() - started
    if replay_pending:
        refresh_replay()
        if not collection_remaining:
            block_done = 0
        checkpoint()
    gate_reason = require_gate(report)
    if clock is not None:
        if not resume and cfg.training.behavior_cloning_updates:
            policy_initialization = {"method": "fresh actor with expert training-split warm-up",
                "teacher": cfg.training.behavior_cloning_teacher, "updates": 0, "loss": None}
        checkpoint("initial_policy", force=True)
    demonstrations = rebuild_demonstrations()
    warmup_done = ((policy_initialization or {}).get("updates", cfg.training.behavior_cloning_updates)
                   if resume else 0)
    if warmup_done < cfg.training.behavior_cloning_updates:
        started = time.monotonic()
        policy_initialization = {"method": "fresh actor with expert training-split warm-up",
            "teacher": cfg.training.behavior_cloning_teacher,
            "permute_objects": cfg.training.behavior_cloning_permute_objects,
            "updates": warmup_done, "episode_ids": demonstrations.episode_ids,
            "traffic_sampling": {"fraction": demonstrations.traffic_fraction,
                "moving_stopping_states": len(demonstrations.traffic_moving),
                "stationary_stopping_states": len(demonstrations.traffic_waiting)},
            "loss": (policy_initialization or {}).get("loss")}
        for update in range(warmup_done, cfg.training.behavior_cloning_updates):
            if clock is not None and clock.expired:
                break
            loss = sac.clone_expert(demonstrations.sample(cfg.training.batch_size, rng))
            policy_initialization.update(updates=update + 1, loss=loss)
            if clock is not None and (clock.checkpoint_due or (update + 1) % cfg.training.checkpoint_updates == 0):
                checkpoint("warmup")
            if (update + 1) % 1000 == 0 or update + 1 == cfg.training.behavior_cloning_updates:
                append_json(output / "policy_bootstrap.jsonl", {"update": update + 1, "loss": loss,
                    "seconds": time.monotonic() - started, "device": str(device)})
                print(f"Expert warm-up {update + 1}: loss={loss:.6f}", flush=True)
        if clock is not None:
            checkpoint("warmup_deadline" if clock.expired else "warmup_complete", force=True)
            if clock.expired:
                append_json(output / "policy_bootstrap.jsonl", {"stage": "deadline", **policy_initialization,
                    "seconds": time.monotonic() - started, "device": str(device)})
                return {"policy_updates": sac.updates, "training_episodes": training_episodes,
                        "additional_policy_updates": 0, "additional_collection_episodes": 0,
                        "warmup_updates": policy_initialization["updates"], "deadline_reached": True,
                        "device": str(device)}
    print(f"SAC device={device}; world model frozen; real replay={len(real)}; "
          f"imagination gate passed: {gate_reason}", flush=True)
    write_json(experiment_config_path(output), cfg.to_dict())
    initial_updates, initial_episodes = sac.updates, training_episodes
    target_updates = initial_updates + (cfg.training.policy_updates if policy_updates is None else policy_updates)
    target_episodes = initial_episodes + episodes if episodes is not None else None
    invocation_started = time.monotonic()
    def done():
        return ((clock is not None and clock.expired) or
                (training_episodes >= target_episodes if target_episodes is not None else sac.updates >= target_updates))
    checkpoint()
    while not done():
        if collection_remaining:
            # This scope is the only simulator connection used during policy training.
            count = min(collection_remaining, target_episodes - training_episodes) if target_episodes is not None else collection_remaining
            with CarlaDrivingEnv(cfg) as env:
                for _ in range(count):
                    episode, summary = drive_episode(env, dataset.next_id(), model=model, normalizer=normalizer,
                                                     sac=sac, device=device)
                    dataset.save(episode)
                    real_steps += len(episode)
                    training_episodes += 1
                    collection_remaining -= 1
                    replay_pending = True
                    checkpoint()
                    append_json(output / "training.jsonl", {**summary, "stage": "collection", "device": str(device),
                        "policy_updates": sac.updates, "real_steps": real_steps, "timings": dict(timings)})
            # Actors have been destroyed before encoding the newly collected replay.
            refresh_replay()
            if not collection_remaining:
                block_done = 0
            checkpoint()
            gate_reason = require_gate(report)
            continue
        remaining = cfg.training.policy_block_updates - block_done
        if target_episodes is None:
            remaining = min(remaining, target_updates - sac.updates)
        block_started = time.monotonic()
        generated = accepted = used_synthetic = 0
        completed = 0
        synthetic = None
        sac_metrics = {}
        imagination_seconds = 0.0
        for update in range(remaining):
            if clock is not None and clock.expired:
                break
            if synthetic is None or update % cfg.training.imagination_refresh_updates == 0:
                started = time.monotonic()
                synthetic, counts = imagine(model, sac, real, normalizer, cfg, device, rng)
                imagination_seconds += time.monotonic() - started
                generated += counts["generated"]
                accepted += counts["accepted"]
                if not len(synthetic):
                    checkpoint()
                    raise RuntimeError("All imagined transitions failed numeric/physical checks; no SAC fallback was run")
            if clock is not None and clock.expired:
                break
            batch = mixed_batch(real, synthetic, cfg.training.batch_size, cfg.training.synthetic_fraction, rng)
            sac_metrics = (sac.update(batch, demonstrations.sample(cfg.training.batch_size, rng))
                           if demonstrations is not None and cfg.training.behavior_cloning_weight else sac.update(batch))
            block_done += 1
            completed += 1
            used_synthetic += int(cfg.training.batch_size * cfg.training.synthetic_fraction)
            if clock is not None and sac.updates == 1:
                checkpoint("first_sac", force=True)
            elif sac.updates % cfg.training.checkpoint_updates == 0 or (clock is not None and clock.checkpoint_due):
                checkpoint()
            if completed % 100 == 0 or completed == remaining:
                elapsed = time.monotonic() - block_started
                eta = ((target_updates - sac.updates) * elapsed / completed) if target_episodes is None else None
                metrics = {"stage": "sac", "device": str(device), "policy_updates": sac.updates,
                    "training_episodes": training_episodes, "real_steps": real_steps,
                    "real_replay_steps": len(real), "model_version": version, "model_updates": model_updates,
                    "termination_updates": termination_updates,
                    "imagination_enabled": True, "imagination_gate": gate_reason,
                    "imagination": {"generated": generated, "accepted": accepted},
                    "synthetic_samples_used": used_synthetic,
                    "synthetic_fraction": used_synthetic / (completed * cfg.training.batch_size),
                    "policy_block_seconds": elapsed, "imagination_seconds": imagination_seconds,
                    "policy_updates_per_second": completed / elapsed, "policy_compute_eta_seconds": eta,
                    "invocation_seconds": time.monotonic() - invocation_started,
                    "timings": dict(timings), **sac_metrics}
                append_json(output / "training.jsonl", metrics)
                print(f"SAC {sac.updates} updates; {completed / elapsed:.1f} updates/s; "
                      f"synthetic={metrics['synthetic_fraction']:.1%}; policy compute ETA={eta}", flush=True)
        if block_done == cfg.training.policy_block_updates:
            collection_remaining = cfg.training.collection_episodes
            if not collection_remaining:
                block_done = 0
        checkpoint()
    if clock is not None:
        checkpoint("deadline" if clock.expired else "policy_complete", force=True)
        append_json(output / "training.jsonl", {"stage": "deadline" if clock.expired else "timed_policy_complete",
            "policy_updates": sac.updates, "model_updates": model_updates,
            "warmup_updates": (policy_initialization or {}).get("updates", 0),
            "training_clock": clock.summary(), "device": str(device)})
    return {"policy_updates": sac.updates, "training_episodes": training_episodes,
            "additional_policy_updates": sac.updates - initial_updates,
            "additional_collection_episodes": training_episodes - initial_episodes,
            "device": str(device),
            **({"deadline_reached": clock.expired} if clock is not None else {})}


def evaluate(cfg, checkpoint, output, episodes=10, seed_offset=100000, plan_path=None, record_traces=False):
    import json
    from .select_checkpoint import file_hash
    from .evaluation import evaluation_plan, validate_plan, evaluation_metrics, save_evaluation_trace

    if Path(output).exists():
        raise FileExistsError(f"Choose a new evaluation output; refusing to overwrite {output}")
    if plan_path is not None:
        plan_path = Path(plan_path)
        if not plan_path.exists():
            plan_path = experiment_config_path(plan_path.parent, plan_path.name)
    plan = (json.loads(plan_path.read_text()) if plan_path else
            evaluation_plan(cfg, episodes, seed_offset))
    scenarios = validate_plan(plan, cfg)
    trace_directory = Path(output).with_suffix('').with_name(Path(output).stem + '_traces') if record_traces else None
    if trace_directory is not None and trace_directory.exists():
        raise FileExistsError(f"Choose a new evaluation trace directory: {trace_directory}")
    seed_everything(cfg.training.seed + seed_offset, cfg.training.deterministic_algorithms)
    device = choose_device(cfg.training.device)
    payload, model, normalizer, _ = load_checkpoint(checkpoint, cfg, device)
    if payload.get("kind") != "agent":
        raise ValueError("Policy evaluation requires an agent checkpoint")
    sac = SAC(policy_feature_size(model, cfg), cfg, device)
    sac.load_state_dict(payload["sac"])
    checkpoint_hash = file_hash(checkpoint)
    summaries = []
    def persist():
        result = {"episodes": summaries, "seed_offset": seed_offset, "plan": plan,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "policy_updates": sac.updates,
                  "model_version": payload["model_version"], "config": cfg.to_dict(),
                  "trace_directory": str(trace_directory) if trace_directory is not None else None,
                  **evaluation_metrics(summaries, plan, cfg)}
        write_json(output, result)
        return result
    persist()
    with CarlaDrivingEnv(cfg) as env:
        for scenario in scenarios:
            episode, summary = drive_episode(env, scenario["episode_id"], model=model, normalizer=normalizer, sac=sac,
                device=device, deterministic=True, seed=scenario["seed"])
            if summary["weather"] != scenario["weather"] or summary["seed"] != scenario["seed"]:
                raise RuntimeError("Driven scenario differs from the fixed evaluation plan")
            summary['split'] = 'evaluation'
            if trace_directory is not None:
                episode.metadata['evaluation_checkpoint'] = str(checkpoint)
                save_evaluation_trace(episode, trace_directory)
            summaries.append(summary)
            persist()
    return persist()


def run(cfg, output, episodes=None, resume=False, policy_updates=None):
    output = Path(output)
    data_path, model_path, agent_path = output / "data", output / "world_model.pt", output / "agent.pt"
    if resume:
        if not agent_path.exists():
            raise FileNotFoundError(f"No resumable agent at {agent_path}")
        return train(cfg, data_path, output, episodes=episodes, resume=agent_path, policy_updates=policy_updates)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Fresh training requires a new or empty output directory: {output}")
    if cfg.training.bootstrap_episodes < 10 or cfg.training.pretrain_updates < 1:
        raise ValueError("Fresh training requires at least ten bootstrap episodes and a positive pretrain_updates budget")
    output.mkdir(parents=True, exist_ok=True)
    write_json(experiment_config_path(output), cfg.to_dict())
    print(f"Fresh run: {output}; collecting {cfg.training.bootstrap_episodes} new episodes, "
          "then fitting a new world model and SAC policy", flush=True)
    collect(cfg, data_path, cfg.training.bootstrap_episodes)
    fit(cfg, data_path, model_path)
    return train(cfg, data_path, output, episodes=episodes, model_path=model_path, policy_updates=policy_updates)

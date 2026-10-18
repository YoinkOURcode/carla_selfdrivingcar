"""Whole real episodes, immutable split assignments, and no pickle/object hashing."""

import json
import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import require_compatible
from .rewards import REWARD_VERSION, learning_rewards
from .schema import (CONTINUOUS, EVENTS, OBS_KEYS, SCHEMA_VERSION, Observation,
                     schema_description)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def split_for(episode_id, seed):
    assignments = ["train"] * 8 + ["validation", "test"]
    random.Random(seed + episode_id // 10).shuffle(assignments)
    return assignments[episode_id % 10]


@dataclass
class Episode:
    arrays: dict
    metadata: dict

    def __len__(self):
        return len(self.arrays["rewards"])

    def validate(self):
        a, n = self.arrays, len(self)
        if self.metadata.get("schema") != SCHEMA_VERSION or n < 1:
            raise ValueError("Empty episode or incompatible schema")
        for key in OBS_KEYS:
            if len(a[key]) != n + 1:
                raise ValueError(f"{key} must include T+1 observations")
        for i in range(n + 1):
            Observation(**{key: a[key][i] for key in OBS_KEYS}).validate()
        action_fields = ["requested_actions", "applied_actions"]
        if self.metadata.get("collector") == "dagger":
            if "expert_actions" not in a:
                raise ValueError("Expert-labelled policy episode is missing expert_actions")
            action_fields.append("expert_actions")
        elif "expert_actions" in a:
            raise ValueError("expert_actions require explicit dagger collection provenance")
        for key in action_fields:
            if a[key].shape != (n, 3) or not np.isfinite(a[key]).all():
                raise ValueError(f"Invalid {key}")
            if np.any(a[key] < [-1.00001, -0.00001, -0.00001]) or np.any(a[key] > 1.00001):
                raise ValueError(f"Out-of-range {key}")
        if np.any(a["applied_actions"][:, 1] * a["applied_actions"][:, 2] > 1e-6):
            raise ValueError("Applied throttle and brake overlap")
        for key in ("rewards", "terminated", "truncated", "events", "components"):
            if len(a[key]) != n or not np.isfinite(a[key]).all():
                raise ValueError(f"Invalid {key}")
        if a["terminated"].dtype != bool or a["truncated"].dtype != bool:
            raise ValueError("Ending flags must be boolean")
        if a["events"].shape != (n, len(EVENTS)):
            raise ValueError("Invalid event array")
        if np.any(a["terminated"] & a["truncated"]) or np.any(
                a["terminated"][:-1] | a["truncated"][:-1]):
            raise ValueError("Episode has an internal ending or conflicting end flags")
        if not (a["terminated"][-1] or a["truncated"][-1]):
            raise ValueError("Final transition must end the episode")
        if a["frames"].shape != (n + 1,) or a["times"].shape != (n + 1,):
            raise ValueError("Invalid frame/time alignment")
        expected = self.metadata["config"]["environment"]
        if np.any(np.diff(a["frames"]) != expected["action_repeat"]):
            raise ValueError("Frame interval differs from action_repeat")
        dt = expected["fixed_delta_seconds"] * expected["action_repeat"]
        if not np.allclose(np.diff(a["times"]), dt, atol=1e-4):
            raise ValueError("Simulation timestamps differ from the decision interval")
        if a["components"].shape[1] != len(self.metadata["component_names"]) or not np.allclose(
                a["components"].sum(-1), a["rewards"], atol=1e-4, rtol=1e-5):
            raise ValueError("Reward components do not sum to reward")
        if not np.allclose(a["continuous"][1:, 5:8], a["applied_actions"], atol=1e-6):
            raise ValueError("Observation action history differs from applied actions")
        if not np.allclose(a["continuous"][:, 16:20], a["continuous"][0, 16:20], atol=1e-5):
            raise ValueError("Weather must remain fixed within an episode")
        return self


class EpisodeBuilder:
    def __init__(self, observation, info, metadata):
        self.observations = [observation.validate().arrays()]
        self.frames = [info["frame"]]
        self.times = [info["sim_time"]]
        self.steps = []
        self.metadata = metadata
        self.metadata["reward_version"] = REWARD_VERSION

    def append(self, requested, observation, reward, terminated, truncated, info):
        self.observations.append(observation.validate().arrays())
        self.frames.append(info["frame"])
        self.times.append(info["sim_time"])
        self.steps.append({"requested_actions": requested, "applied_actions": info["applied_action"],
                           "rewards": reward, "terminated": terminated, "truncated": truncated,
                           "events": [info["events"][name] for name in EVENTS],
                           "components": list(info["components"].values())})
        self.metadata["component_names"] = list(info["components"])
        self.metadata["terminal_reason"] = info["terminal_reason"]

    def finish(self, expert_actions=None):
        arrays = {key: np.stack([obs[key] for obs in self.observations]) for key in OBS_KEYS}
        for key in self.steps[0]:
            dtype = bool if key in ("terminated", "truncated") else np.float32
            arrays[key] = np.array([step[key] for step in self.steps], dtype=dtype)
        arrays["frames"] = np.array(self.frames, np.int64)
        arrays["times"] = np.array(self.times, np.float64)
        if expert_actions is not None:
            arrays["expert_actions"] = np.asarray(expert_actions, np.float32)
        return Episode(arrays, self.metadata).validate()


class Dataset:
    def __init__(self, path, cfg, create=False):
        self.path = Path(path)
        manifest = self.path / "dataset.json"
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if value["schema"]["version"] != SCHEMA_VERSION:
                raise ValueError("Incompatible dataset schema")
            if value.get("reward_version") != REWARD_VERSION:
                raise ValueError("Incompatible reward dataset; collect fresh data in a new output directory")
            require_compatible(value["config"], cfg)
            if value["split_seed"] != cfg.training.seed:
                raise ValueError("Dataset split seed differs from configuration")
        elif create:
            write_json(manifest, {"schema": schema_description(), "config": cfg.to_dict(),
                                  "reward_version": REWARD_VERSION,
                                  "split_seed": cfg.training.seed,
                                  "split_rule": "seeded shuffle of 8/1/1 per ten episodes",
                                  "observation_source": cfg.environment.observation_provider})
        else:
            raise FileNotFoundError(f"Dataset does not exist: {manifest}")
        self.cfg = cfg

    def paths(self):
        return sorted(self.path.glob("episode_*.npz"))

    def next_id(self):
        identity = max([int(p.stem.split("_")[-1]) for p in self.paths()], default=-1) + 1
        if (self.path / f"episode_{identity:06d}.json").exists():
            raise RuntimeError("Interrupted episode sidecar found; inspect it before reusing this dataset")
        return identity

    def transition_count(self):
        total = 0
        for path in self.paths():
            with np.load(path, allow_pickle=False) as archive:
                total += len(archive["rewards"])
        return total

    def save(self, episode):
        if episode.metadata.get("usage") == "evaluation_only_never_train" or episode.metadata.get("split") == "evaluation":
            raise ValueError("Evaluation traces must never be saved to a training dataset")
        episode.validate()
        if episode.metadata.get("reward_version") != REWARD_VERSION:
            raise ValueError("Episode reward version differs; collect fresh compatible data")
        require_compatible(episode.metadata["config"], self.cfg)
        identity = int(episode.metadata["episode_id"])
        path = self.path / f"episode_{identity:06d}.npz"
        sidecar = path.with_suffix(".json")
        if path.exists() or sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **episode.arrays)
        # NPZ is the commit marker; interrupted sidecars are detected, never silently reused.
        write_json(sidecar, episode.metadata)
        temporary.replace(path)
        episodes = [json.loads(p.with_suffix(".json").read_text()) for p in self.paths()]
        write_json(self.path / "splits.json", {
            split: [ep["episode_id"] for ep in episodes if ep["split"] == split]
            for split in ("train", "validation", "test")})

    def load(self, split=None, limit_steps=None):
        episodes = []
        paths = list(reversed(self.paths())) if limit_steps else self.paths()
        total = 0
        for path in paths:
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata.get("usage") == "evaluation_only_never_train":
                raise ValueError(f"Evaluation trace must never enter training replay: {path}")
            if metadata.get("reward_version") != REWARD_VERSION:
                raise ValueError(f"Episode reward version mismatch: {path}")
            require_compatible(metadata["config"], self.cfg)
            if metadata["episode_id"] != int(path.stem.split("_")[-1]):
                raise ValueError(f"Episode identity differs from filename: {path}")
            if metadata["schema"] != SCHEMA_VERSION or metadata["split"] != split_for(
                    metadata["episode_id"], self.cfg.training.seed):
                raise ValueError(f"Invalid schema/split assignment: {path}")
            if split is not None and metadata["split"] != split:
                continue
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            episodes.append(Episode(arrays, metadata))
            if limit_steps:
                total += len(episodes[-1])
                if total >= limit_steps:
                    break
        if limit_steps:
            episodes.reverse()
        return episodes


def episode_fingerprint(episode):
    """Deduplicate recording contents independently of compression and global ID."""
    digest = hashlib.sha256()
    context = {key: episode.metadata.get(key) for key in ("seed", "weather", "collector")}
    context["town"] = episode.metadata["config"]["environment"]["town"]
    digest.update(json.dumps(context, sort_keys=True).encode())
    for key, value in sorted(episode.arrays.items()):
        digest.update(json.dumps([key, str(value.dtype), value.shape]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def merge_datasets(cfg, destination, sources):
    """Append unique real recordings, preserving source splits and provenance.

    All sources are checked before the first write. Re-running is idempotent.
    Source files are never changed; conflicting numeric IDs receive new IDs whose
    seeded split equals the original. Evaluation traces remain forbidden.
    """
    target_path = Path(destination)
    existing = Dataset(target_path, cfg).load() if (target_path / "dataset.json").exists() else []
    fingerprints = {episode_fingerprint(ep) for ep in existing}
    seed_splits = {}
    def check_seed(ep):
        key = (ep.metadata["config"]["environment"]["town"], ep.metadata["seed"])
        split = ep.metadata["split"]
        if seed_splits.setdefault(key, split) != split:
            raise ValueError(f"Scenario seed {key} occurs in multiple splits")
    for ep in existing:
        check_seed(ep)
    prepared, counts = [], []
    for source in sources:
        if Path(source).resolve() == target_path.resolve():
            raise ValueError("Source and destination datasets must differ")
        episodes = Dataset(source, cfg).load()
        count = {"source": str(source), "episodes": len(episodes), "imported": 0, "duplicates": 0}
        for ep in episodes:
            ep.validate()
            check_seed(ep)
            fingerprint = episode_fingerprint(ep)
            if fingerprint in fingerprints:
                count["duplicates"] += 1
                continue
            fingerprints.add(fingerprint)
            prepared.append((source, ep))
            count["imported"] += 1
        counts.append(count)
    target = Dataset(destination, cfg, create=True)
    for source, ep in prepared:
        metadata = copy.deepcopy(ep.metadata)
        original_id = metadata["episode_id"]
        identity = target.next_id()
        while split_for(identity, cfg.training.seed) != metadata["split"]:
            identity += 1
        metadata["episode_id"] = identity
        metadata["imported_from"] = {"dataset": str(source), "episode_id": original_id,
                                     "split": metadata["split"], "fingerprint": episode_fingerprint(ep)}
        if "summary" in metadata:
            metadata["summary"]["episode_id"] = identity
        target.save(Episode(ep.arrays, metadata))
    report = {"destination": str(destination), "sources": counts,
              "imported": len(prepared), "episodes": len(target.paths()),
              "next_episode_id": target.next_id(), "source_splits_preserved": True}
    append_json(target.path / "imports.jsonl", report)
    return report


def sample_sequences(episodes, batch_size, length, rng, failure_fraction=0.0, reward_cfg=None):
    if not episodes:
        raise ValueError("No training episodes")
    length = min(length, max(len(ep) for ep in episodes))
    batch = []
    # The existing configuration name is retained, but every task ending needs coverage:
    # successes are positive termination targets too. Truncations are not.
    endings = [ep for ep in episodes if ep.arrays["terminated"][-1]]
    for _ in range(batch_size):
        ending = bool(endings) and rng.random() < failure_fraction
        pool = endings if ending else episodes
        ep = pool[int(rng.integers(len(pool)))]
        actual = min(length, len(ep))
        start = len(ep) - actual if ending else int(rng.integers(len(ep) - actual + 1))
        item = {}
        for key in (*OBS_KEYS, "applied_actions", "rewards", "terminated"):
            value = ep.arrays[key][start:start + actual + (1 if key in OBS_KEYS else 0)]
            if key == "rewards" and reward_cfg is not None:
                value = learning_rewards(ep.arrays, reward_cfg, start, start + actual)
            if actual < length:
                pad = (np.repeat(value[-1:], length - actual, axis=0) if key in OBS_KEYS else
                       np.zeros((length - actual, *value.shape[1:]), value.dtype))
                value = np.concatenate([value, pad])
            item[key] = value
        item["sequence_valid"] = np.arange(length) < actual
        batch.append(item)
    return {key: np.stack([item[key] for item in batch]) for key in batch[0]}

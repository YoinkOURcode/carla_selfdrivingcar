"""Fixed evaluation scenarios and episode-level acceptance metrics; no training."""

from collections import Counter
from pathlib import Path

import numpy as np


def save_evaluation_trace(episode, directory):
    from .data import write_json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episode.metadata.update(split="evaluation", usage="evaluation_only_never_train")
    path = directory / f"episode_{episode.metadata['episode_id']:06d}.npz"
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f"Refusing to overwrite evaluation trace {path}")
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **episode.arrays)
    write_json(path.with_suffix('.json'), episode.metadata)
    temporary.replace(path)


def evaluation_plan(cfg, episodes=50, seed_offset=1000000):
    return {"format": 1, "town": cfg.environment.town,
            "scenarios": [{"episode_id": i, "seed": cfg.training.seed + seed_offset + i,
                           "weather": cfg.environment.weather[i % len(cfg.environment.weather)]}
                          for i in range(episodes)]}


def validate_plan(plan, cfg):
    if plan.get("format") != 1 or plan.get("town") != cfg.environment.town:
        raise ValueError("Evaluation plan format/town differs from configuration")
    scenarios = plan.get("scenarios", [])
    if not scenarios:
        raise ValueError("Evaluation plan must contain scenarios")
    for key in ("episode_id", "seed"):
        values = [item[key] for item in scenarios]
        if any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Evaluation {key} values must be unique nonnegative integers")
    for item in scenarios:
        expected = cfg.environment.weather[item["episode_id"] % len(cfg.environment.weather)]
        if item["weather"] != expected:
            raise ValueError("Evaluation weather differs from the environment's episode schedule")
    return scenarios


def evaluation_metrics(summaries, plan, cfg):
    scenarios = validate_plan(plan, cfg)
    if len(summaries) > len(scenarios):
        raise ValueError("More evaluation results than planned scenarios")
    for result, scenario in zip(summaries, scenarios):
        if any(result[key] != scenario[key] for key in ("episode_id", "seed", "weather")):
            raise ValueError("Evaluation results do not match the fixed scenario order")
    def aggregate(items):
        def count(item, name):
            return item.get("event_totals", {}).get(name, item["event_decision_counts"].get(name, 0))
        violation_fields = ("collision", "red_light", "illegal_lane", "offroad",
                            "wrong_way", "unsafe_child", "unsafe_weather")
        collisions = sum(bool(count(item, "collision")) or
                         item["terminal_reason"] == "collision" for item in items)
        successes = sum(bool(item["success"]) for item in items)
        red_light_violations = sum(count(item, "red_light") for item in items)
        red_light_episodes = sum(bool(count(item, "red_light")) for item in items)
        clean_arrivals = sum(bool(item["success"]) and not any(
            count(item, key) for key in violation_fields) for item in items)
        return {"episodes": len(items), "successes": successes, "collision_episodes": collisions,
                "arrivals_without_recorded_violations": clean_arrivals,
                "monitored_violation_fields": list(violation_fields),
                "red_light_violations": red_light_violations, "red_light_episodes": red_light_episodes,
                "red_light_episode_rate": red_light_episodes / len(items) if items else None,
                "success_rate": successes / len(items) if items else None,
                "collision_episode_rate": collisions / len(items) if items else None,
                "mean_return": float(np.mean([item["return"] for item in items])) if items else None,
                "mean_route_completion": float(np.mean([item["route_completion"] for item in items])) if items else None}
    total = aggregate(summaries)
    weather_counts = Counter(item["weather"] for item in scenarios)
    complete = len(summaries) == len(scenarios)
    qualifies = (complete and len(scenarios) == 50 and len(cfg.environment.weather) == 5 and
                 len(set(cfg.environment.weather)) == 5 and
                 all(weather_counts[name] == 10 for name in cfg.environment.weather))
    return {**{key: value for key, value in total.items() if key != "episodes"},
            "completed_episodes": len(summaries), "planned_episodes": len(scenarios),
            "complete": complete,
            "per_weather": {name: aggregate([item for item in summaries if item["weather"] == name])
                            for name in cfg.environment.weather},
            "prototype_acceptance": {"eligible": qualifies,
                "required_success_rate": 0.8, "maximum_collision_episode_rate": 0.1,
                "passed": bool(qualifies and total["success_rate"] >= 0.8 and
                               total["collision_episode_rate"] <= 0.1)}}

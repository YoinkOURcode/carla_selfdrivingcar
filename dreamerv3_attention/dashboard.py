"""Summarize evaluation runs (or training metrics.jsonl) into per-episode JSON for dashboard.html.

    python -m dreamerv3_attention.dashboard                     # every run in runs/experiment_28/evaluation
    python -m dreamerv3_attention.dashboard runs/experiment_28/evaluation/eval24 runs/X/metrics.jsonl

A run is an evaluation directory (trace.jsonl, plus result.json when the run
finished) or a metrics.jsonl file. Each output file holds one run. Open
dashboard.html in a browser and load them.

To add a metric, decorate a function of an episode's steps with @metric (one
number) or @breakdown (a {category: number} dict). The dashboard builds its
tables, charts and pickers from the metadata written here.

``agg`` says how episodes combine in run totals and rolling trends:
  mean            mean of episode values (None values are skipped)
  sum             total over episodes
  per_km          episode values are counts; shown as total / total km
  share_of_steps  episode values are step counts; shown as % of all steps
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

TTC_THRESHOLD_S = 1.5
# CARLA Leaderboard infraction coefficients. Off-road distance is removed from
# route completion instead of being penalized.
PENALTY = {"pedestrian": .50, "vehicle": .60, "static": .65, "red_light": .70}
MAX_STEP_S = .5      # larger timestamp gaps are not treated as one step
HAZARD_JUMP_MPS = 40.  # faster range changes mean the front hazard switched objects
# CARLA's RSS sensor defaults (LibCarla/source/carla/rss/RssCheck.cpp): response time,
# ego acceleration during it, ego minimum braking, lead maximum braking.
RSS = {"response_s": 1.0, "accel_max": 3.5, "brake_min": 4.0, "brake_max": 8.0}
EVALUATIONS = Path("runs/experiment_28/evaluation")

METRICS, BREAKDOWNS = [], []


def metric(label, group, unit="", agg="mean", better="higher", note=""):
    def register(fn):
        METRICS.append({"key": fn.__name__, "label": label, "group": group, "unit": unit,
                        "agg": agg, "better": better, "note": note, "fn": fn})
        return fn
    return register


def breakdown(label, unit="", agg="mean", note=""):
    def register(fn):
        BREAKDOWNS.append({"key": fn.__name__, "label": label, "unit": unit,
                           "agg": agg, "note": note, "fn": fn})
        return fn
    return register


# ---------------------------------------------------------------- helpers

def rising(steps, test):
    """Indices where ``test(step)`` becomes true; a multi-step event counts once."""
    hits, previous = [], False
    for i, step in enumerate(steps):
        now = bool(test(step))
        if now and not previous:
            hits.append(i)
        previous = now
    return hits


def collision_kind(actor_type):
    if actor_type.startswith("walker."):
        return "pedestrian"
    if actor_type.startswith("vehicle."):
        return "vehicle"
    return "static"  # Leaderboard rule: anything else, fences included


def collision_events(steps):
    """[(kinds, assumed)] per collision event.

    Logs before ``collision_actor_types`` existed only flag pedestrian hits, so
    other collisions are assumed static (the common case in these runs).
    """
    events = []
    for i in rising(steps, lambda s: s.get("collision") or s.get("pedestrian_collision")):
        step = steps[i]
        types = step.get("collision_actor_types")
        if types:
            events.append(({collision_kind(t) for t in types}, False))
        elif step.get("pedestrian_collision"):
            events.append(({"pedestrian"}, False))
        else:
            events.append(({"static"}, "collision_actor_types" not in step))
    return events


def closing_speeds(steps):
    """Rate (m/s) at which the ground-truth front gap shrinks; None when unknown.

    Unknown on the first step, without a front hazard, or when the gap jumps
    because a different object became the front hazard.
    """
    out = [None]*len(steps)
    for i in range(1, len(steps)):
        d0, d1 = steps[i-1].get("front_hazard_m"), steps[i].get("front_hazard_m")
        dt = steps[i]["_dt"]
        if d0 is not None and d1 is not None and dt > 0 and abs(d1 - d0) <= HAZARD_JUMP_MPS*dt:
            out[i] = (d0 - d1)/dt
    return out


def rss_min_gap(v_rear, v_front, p=RSS):
    """RSS safe longitudinal distance (Shalev-Shwartz et al. 2017) for same-direction travel."""
    rho = p["response_s"]
    v_reacted = v_rear + rho*p["accel_max"]
    gap = (v_rear*rho + .5*p["accel_max"]*rho**2 + v_reacted**2/(2*p["brake_min"])
           - v_front**2/(2*p["brake_max"]))
    return max(gap, 0.)


def prepare(steps, threshold_s=TTC_THRESHOLD_S, rss=RSS):
    """Attach per-step duration, TTC and RSS margin; shared by all metric functions."""
    stamps = [s.get("timestamp") for s in steps]
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if a is not None and b is not None and 0 < b - a <= MAX_STEP_S]
    typical = sorted(gaps)[len(gaps)//2] if gaps else .05
    for i, step in enumerate(steps):
        gap = stamps[i] - stamps[i-1] if i and None not in (stamps[i], stamps[i-1]) else typical
        step["_dt"] = gap if 0 < gap <= MAX_STEP_S else typical
    for step, closing in zip(steps, closing_speeds(steps)):
        gap = step.get("front_hazard_m")
        step["_ttc"] = max(gap, 0.)/closing if closing is not None and closing > .1 else None
        step["_ttc_low"] = step["_ttc"] is not None and step["_ttc"] < threshold_s
        if closing is None:
            step["_rss_margin"] = None
        else:
            ego = max(step.get("speed_mps", 0.), 0.)
            step["_rss_margin"] = gap - rss_min_gap(ego, max(ego - closing, 0.), rss)
    return steps


def distance_km(steps):
    return sum(max(s.get("speed_mps", 0.), 0.)*s["_dt"] for s in steps)/1000.


# ---------------------------------------------------------------- driving score

@metric("Driving score", "Driving score", "%", note="Route completion × infraction penalty, CARLA Leaderboard style.")
def driving_score(steps):
    return route_completion(steps)*infraction_penalty(steps)


@metric("Route completion", "Driving score", "%", note="Furthest route progress (100% on success) minus progress made off-road.")
def route_completion(steps):
    length = steps[-1].get("route_length_m") or 0.
    if length <= 0:
        return 0.
    offroad, previous = 0., steps[0].get("route_progress_m", 0.)
    for step in steps[1:]:
        progress = step.get("route_progress_m", previous)
        if step.get("offroad"):
            offroad += max(progress - previous, 0.)
        previous = progress
    # The environment declares success a few metres before the route end.
    furthest = length if steps[-1].get("success") else max(s.get("route_progress_m", 0.) for s in steps)
    return 100.*min(max((furthest - offroad)/length, 0.), 1.)


@metric("Infraction penalty", "Driving score", "×",
        note="Product of 0.50 per pedestrian, 0.60 per vehicle, 0.65 per static collision and 0.70 per red light.")
def infraction_penalty(steps):
    penalty = PENALTY["red_light"]**len(rising(steps, lambda s: s.get("red_light_violation")))
    for kinds, _ in collision_events(steps):
        for kind in kinds:
            penalty *= PENALTY[kind]
    return penalty


@metric("Success", "Driving score", "%")
def success(steps):
    return 100. if steps[-1].get("success") else 0.


# ---------------------------------------------------------------- infractions

def _count_collisions(steps, kind):
    return sum(kind in kinds for kinds, _ in collision_events(steps))


@metric("Static collisions", "Infractions", "/km", "per_km", "lower", note="Fences, poles, buildings and props.")
def collisions_static(steps):
    return _count_collisions(steps, "static")


@metric("Vehicle collisions", "Infractions", "/km", "per_km", "lower")
def collisions_vehicle(steps):
    return _count_collisions(steps, "vehicle")


@metric("Pedestrian collisions", "Infractions", "/km", "per_km", "lower")
def collisions_pedestrian(steps):
    return _count_collisions(steps, "pedestrian")


@metric("Collisions with assumed type", "Infractions", "", "sum", "lower",
        note="Logged before collision_actor_types existed; counted as static.")
def collisions_type_assumed(steps):
    return sum(assumed for _, assumed in collision_events(steps))


@metric("Red-light violations", "Infractions", "/km", "per_km", "lower")
def red_light_violations(steps):
    return len(rising(steps, lambda s: s.get("red_light_violation")))


@metric("Crosswalk violations", "Infractions", "/km", "per_km", "lower", note="Not part of the driving score.")
def crossing_violations(steps):
    return len(rising(steps, lambda s: s.get("crossing_violation")))


@metric("Lane departures", "Infractions", "/km", "per_km", "lower", note="Not part of the driving score.")
def lane_departures(steps):
    return len(rising(steps, lambda s: s.get("lane_departure")))


@metric("Off-road events", "Infractions", "/km", "per_km", "lower")
def offroad_events(steps):
    return len(rising(steps, lambda s: s.get("offroad")))


# ---------------------------------------------------------------- surrogate safety

@metric("Minimum TTC", "Time to collision", "s", "mean", "higher",
        note="Lowest TTC in the episode; episodes that never close on a hazard are skipped.")
def min_ttc(steps):
    values = [s["_ttc"] for s in steps if s["_ttc"] is not None]
    return min(values) if values else None


@metric("TET", "Time to collision", "s/km", "per_km", "lower",
        note="Seconds spent with TTC under the threshold.")
def tet(steps):
    return sum(s["_dt"] for s in steps if s["_ttc_low"])


@metric("Low-TTC events", "Time to collision", "/km", "per_km", "lower",
        note="Times TTC dropped under the threshold.")
def ttc_events(steps):
    return len(rising(steps, lambda s: s["_ttc_low"]))


@metric("Time closer than RSS distance", "Safe distance (RSS)", "% steps", "share_of_steps", "lower",
        note="Steps where the front gap is under the RSS safe distance: the ego could not stop "
             "if the lead braked as hard as allowed.")
def rss_unsafe_steps(steps):
    return sum(s["_rss_margin"] is not None and s["_rss_margin"] < 0 for s in steps)


@metric("RSS violations", "Safe distance (RSS)", "/km", "per_km", "lower",
        note="Times the front gap fell under the RSS safe distance.")
def rss_violations(steps):
    return len(rising(steps, lambda s: s["_rss_margin"] is not None and s["_rss_margin"] < 0))


@metric("Worst RSS margin", "Safe distance (RSS)", "m", "mean", "higher",
        note="Smallest front gap minus RSS safe distance in the episode; negative means unsafe. "
             "Episodes without a front hazard are skipped.")
def rss_min_margin(steps):
    values = [s["_rss_margin"] for s in steps if s["_rss_margin"] is not None]
    return min(values) if values else None


# ---------------------------------------------------------------- safety supervisor

@metric("Steps with an intervention", "Safety supervisor", "% steps", "share_of_steps", "lower")
def intervention_steps(steps):
    return sum(bool(s.get("interventions")) for s in steps)


@metric("Intervention events", "Safety supervisor", "/km", "per_km", "lower",
        note="Times any intervention started.")
def intervention_events(steps):
    return len(rising(steps, lambda s: s.get("interventions")))


@breakdown("Intervention share by type", "% steps", "share_of_steps")
def intervention_types(steps):
    return dict(Counter(r for s in steps for r in set(s.get("interventions") or [])))


@breakdown("Intervention events by type", "/km", "per_km")
def intervention_type_events(steps):
    reasons = {r for s in steps for r in (s.get("interventions") or [])}
    return {r: len(rising(steps, lambda s, r=r: r in (s.get("interventions") or []))) for r in reasons}


# ---------------------------------------------------------------- reward and episode

@breakdown("Reward components", "per episode", "mean", note="Sum of each component over the episode.")
def reward_components(steps):
    totals = Counter()
    for step in steps:
        totals.update({k: float(v) for k, v in (step.get("reward_components") or {}).items()})
    return dict(totals)


@breakdown("Episode endings", "% episodes", "mean")
def episode_end(steps):
    last = steps[-1]
    reason = ("success" if last.get("success") else last.get("termination_reason")
              or last.get("truncation_reason") or "unknown")
    return {reason: 100.}


@metric("Episode return", "Episode", "")
def episode_return(steps):
    # Evaluation traces before 2026-11-24 have no reward field, only its components.
    return sum(s["reward"] if "reward" in s else sum((s.get("reward_components") or {}).values())
               for s in steps)


@metric("Mean speed", "Episode", "m/s")
def mean_speed(steps):
    duration = sum(s["_dt"] for s in steps)
    return distance_km(steps)*1000./duration if duration else 0.


@metric("Episode length", "Episode", "s", better="none")
def duration_s(steps):
    return sum(s["_dt"] for s in steps)


# ---------------------------------------------------------------- evaluation report
# Read from the evaluation's result.json (evaluation.py), so training runs leave them blank.

REPORT = [  # (result.json key, label, unit, agg, better, note, scale)
    ("speed_violation_events", "Speeding events", "/km", "per_km", "lower",
     "Times speed rose above the posted limit."),
    ("speed_excess_mean_mps", "Speed over the limit", "m/s", "mean", "lower",
     "Average excess while speeding."),
    ("unnecessary_stop_events", "Unnecessary stops", "/km", "per_km", "lower",
     "Stops (under 0.5 m/s) with no red light, pedestrian conflict or crossing ahead. "
     "Queueing behind a vehicle counts."),
    ("unnecessary_stopping_s", "Unnecessary stopped time", "s", "mean", "lower", ""),
    ("pedestrian_conflict_events", "Pedestrian conflicts", "/km", "per_km", "lower",
     "Times a pedestrian entered the conflict zone ahead."),
    ("minimum_relevant_separation_m", "Minimum separation", "m", "mean", "higher",
     "Smallest front, side, rear or pedestrian gap in the episode."),
    ("speed_tracking_error_mean_mps", "Speed tracking error", "m/s", "mean", "lower",
     "Mean |speed − supervisor target speed|."),
    ("acceleration_rms_mps2", "Acceleration RMS", "m/s²", "mean", "lower", ""),
    ("jerk_rms_mps3", "Jerk RMS", "m/s³", "mean", "lower", ""),
    ("policy_inference_latency_p95_s", "Policy latency (p95)", "ms", "mean", "lower", "", 1000.),
]
for _key, _label, _unit, _agg, _better, _note, *_scale in REPORT:
    def _read(steps, key=_key, scale=(_scale or [1.])[0]):
        value = (steps[0].get("_report") or {}).get(key)
        return None if value is None else value*scale
    _read.__name__ = f"report_{_key}"
    metric(_label, "Evaluation report", _unit, _agg, _better, _note)(_read)


# ---------------------------------------------------------------- summarizing

def summarize_episode(steps, threshold_s=TTC_THRESHOLD_S, rss=RSS):
    prepare(steps, threshold_s, rss)
    first = steps[0]
    return {
        "info": {"episode": first.get("episode"), "frames": first.get("frames"),
                 "source": first.get("action_source", "unknown"),
                 "scenario": first.get("scenario", "unknown"), "weather": first.get("weather", "unknown"),
                 "perception": first.get("perception_source", "unknown"),
                 "steps": len(steps), "km": distance_km(steps)},
        "m": {m["key"]: _clean(m["fn"](steps)) for m in METRICS},
        "b": {b["key"]: {k: _clean(v) for k, v in b["fn"](steps).items()} for b in BREAKDOWNS},
    }


def _clean(value):
    if value is None or isinstance(value, bool):
        return value
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


def episode_ended(step):
    if "terminated" in step:
        return step["terminated"] or step.get("truncated")
    # Evaluation traces before 2026-11-24 carry only the reasons.
    return bool(step.get("termination_reason") or step.get("truncation_reason") or step.get("success"))


def summarize(path, name=None, threshold_s=TTC_THRESHOLD_S, rss=RSS):
    """Stream one run; only complete episodes are kept.

    ``path`` is an evaluation directory or a metrics.jsonl/trace.jsonl file.
    """
    path = Path(path)
    if path.is_dir():
        path = path/"trace.jsonl" if (path/"trace.jsonl").exists() else path/"metrics.jsonl"
    result_path = path.parent/"result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    reports = result.get("episodes", [])
    rows, steps, incomplete = [], [], 0
    with path.open() as lines:
        for line in lines:
            if not line.strip():
                continue
            step = json.loads(line)
            if steps and step.get("episode") != steps[-1].get("episode"):
                incomplete += 1  # no terminal step: the run stopped or resumed mid-episode
                steps = []
            step.setdefault("action_source", (result.get("policy") or {}).get("type", "unknown"))
            steps.append(step)
            if episode_ended(step):
                report = reports[len(rows)] if len(rows) < len(reports) else None
                if report and report.get("name") == step.get("scenario"):
                    steps[0]["_report"] = report.get("metrics")
                rows.append(summarize_episode(steps, threshold_s, rss))
                steps = []
    checkpoint = result.get("checkpoint")
    strip = lambda d: {k: v for k, v in d.items() if k != "fn"}
    return {"name": name or f"{path.parent.parent.name}/{path.parent.name}", "path": str(path),
            "kind": "evaluation" if path.name == "trace.jsonl" else "training",
            "checkpoint": "/".join(Path(checkpoint).parts[-3:-1]) if checkpoint else None,
            "ttc_threshold_s": threshold_s, "rss": rss, "incomplete_episodes": incomplete + bool(steps),
            "penalties": PENALTY, "metrics": [strip(m) for m in METRICS],
            "breakdowns": [strip(b) for b in BREAKDOWNS], "episodes": rows}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("paths", nargs="*", type=Path,
                     help=f"evaluation directories or metrics.jsonl files (default: every run in {EVALUATIONS})")
    cli.add_argument("-o", "--out", type=Path, default=Path("runs/dashboard"), help="output directory")
    cli.add_argument("--ttc-threshold", type=float, default=TTC_THRESHOLD_S, help="TET threshold in seconds")
    for key, value in RSS.items():
        cli.add_argument(f"--rss-{key.replace('_', '-')}", type=float, default=value,
                         help=f"RSS {key.replace('_', ' ')} (default {value})")
    args = cli.parse_args(argv)
    rss = {key: getattr(args, f"rss_{key}") for key in RSS}
    args.out.mkdir(parents=True, exist_ok=True)
    paths = args.paths or sorted(p for p in EVALUATIONS.iterdir() if (p/"trace.jsonl").exists())
    for path in paths:
        summary = summarize(path, threshold_s=args.ttc_threshold, rss=rss)
        target = args.out/(summary["name"].replace("/", "__") + ".json")
        target.write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False))
        print(f"{target}: {len(summary['episodes'])} episodes, {summary['incomplete_episodes']} incomplete skipped")


if __name__ == "__main__":
    main()

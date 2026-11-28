"""Build the Town02 evaluation summary (markdown) from every per-scenario run directory."""
import json, os, glob, collections, statistics as st
H = os.path.expanduser
REPO = H("~/carla_selfdrivingcar/")
dirs = [open(H("~/demo/reeval_dir")).read().strip()] + open(H("~/demo/repeat_dirs")).read().split()
base = {e["name"]: e for e in json.load(open(H("~/demo/baseline_result.json")))["episodes"]}
NAME = "held_out_long_mixed_%d"
def load(d, n):
    p = f"{REPO}{d}/{NAME % n}"
    if not os.path.exists(p + "/result.json"):
        prog = open(f"{REPO}{d}/progress.log").read() if os.path.exists(f"{REPO}{d}/progress.log") else ""
        return {"state": "crashed" if f"END {NAME % n} " in prog else ("running" if f"START {NAME % n}" in prog else "queued")}
    e = json.load(open(p + "/result.json"))["episodes"][0]
    src, err, last = collections.Counter(), [], None
    for l in open(p + "/trace.jsonl"):
        last = json.loads(l); src[last.get("policy_lane_source")] += 1; err.append(abs(last["lane_error_m"]))
    return {"state": "done", "e": e, "m": e["metrics"], "reason": last.get("termination_reason") or last.get("truncation_reason"),
            "vision": src["vision"] / max(1, sum(src.values())), "err_mean": st.mean(err), "err_p95": sorted(err)[int(.95 * len(err))],
            "length": last["route_length_m"], "coll": [t for l in [last] for t in []]}
runs = {n: [load(d, n) for d in dirs] for n in range(1, 17)}
done = lambda n: [r for r in runs[n] if r["state"] == "done"]
out = []
P = out.append
total = sum(len(done(n)) for n in runs); ok = sum(r["e"]["success"] for n in runs for r in done(n))
P("# Town02 re-evaluation after the ground-truth patch\n")
P("Code: `deployment` at `cc5cff0da` (includes patch `47c77c22d`). Checkpoint: `final/latest.pt`, unchanged. Camera perception (`external`), held-out Town02, one scenario per process, on the Vast server.\n")
P(f"**Runs finished: {total}. Successful: {ok} ({100*ok/max(1,total):.0f}%).** Scenarios 1 to 6 are the original suite; 7 to 16 are new.\n")
P("## Result per scenario\n")
P("| # | Weather | Route m | Encounters | Before patch | " + " | ".join(f"Run {i+1}" for i in range(len(dirs))) + " | Passes |")
P("|---|---|---|---|---|" + "---|" * (len(dirs) + 1))
def cell(r):
    if r["state"] != "done": return r["state"]
    return "Pass" if r["e"]["success"] else f"Fail {100*r['m']['route_completion']:.0f}% ({r['reason']})"
for n in runs:
    d = done(n)
    sc = d[0]["e"]["scenario"] if d else None
    enc = "".join({"vehicle": "V", "adult": "A", "child": "C"}[x["kind"]] for x in sc["encounters"]) if sc else "-"
    b = base.get(NAME % n)
    P(f"| {n} | {sc['weather'] if sc else '-'} | {d[0]['length']:.0f} | {enc} | " + (("Pass" if b["success"] else "Fail") if b else "new") + " | " +
      " | ".join(cell(r) for r in runs[n]) + f" | {sum(r['e']['success'] for r in d)} of {len(d)} |" if d else f"| {n} | - | - | - | " + ("Pass" if b else "new") + " | " + " | ".join(cell(r) for r in runs[n]) + " | - |")
P("\nEncounters: V = slow lead vehicle, A = adult crossing, C = child crossing, in route order.\n")
P("## Metrics per run\n")
P("| # | Run | Result | Route done | Distance m | Sim time s | Collisions | Pedestrian incidents | Red-light violations | Lane departures | Speeding s | Min separation m | Supervisor override | Lane from camera | Mean lane error m | p95 lane error m |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in runs:
    for i, r in enumerate(runs[n]):
        if r["state"] != "done": continue
        m = r["m"]
        P(f"| {n} | {i+1} | {'Pass' if r['e']['success'] else 'Fail: ' + str(r['reason'])} | {100*m['route_completion']:.1f}% | {m['distance_traveled_m']:.0f} | {m['duration_s']:.0f} | {m['collision_events']} | {m['pedestrian_incidents']} | {m['red_light_violations']} | {m['lane_departures']} | {m['speed_violation_duration_s']:.1f} | {m['minimum_relevant_separation_m']:.2f} | {100*m['action_override_fraction']:.0f}% | {100*r['vision']:.0f}% | {r['err_mean']:.2f} | {r['err_p95']:.2f} |")
allr = [r for n in runs for r in done(n)]
if allr:
    P("\n## Totals over finished runs\n")
    P("| Measure | Value |\n|---|---|")
    P(f"| Runs | {len(allr)} |")
    P(f"| Successful | {ok} ({100*ok/len(allr):.0f}%) |")
    for grp, rng in (("Original scenarios 1 to 6", range(1, 7)), ("New scenarios 7 to 16", range(7, 17))):
        g = [r for n in rng for r in done(n)]
        if g: P(f"| {grp} | {sum(r['e']['success'] for r in g)} of {len(g)} runs passed |")
    P(f"| Mean route completion | {100*st.mean(r['m']['route_completion'] for r in allr):.1f}% |")
    for k, label in (("collision_events", "Collisions"), ("pedestrian_incidents", "Pedestrian incidents"), ("red_light_violations", "Red-light violations"), ("lane_departures", "Lane departures")):
        P(f"| {label} | {sum(r['m'][k] for r in allr)} |")
    P(f"| Speeding, total | {sum(r['m']['speed_violation_duration_s'] for r in allr):.1f} s |")
    P(f"| Supervisor override, mean | {100*st.mean(r['m']['action_override_fraction'] for r in allr):.0f}% of steps |")
    P(f"| Lane signal from the camera road model, mean | {100*st.mean(r['vision'] for r in allr):.0f}% of steps |")
    ends = collections.Counter(r["reason"] for r in allr if not r["e"]["success"])
    P(f"| Failure endings | {', '.join(f'{k}: {v}' for k, v in ends.items()) or 'none'} |")
if base:
    P("\n## Before the patch (team's earlier report, scenarios 1 to 6)\n")
    P("| # | Result | Route done | Collisions | Lane departures | Supervisor override |\n|---|---|---|---|---|---|")
    for n in range(1, 7):
        b = base[NAME % n]; m = b["metrics"]
        P(f"| {n} | {'Pass' if b['success'] else 'Fail'} | {100*m['route_completion']:.1f}% | {m['collision_events']} | {m['lane_departures']} | {100*m.get('action_override_fraction', 0):.0f}% |")
open(H("~/demo/evaluation_summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

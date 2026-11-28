"""Pick new Town02 held-out scenarios: try seeds, keep those whose route is new and whose encounters can be placed."""
import json, sys, traceback
from dataclasses import replace
from pathlib import Path
from dreamerv3_attention.config import Config
from dreamerv3_attention.environment import CarlaEnv

WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CATALOG = Path("runs/experiment_28/evaluation/scenarios.json")
OUT = Path.home() / "demo/new_scenarios.json"
V = lambda t, r=20: {"kind": "vehicle", "trigger_m": t, "release_s": r}
A = lambda t: {"kind": "adult", "trigger_m": t}
C = lambda t: {"kind": "child", "trigger_m": t}
DESIGNS = [  # weather fields, encounters (ordered by trigger)
    ({"weather": "ClearSunset"}, [V(50, 15), C(120), A(190), C(260), V(310, 15)]),
    ({"weather": "CloudyNoon"}, [C(60), C(130), A(200), V(260), C(345)]),
    ({"weather": "WetNoon", "wetness": 60}, [A(70), V(130), C(200), A(270), C(340)]),
    ({"weather": "MidRainyNoon", "wetness": 75}, [V(45, 25), A(110), C(170), V(240, 15), A(330)]),
    ({"weather": "CloudySunset"}, [C(80), A(140), C(200), A(260), C(320)]),
    ({"weather": "SoftRainSunset", "wetness": 70}, [V(60), C(125), A(185), C(250), V(305)]),
    ({"weather": "WetCloudyNoon", "wetness": 50, "fog_density": 25, "fog_distance": 20}, [A(55), C(115), V(180, 25), A(250), C(350)]),
    ({"weather": "ClearNight", "sun_altitude_angle": -30}, [V(55), A(120), C(190), A(280)]),
    ({"weather": "WetNight", "wetness": 60, "sun_altitude_angle": -25}, [C(90), V(150), A(220), C(300)]),
    ({"weather": "MidRainyNight", "wetness": 80, "sun_altitude_angle": -25, "fog_density": 20, "fog_distance": 25}, [A(65), C(130), V(200, 15), C(270), A(340)]),
]
cfg = Config.load("final/evaluation.json").resolved()
existing = json.loads(CATALOG.read_text())["scenarios"]
template = {k: existing[0][k] for k in ("category", "split", "minimum_route_length_m", "maximum_route_length_m",
                                        "requires_traffic_light", "background_traffic", "actors", "randomize_routes")}

def probe(scenario):
    env = CarlaEnv(replace(cfg, seed=scenario["seed"]), scenario)
    try:
        env.reset()
        return (int(env.route_start), int(env.route_end)), float(env.route.length), len(env.demo_pending)
    finally:
        env.close()

seen = {}
for s in existing:
    route, length, placed = probe(s)
    seen[route] = s["name"]
    print("existing", s["name"], route, round(length), "m", placed, "encounters", flush=True)
new, seed, number = [], max(s["seed"] for s in existing) + 1, len(existing) + 1
while len(new) < WANT and seed < 380200:
    weather, encounters = DESIGNS[len(new) % len(DESIGNS)]
    scenario = {**template, "seed": seed, "route_start": 2 * number, "route_end": 2 * number + 17,
                "name": f"held_out_long_mixed_{number}", **weather, "encounters": encounters}
    try:
        route, length, placed = probe(scenario)
    except Exception as exc:
        print("seed", seed, "rejected:", type(exc).__name__, str(exc)[:120], flush=True)
        seed += 1
        continue
    clash = seen.get(route) or seen.get(route[::-1])
    if clash or placed != len(encounters):
        print("seed", seed, "rejected: route", route, "same as" if clash else "encounters placed", clash or placed, flush=True)
        seed += 1
        continue
    seen[route] = scenario["name"]
    new.append(scenario)
    print("ACCEPT", scenario["name"], "seed", seed, route, round(length), "m", weather["weather"],
          [e["kind"] for e in encounters], flush=True)
    seed += 1
    number += 1
    OUT.write_text(json.dumps(new, indent=1) + "\n")
print("accepted", len(new))

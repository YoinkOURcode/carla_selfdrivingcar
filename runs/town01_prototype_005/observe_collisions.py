"""Passive collision provenance for the running collection; never ticks or controls CARLA."""

import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import carla
from world_model_rl.config import Config


root = Path(__file__).resolve().parent
cfg = Config.load(root / "config.json")
client = carla.Client(cfg.environment.host, cfg.environment.port)
client.set_timeout(5)
world = client.get_world()
lock = threading.Lock()
output = root / "development_collision_observations.jsonl"
result = root / "development_evaluation.json"
sensor = None
ego_id = None
deadline = time.monotonic() + 1200


def receive(event, identity):
    impulse = event.normal_impulse
    row = {"frame": event.frame, "ego_id": identity,
           "other_id": event.other_actor.id, "other_type": event.other_actor.type_id,
           "other_role": event.other_actor.attributes.get("role_name", ""),
           "normal_impulse": [impulse.x, impulse.y, impulse.z],
           "ego_position": [event.transform.location.x, event.transform.location.y]}
    with lock:
        with output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


try:
    while time.monotonic() < deadline:
        if result.exists() and json.loads(result.read_text()).get("complete"):
            break
        actors = [actor for actor in world.get_actors().filter("vehicle.*")
                  if actor.attributes.get("role_name") == "world_model_rl"]
        if len(actors) > 1:
            raise RuntimeError("More than one driving ego exists; observer will not guess")
        ego = actors[0] if actors else None
        identity = ego.id if ego else None
        if identity != ego_id:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
            sensor, ego_id = None, identity
            if ego is not None:
                sensor = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                           carla.Transform(), attach_to=ego)
                sensor.listen(lambda event, identity=identity: receive(event, identity))
                print(f"Observing ego {identity}", flush=True)
        time.sleep(.25)
finally:
    if sensor is not None and sensor.is_alive:
        sensor.stop()
        sensor.destroy()

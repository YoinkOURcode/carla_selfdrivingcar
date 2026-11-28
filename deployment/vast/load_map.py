"""Wait for the CARLA server, clear any leftover session, and load the configured town."""
import json, sys, time
import carla
town = json.load(open("final/viewer.json"))["town"]
for attempt in range(60):
    try:
        client = carla.Client("127.0.0.1", 2000); client.set_timeout(60)
        world = client.get_world()
        settings = world.get_settings()
        if settings.synchronous_mode:  # left behind by a run that was killed mid-drive
            settings.synchronous_mode = False; settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        client.load_world(town)  # also removes leftover actors
        if client.get_world().get_map().name.endswith(town):
            print("map ready:", town); sys.exit(0)
    except RuntimeError as error:
        print("waiting for CARLA:", str(error)[:80], flush=True)
    time.sleep(5)
sys.exit(1)

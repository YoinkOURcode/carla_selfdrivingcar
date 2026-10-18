#!/usr/bin/env python3
"""Collect unlabeled CARLA RGB images with balanced annotation assignments.

CARLA is imported only for inspect/collect; plan/verify run with the standard library.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import queue
import random
import struct
import sys
import time
import zipfile
import zlib


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_config(path):
    cfg = json.loads(path.read_text())
    if len(cfg["proportions"]) != 5 or cfg["proportions"] != [0.1, 0.1, 0.2, 0.2, 0.4]:
        raise ValueError("This assignment requires proportions [0.1, 0.1, 0.2, 0.2, 0.4]")
    for key in ("total_images", "images_per_episode", "width", "height"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["images_per_episode"] % 10 or cfg["total_images"] % cfg["images_per_episode"]:
        raise ValueError("Episode size must be divisible by 10 and divide total_images")
    for key in ("towns", "weather_presets", "ego_models", "vehicle_counts", "pedestrian_counts"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    for key in ("child_fraction", "pedestrian_crossing_fraction"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} must be between zero and one")
    dt = cfg["fixed_delta_seconds"]
    if not 0 < dt <= 0.1:
        raise ValueError("fixed_delta_seconds must be in (0, 0.1]")
    interval = cfg["capture_interval_seconds"]
    if interval <= 0 or not math.isclose(interval / dt, round(interval / dt)):
        raise ValueError("Capture interval must be a positive integer multiple of fixed delta")
    if not 0 < cfg["fov"] < 180:
        raise ValueError("fov must be in (0, 180)")
    for key in ("stationary_capture_seconds", "stuck_seconds", "max_episode_seconds", "pedestrian_radius"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("traffic_light_green_seconds", "traffic_light_yellow_seconds", "traffic_light_red_seconds"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg["warmup_seconds"] < 0 or min(cfg["vehicle_counts"]) < 1 or min(cfg["pedestrian_counts"]) < 2:
        raise ValueError("Invalid warmup or actor counts")
    return cfg


def make_plan(cfg):
    count = cfg["total_images"] // cfg["images_per_episode"]
    n_towns, n_weather = len(cfg["towns"]), len(cfg["weather_presets"])
    if count < n_towns * n_weather:
        raise ValueError("Need at least one episode per town/weather pair")
    episodes = []
    # Round-robin covers the full Cartesian product before repeating it.
    for i in range(count):
        town = i % n_towns
        weather = (i // n_towns) % n_weather
        repeat = i // (n_towns * n_weather)
        episodes.append({
            "id": f"episode_{i:04d}", "town": cfg["towns"][town],
            "weather": cfg["weather_presets"][weather],
            "ego_model": cfg["ego_models"][(town + weather + repeat) % len(cfg["ego_models"])],
            "vehicles": cfg["vehicle_counts"][(town + weather + repeat) % len(cfg["vehicle_counts"])],
            "pedestrians": cfg["pedestrian_counts"][(town + weather + repeat) % len(cfg["pedestrian_counts"])],
            "focus": ["traffic_lights", "speed_limit", "pedestrians"][(town + weather + repeat) % 3],
            "seed": cfg["seed"] + i, "images": cfg["images_per_episode"]})
    return {"schema": 1, "config": cfg, "episodes": episodes}


def assignments(ep):
    rng = random.Random(ep["seed"] ^ 0x5A17)
    result = []
    for _ in range(ep["images"] // 10):
        block = [1, 2, 3, 3, 4, 4, 5, 5, 5, 5]
        rng.shuffle(block)
        result.extend(block)
    return result


def image_path(output, ep, index, people=None):
    people = people if people is not None else assignments(ep)
    return output / f"person_{people[index]}" / ep["id"] / f"{index:06d}.png"


def valid_png(path, cfg, full=False):
    # Fast integrity check without an imaging dependency. CARLA writes PNG atomically below.
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
            stream.seek(-12, 2)
            tail = stream.read()
        valid = (header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR"
                and struct.unpack(">II", header[16:24]) == (cfg["width"], cfg["height"])
                and tail == b"\x00\x00\x00\x00IEND\xaeB`\x82")
        if not valid or not full:
            return valid
        data = path.read_bytes()
        position, compressed = 8, bytearray()
        channels = None
        while position < len(data):
            length = struct.unpack(">I", data[position:position + 4])[0]
            kind = data[position + 4:position + 8]
            body = data[position + 8:position + 8 + length]
            checksum = struct.unpack(">I", data[position + 8 + length:position + 12 + length])[0]
            if zlib.crc32(kind + body) != checksum:
                return False
            if kind == b"IHDR":
                width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
                channels = {2: 3, 6: 4}.get(color)
                if depth != 8 or channels is None or compression or filtering or interlace:
                    return False  # CARLA RGB output is noninterlaced 8-bit RGB/RGBA.
            elif kind == b"IDAT":
                compressed.extend(body)
            position += 12 + length
        if position != len(data) or channels is None:
            return False
        row_size = cfg["width"] * channels + 1
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(bytes(compressed), row_size * cfg["height"] + 1)
        return (decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
                and len(pixels) == row_size * cfg["height"]
                and all(pixels[row * row_size] <= 4 for row in range(cfg["height"])))
    except (OSError, struct.error, zlib.error):
        return False


def verify(output, plan, completed_only=False, full_png=False):
    all_episodes = plan["episodes"]
    episodes = all_episodes
    if completed_only:
        episodes = [ep for ep in all_episodes if (output / "episodes" / (ep["id"] + ".json")).exists()]
        if not episodes:
            raise RuntimeError("No completed episodes to verify")
    totals = Counter()
    missing = []
    expected = set()
    for ep in episodes:
        people = assignments(ep)
        for i, person in enumerate(people):
            path = image_path(output, ep, i, people)
            expected.add(path)
            if valid_png(path, plan["config"], full=full_png):
                totals[person] += 1
            else:
                missing.append(str(path))
    actual = set(output.glob("person_*/*/*.png"))
    if completed_only:
        actual = {path for path in actual if path.parent.name in {ep["id"] for ep in episodes}}
    extra = actual - expected
    incomplete = []
    for ep in episodes:
        marker = output / "episodes" / (ep["id"] + ".json")
        try:
            record = json.loads(marker.read_text())
            if any(record.get(key) != value for key, value in ep.items()):
                incomplete.append(ep["id"])
        except (OSError, ValueError):
            incomplete.append(ep["id"])
    bad_manifests = []
    for person in range(1, 6):
        intended = [str(image_path(output, ep, i, people).relative_to(output))
                    for ep in plan["episodes"] for people in [assignments(ep)]
                    for i, owner in enumerate(people) if owner == person]
        try:
            actual = (output / f"person_{person}" / "images.txt").read_text().splitlines()
        except OSError:
            actual = []
        if actual != intended:
            bad_manifests.append(person)
    report = {"planned_images": plan["config"]["total_images"],
              "scope": "completed_episodes" if completed_only else "full_plan",
              "png_validation": "chunk_crc_and_decompressed_scanlines" if full_png else "header_dimensions_end_marker",
              "verified_episodes": [ep["id"] for ep in episodes],
              "expected_images": sum(ep["images"] for ep in episodes),
              "valid_images": sum(totals.values()), "per_person": dict(sorted(totals.items())),
              "missing_or_invalid": len(missing), "unexpected_images": len(extra),
              "episodes_without_completion_record": incomplete,
              "invalid_assignment_manifests": bad_manifests,
              "missing_examples": missing[:10], "unexpected_examples": sorted(map(str, extra))[:10]}
    write_json(output / ("verification_completed.json" if completed_only else "verification.json"), report)
    print(json.dumps(report, indent=2))
    if missing or extra or incomplete or bad_manifests:
        raise RuntimeError("Dataset is incomplete or inconsistent; see verification.json")


def export_completed(output, plan, archive):
    verify(output, plan, completed_only=True, full_png=True)
    episodes = [ep for ep in plan["episodes"] if (output / "episodes" / (ep["id"] + ".json")).exists()]
    archive = archive.resolve()
    if archive.exists():
        raise ValueError(f"Archive already exists: {archive}; choose a different --archive path")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.write(output / "verification_completed.json", "verification_completed.json")
        bundle.write(output / "plan.json", "full_collection_plan.json")
        manifests = {person: [] for person in range(1, 6)}
        for ep in episodes:
            bundle.write(output / "episodes" / (ep["id"] + ".json"), "episodes/" + ep["id"] + ".json")
            owners = assignments(ep)
            for i, person in enumerate(owners):
                path = image_path(output, ep, i, owners)
                relative = str(path.relative_to(output))
                bundle.write(path, relative)
                manifests[person].append(relative)
        for person, paths in manifests.items():
            bundle.writestr(f"person_{person}/images.txt", "\n".join(paths) + "\n")
        bundle.writestr("README.txt", "Unlabeled CARLA RGB images from completed collection episodes.\n"
                        "images.txt paths are relative to the archive root and include only exported images.\n"
                        "full_collection_plan.json describes the larger planned collection; only episodes\n"
                        "listed in verification_completed.json are included here. No object annotations.\n")
    temporary.replace(archive)
    print(f"Exported {sum(ep['images'] for ep in episodes)} images to {archive}")


def connect(args):
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError("Install the CARLA Python client built from your server's checkout. See data/README.md") from exc
    client = carla.Client(args.host, args.port)
    # Cold source builds can compile thousands of shaders before the RPC world is ready.
    deadline = time.monotonic() + args.startup_timeout
    attempt = 0
    while True:
        if attempt:
            # A client connected before a server restart may retain a dead streaming connection.
            client = carla.Client(args.host, args.port)
        client.set_timeout(min(args.timeout, 5.0))
        try:
            sv = client.get_server_version()
            client.get_world().get_map()
            break
        except RuntimeError as exc:
            attempt += 1
            if time.monotonic() >= deadline:
                raise RuntimeError(f"CARLA not ready at {args.host}:{args.port} after {args.startup_timeout}s. "
                                   "Start data/start_server.sh, wait for map/shader loading, and use the matching "
                                   f"Python environment. Last error: {exc}") from exc
            if attempt == 1 or attempt % 6 == 0:
                print(f"Waiting for CARLA world at {args.host}:{args.port} (startup/shader compilation)...", flush=True)
            time.sleep(min(2, max(0, deadline - time.monotonic())))
    client.set_timeout(args.timeout)
    cv = client.get_client_version()
    if cv != sv:
        raise RuntimeError(f"Client/server versions differ: {cv!r} vs {sv!r}. Use a matching build.")
    return carla, client


def pedestrian_pools(library):
    walkers = sorted(library.filter("walker.pedestrian.*"), key=lambda bp: bp.id)
    children, adults = [], []
    # Older content packs omit age. IDs below come from Docs/catalogue_pedestrians.md.
    child_ids = {"0009", "0010", "0011", "0012", "0013", "0014", "0048", "0049"}
    for bp in walkers:
        ped_id = bp.id.rsplit(".", 1)[-1]
        age = bp.get_attribute("age").as_str().lower() if bp.has_attribute("age") else (
            "child" if ped_id in child_ids else "adult")
        if age == "child":
            children.append(bp)
        elif age in ("adult", "elderly"):
            adults.append(bp)
    if not children or not adults:
        raise RuntimeError("Need child and adult/elderly blueprints; install pedestrian assets before collection")
    return children, adults


def inspect(args):
    _, client = connect(args)
    world = client.get_world()
    children, adults = pedestrian_pools(world.get_blueprint_library())
    print(json.dumps({"version": client.get_server_version(), "current_map": world.get_map().name,
                      "maps": client.get_available_maps(),
                      "vehicles": sorted(bp.id for bp in world.get_blueprint_library().filter("vehicle.*")),
                      "child_models": [bp.id for bp in children], "adult_models": [bp.id for bp in adults]}, indent=2))


def weather_value(carla, name):
    if name == "ClearNight":
        return carla.WeatherParameters(cloudiness=10, precipitation=0, sun_altitude_angle=-30)
    value = getattr(carla.WeatherParameters, name, None)
    if value is None:
        raise ValueError(f"Weather preset unavailable: {name}")
    return value


def ranked_spawns(world, ep, rng):
    points = list(world.get_map().get_spawn_points())
    rng.shuffle(points)
    pattern = "traffic.speed_limit.*" if ep["focus"] == "speed_limit" else "traffic.traffic_light*"
    targets = [actor.get_location() for actor in world.get_actors().filter(pattern)]
    if targets:
        def score(point):
            loc, forward = point.location, point.get_forward_vector()
            scores = []
            for target in targets:
                delta = target - loc
                distance = loc.distance(target)
                facing = delta.x * forward.x + delta.y * forward.y
                scores.append(abs(distance - 35) + (100 if facing < 0 else 0))
            return min(scores) + rng.uniform(0, 15)
        # Do not use random calls inside sort's key again when resuming: seeded order is fixed.
        points.sort(key=score)
    return points


def stop_traffic_manager(tm):
    # Dev TM starts a worker when switched back to async. That worker can race
    # actor destruction/map reload and throw from a stale EpisodeProxy.
    # Join any async worker first, then remove this TM while its episode is valid.
    tm.set_synchronous_mode(True)
    tm.shut_down()

def capture_episode(carla, client, args, output, cfg, ep):
    rng = random.Random(ep["seed"])
    print(f"Loading {ep['town']} (map timeout {args.load_timeout}s)...", flush=True)
    client.set_timeout(args.load_timeout)
    try:
        world = client.load_world(ep["town"])
    finally:
        client.set_timeout(args.timeout)
    print("Map loaded; configuring traffic and pedestrians...", flush=True)
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    tm = client.get_trafficmanager(args.tm_port)
    actors, controllers = [], []
    camera = None
    people = assignments(ep)
    existing = [valid_png(image_path(output, ep, i, people), cfg) for i in range(ep["images"])]
    # A partial episode is replayed from its seed; saved indices are retained.
    lights = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = cfg["fixed_delta_seconds"]
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(ep["seed"])
        tm.set_global_distance_to_leading_vehicle(2.5)
        world.set_pedestrians_seed(ep["seed"])
        world.set_pedestrians_cross_factor(cfg["pedestrian_crossing_fraction"])
        world.set_weather(weather_value(carla, ep["weather"]))
        library = world.get_blueprint_library()
        children, adults = pedestrian_pools(library)
        for light in world.get_actors().filter("traffic.traffic_light*"):
            lights.append((light, light.get_green_time(), light.get_yellow_time(), light.get_red_time()))
            light.set_green_time(cfg["traffic_light_green_seconds"])
            light.set_yellow_time(cfg["traffic_light_yellow_seconds"])
            light.set_red_time(cfg["traffic_light_red_seconds"])
        if not lights or not list(world.get_actors().filter("traffic.speed_limit.*")):
            raise RuntimeError(f"{ep['town']} needs traffic lights and speed-limit signs; choose an urban map")
        spawns = ranked_spawns(world, ep, rng)
        ego_bp = library.find(ep["ego_model"])
        ego_bp.set_attribute("role_name", "hero")
        ego = None
        for transform in spawns:
            ego = world.try_spawn_actor(ego_bp, transform)
            if ego is not None:
                actors.append(ego)
                break
        if ego is None:
            raise RuntimeError("Could not spawn camera car")
        ego.set_autopilot(True, args.tm_port)
        tm.vehicle_percentage_speed_difference(ego, 20)
        tm.update_vehicle_lights(ego, True)
        vehicles = sorted(library.filter("vehicle.*"), key=lambda bp: bp.id)
        traffic = []
        # Populate nearby road spawn points first, then the rest of town.
        near = sorted(world.get_map().get_spawn_points(), key=lambda t: t.location.distance(ego.get_location()))
        model_cycle = list(vehicles)
        rng.shuffle(model_cycle)
        for transform in near:
            if len(traffic) >= ep["vehicles"]:
                break
            bp = model_cycle[len(traffic) % len(model_cycle)]
            for attribute in ("color", "driver_id"):
                if bp.has_attribute(attribute):
                    choices = bp.get_attribute(attribute).recommended_values
                    if choices:
                        bp.set_attribute(attribute, rng.choice(choices))
            bp.set_attribute("role_name", "autopilot")
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                actors.append(vehicle)
                traffic.append(vehicle)
                vehicle.set_autopilot(True, args.tm_port)
                tm.update_vehicle_lights(vehicle, True)
        # Draw navigable sidewalks preferentially near the camera car. Balance successful spawns.
        walkers, ages = [], Counter()
        child_target = max(1, min(ep["pedestrians"] - 1, round(ep["pedestrians"] * cfg["child_fraction"])))
        for attempt in range(ep["pedestrians"] * 15):
            if len(walkers) >= ep["pedestrians"]:
                break
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if attempt < ep["pedestrians"] * 10 and loc.distance(ego.get_location()) > cfg["pedestrian_radius"]:
                continue
            age = "child" if ages["child"] < child_target else "adult"
            bp = rng.choice(children if age == "child" else adults)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is not None:
                actors.append(walker)
                controller = world.try_spawn_actor(library.find("controller.ai.walker"), carla.Transform(), walker)
                if controller is None:
                    walker.destroy()
                    actors.remove(walker)
                    continue
                actors.append(controller)
                controllers.append(controller)
                walkers.append(walker)
                ages[age] += 1
        if not traffic or not ages["child"] or not ages["adult"]:
            raise RuntimeError(f"Insufficient successful actor spawns: vehicles={len(traffic)}, ages={dict(ages)}")
        world.tick(args.timeout)
        for controller in controllers:
            controller.start()
            target = world.get_random_location_from_navigation()
            if target is not None:
                controller.go_to_location(target)
            controller.set_max_speed(rng.uniform(0.8, 1.8))
        sensor_bp = library.find("sensor.camera.rgb")
        for key, value in {"image_size_x": cfg["width"], "image_size_y": cfg["height"],
                           "fov": cfg["fov"], "sensor_tick": 0}.items():
            sensor_bp.set_attribute(key, str(value))
        camera = world.spawn_actor(sensor_bp, carla.Transform(carla.Location(x=cfg["camera_x"], z=cfg["camera_z"])), attach_to=ego)
        actors.append(camera)
        frames = queue.Queue()
        camera.listen(frames.put)
        print(f"Camera ready; {len(traffic)} vehicles, {dict(ages)} pedestrians. Warming up...", flush=True)

        def tick():
            frame = world.tick(args.timeout)
            deadline = time.monotonic() + args.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Camera timeout waiting for simulation frame {frame}")
                try:
                    image = frames.get(timeout=remaining)
                except queue.Empty as exc:
                    raise RuntimeError(f"Camera timeout waiting for simulation frame {frame}") from exc
                if image.frame == frame:
                    return image
                if image.frame > frame:
                    raise RuntimeError("Camera frame ahead of world tick; another client may be ticking")

        dt = cfg["fixed_delta_seconds"]
        for _ in range(round(cfg["warmup_seconds"] / dt)):
            tick()
        interval = round(cfg["capture_interval_seconds"] / dt)
        saved = 0
        last_save_loc = ego.get_location()
        last_moving_loc = ego.get_location()
        last_save_tick, last_motion_tick = -interval, 0
        relocations = 0
        for step in range(1, round(cfg["max_episode_seconds"] / dt) + 1):
            image = tick()
            loc = ego.get_location()
            if step % round(30 / dt) == 0:
                # Move walkers toward navigable sidewalks near the car as its route progresses.
                nearby_goals = []
                for _ in range(200):
                    goal = world.get_random_location_from_navigation()
                    if goal is not None and goal.distance(loc) < cfg["pedestrian_radius"]:
                        nearby_goals.append(goal)
                if nearby_goals:
                    for controller in controllers:
                        controller.go_to_location(rng.choice(nearby_goals))
            if loc.distance(last_moving_loc) >= 1:
                last_motion_tick, last_moving_loc = step, loc
            if (step - last_motion_tick) * dt >= cfg["stuck_seconds"]:
                # Keep collection moving when autopilot is trapped. No image at the teleport tick.
                candidates = list(world.get_map().get_spawn_points())
                rng.shuffle(candidates)
                other_locations = [v.get_location() for v in traffic if v.is_alive]
                destination = next((p for p in candidates if all(p.location.distance(v) > 10 for v in other_locations)), None)
                if destination is None:
                    raise RuntimeError("No clear spawn point for stuck-car recovery")
                ego.set_transform(destination)
                ego.set_target_velocity(carla.Vector3D())
                last_motion_tick, last_save_tick = step, step
                last_moving_loc = last_save_loc = destination.location
                relocations += 1
                continue
            if step % interval:
                continue
            if loc.distance(last_save_loc) < 2 and (step - last_save_tick) * dt < cfg["stationary_capture_seconds"]:
                continue
            path = image_path(output, ep, saved, people)
            if not existing[saved]:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.stem + ".tmp.png")
                image.save_to_disk(str(temp))
                if not valid_png(temp, cfg):
                    raise RuntimeError(f"Invalid saved image: {temp}")
                temp.replace(path)
            saved += 1
            last_save_tick, last_save_loc = step, loc
            if saved % 50 == 0:
                print(f"{ep['id']} {saved}/{ep['images']} images", flush=True)
            if saved == ep["images"]:
                break
        if saved != ep["images"]:
            raise RuntimeError(f"Episode exhausted time budget with {saved}/{ep['images']} images; resume after investigating")
        write_json(output / "episodes" / (ep["id"] + ".json"), {
            **ep, "actual_vehicles": len(traffic), "actual_pedestrians": dict(ages),
            "traffic_models": sorted(set(v.type_id for v in traffic)),
            "pedestrian_models": sorted(set(w.type_id for w in walkers)),
            "relocations": relocations, "simulated_seconds": step * dt,
            "resumed_images_retained": sum(existing), "server_version": client.get_server_version(),
            "note": "Scene configuration only; not object labels or proof of image visibility"})
    finally:
        try:
            stop_traffic_manager(tm)
        except RuntimeError as exc:
            print(f"Traffic Manager cleanup warning: {exc}", file=sys.stderr)
        if camera is not None:
            try:
                camera.stop()
            except RuntimeError:
                pass
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
        try:
            live_ids = {actor.id for actor in world.get_actors()}
            commands = [carla.command.DestroyActor(actor.id) for actor in reversed(actors) if actor.id in live_ids]
            if commands:
                client.apply_batch_sync(commands, False)
        except RuntimeError as exc:
            print(f"Actor cleanup warning: {exc}", file=sys.stderr)
        for light, green, yellow, red in lights:
            try:
                light.set_green_time(green)
                light.set_yellow_time(yellow)
                light.set_red_time(red)
            except RuntimeError:
                pass
        # Attempt each restoration even if the connection drops during an earlier cleanup.
        for restore in (lambda: world.apply_settings(original_settings),
                        lambda: world.set_weather(original_weather)):
            try:
                restore()
            except RuntimeError as exc:
                print(f"Cleanup warning: {exc}", file=sys.stderr)


def collect(args, output, plan):
    carla, client = connect(args)
    cfg = plan["config"]
    available = {name.rsplit("/", 1)[-1] for name in client.get_available_maps()}
    missing = set(cfg["towns"]) - available
    if missing:
        raise RuntimeError(f"Unavailable towns: {sorted(missing)}. Run inspect and edit config.json before planning")
    library = client.get_world().get_blueprint_library()
    models = {bp.id for bp in library.filter("vehicle.*")}
    if set(cfg["ego_models"]) - models:
        raise RuntimeError(f"Unavailable ego models: {sorted(set(cfg['ego_models']) - models)}. Run inspect")
    pedestrian_pools(library)
    for weather in cfg["weather_presets"]:
        weather_value(carla, weather)
    completed = 0
    for ep in plan["episodes"]:
        people = assignments(ep)
        marker = output / "episodes" / (ep["id"] + ".json")
        try:
            record = json.loads(marker.read_text())
            complete = all(record.get(key) == value for key, value in ep.items())
        except (OSError, ValueError):
            complete = False
        if complete and all(valid_png(image_path(output, ep, i, people), cfg) for i in range(ep["images"])):
            continue
        print(f"Collecting {ep['id']}: {ep['town']}, {ep['weather']}, {ep['ego_model']}", flush=True)
        capture_episode(carla, client, args, output, cfg, ep)
        completed += 1
        if args.episode_limit and completed >= args.episode_limit:
            print("Episode limit reached. Run collect again to resume.")
            return
    verify(output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "plan", "collect", "verify", "export"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--startup-timeout", type=float, default=600, help="Wait for initial RPC/map readiness")
    parser.add_argument("--load-timeout", type=float, default=300, help="Timeout for changing towns")
    parser.add_argument("--completed-only", action="store_true", help="For verify: check completed episodes of a partial run")
    parser.add_argument("--archive", type=Path, default=ROOT / "completed_episodes.zip", help="Destination for export")
    parser.add_argument("--full-png", action="store_true", help="For verify: validate all PNG CRCs and decompress pixel scanlines")
    parser.add_argument("--episode-limit", type=int, default=0, help="Collect this many unfinished episodes, then stop")
    args = parser.parse_args()
    if min(args.timeout, args.startup_timeout, args.load_timeout) <= 0 or args.episode_limit < 0:
        parser.error("timeout must be positive; episode-limit must be nonnegative")
    if args.command == "inspect":
        inspect(args)
        return
    output = args.output.resolve()
    cfg = load_config(args.config)
    plan = make_plan(cfg)
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Config differs from saved plan. Restore the config or use a new --output folder")
    elif args.command == "plan":
        if output.exists() and any(output.iterdir()):
            raise ValueError("New dataset output must be empty")
        write_json(path, plan)
    else:
        raise ValueError("Run plan first to freeze the collection configuration")
    if args.command == "plan":
        # Assignment manifests contain paths only. No labels, actor states or bounding boxes.
        manifests = {person: [] for person in range(1, 6)}
        for ep in plan["episodes"]:
            people = assignments(ep)
            for i, person in enumerate(people):
                manifests[person].append(str(image_path(output, ep, i, people).relative_to(output)))
        for person, paths in manifests.items():
            file = output / f"person_{person}" / "images.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("\n".join(paths) + "\n")
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        print(f"Plan: {len(plan['episodes'])} episodes, {cfg['total_images']} images; SHA256 {digest}")
        print("Per person:", {p: len(paths) for p, paths in manifests.items()})
    elif args.command == "collect":
        collect(args, output, plan)
    elif args.command == "export":
        export_completed(output, plan, args.archive)
    else:
        verify(output, plan, completed_only=args.completed_only, full_png=args.full_png)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Collection interrupted. Run the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python
"""
set_map.py - list CARLA's maps, or switch the running CARLA to a map.

Usage (CarlaUE4 must be open, or starting up):
    py -3.12 set_map.py --list        show every map you can load
    py -3.12 set_map.py Town03        switch to Town03
    py -3.12 set_map.py Town12        switch to Town12, or just move the view to the city if already loaded

After loading, the CARLA window's view is moved above the busiest streets
(use --no-move to skip this).

Waits for CARLA to finish starting before switching, so it can be run right
after launching CarlaUE4.exe.
"""

import argparse
import sys
import time

import carla
import numpy as np


def short_name(path):
    """'/Game/Carla/Maps/Town03' or 'Carla/Maps/Town03' -> 'Town03'"""
    return path.rstrip("/").split("/")[-1]


def connect(host, port, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    announced = False
    while True:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        try:
            current = client.get_world().get_map().name
            return client, current
        except RuntimeError:
            if time.monotonic() >= deadline:
                print(f"Could not reach CARLA after {wait_seconds} s. Is CarlaUE4 open?")
                sys.exit(1)
            if not announced:
                print("Waiting for CARLA to finish starting...")
                announced = True
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", nargs="?", help="Map to load, e.g. Town03")
    parser.add_argument("--list", action="store_true", help="List available maps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--wait", type=float, default=300, help="Seconds to wait for CARLA to start")
    parser.add_argument("--no-move", action="store_true", help="Don't move the view to the city after loading")
    args = parser.parse_args()

    if not args.map and not args.list:
        parser.print_help()
        return

    client, current = connect(args.host, args.port, args.wait)
    available = sorted({short_name(m): m for m in client.get_available_maps()}.items())
    current_short = short_name(current)

    if args.list:
        print("Maps you can load (* = loaded now):")
        for name, _ in available:
            print(f"  {'*' if name == current_short else ' '} {name}")
        return

    matches = [(name, path) for name, path in available if name.lower() == args.map.lower()]
    if not matches:
        print(f"'{args.map}' is not an available map. Available maps:")
        for name, _ in available:
            print("   ", name)
        sys.exit(1)
    name, path = matches[0]

    if name == current_short:
        print(f"{name} is already loaded.")
    else:
        print(f"Loading {name} (large maps like Town12 can take several minutes)...")
        client.set_timeout(600.0)
        client.load_world(path)
        print(f"Done. CARLA is now on {name}.")

    if not args.no_move:
        client.set_timeout(60.0)
        move_spectator_to_city(client.get_world())


def move_spectator_to_city(world):
    """Move the CARLA window's view to the area with the most road spawn points
    (the busiest streets). On large maps like Town12 only the ground near the view
    is loaded, so this also makes the city appear."""
    points = world.get_map().get_spawn_points()
    if not points:
        return
    xy = np.array([[p.location.x, p.location.y] for p in points])
    cells = np.floor(xy / 250.0).astype(int)  # 250 m squares
    keys, counts = np.unique(cells, axis=0, return_counts=True)
    busiest = keys[np.argmax(counts)]
    in_cell = np.where((cells == busiest).all(axis=1))[0]
    center = xy[in_cell].mean(axis=0)
    best = points[int(in_cell[np.argmin(np.linalg.norm(xy[in_cell] - center, axis=1))])]
    loc = best.location
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=loc.x, y=loc.y, z=loc.z + 40.0),
        carla.Rotation(pitch=-35.0, yaw=best.rotation.yaw)))
    print(f"Moved the view above the busiest streets (x={loc.x:.0f}, y={loc.y:.0f}). "
          "The ground may take a few seconds to appear.")


if __name__ == "__main__":
    main()

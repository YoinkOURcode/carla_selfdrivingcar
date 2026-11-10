"""Offline invariants; no simulator or CARLA installation needed."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from types import SimpleNamespace
import zlib


DATA = Path(__file__).resolve().parents[1]
BALANCED_PROPORTIONS = [0.1, 0.1, 0.2, 0.2, 0.4]
spec = importlib.util.spec_from_file_location("generate_dataset", DATA / "generate_dataset.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


def png(width, height):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\0" * (width * 3)) * height))
            + chunk(b"IEND", b""))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dataset.load_config(DATA / "config.json")

    def test_exact_workloads_and_balanced_temporal_blocks(self):
        cfg = dict(self.cfg, proportions=BALANCED_PROPORTIONS)
        plan = dataset.make_plan(cfg)
        totals = Counter()
        self.assertEqual(len(plan["episodes"]), self.cfg["total_images"] // self.cfg["images_per_episode"])
        for ep in plan["episodes"]:
            owners = dataset.assignments(ep, cfg["proportions"])
            self.assertEqual(owners, dataset.assignments(ep, cfg["proportions"]))
            expected_episode = {person: int(ep["images"] * proportion)
                                for person, proportion in enumerate(BALANCED_PROPORTIONS, 1)}
            expected_block = {person: int(10 * proportion)
                              for person, proportion in enumerate(BALANCED_PROPORTIONS, 1)}
            self.assertEqual(Counter(owners), expected_episode)
            for start in range(0, len(owners), 10):
                self.assertEqual(Counter(owners[start:start + 10]), expected_block)
            totals.update(owners)
        expected = {person: int(self.cfg["total_images"] * proportion)
                    for person, proportion in enumerate(BALANCED_PROPORTIONS, 1)}
        self.assertEqual(totals, expected)

    def test_scenario_coverage_identical_between_people(self):
        cfg = dict(self.cfg, proportions=BALANCED_PROPORTIONS)
        plan = dataset.make_plan(cfg)
        pairs = Counter((ep["town"], ep["weather"]) for ep in plan["episodes"])
        self.assertEqual(len(pairs), 50)
        self.assertEqual(set(pairs.values()), {len(plan["episodes"]) // len(pairs)})
        self.assertEqual(set(ep["ego_model"] for ep in plan["episodes"]), set(self.cfg["ego_models"]))
        self.assertEqual(set(ep["focus"] for ep in plan["episodes"]), {"traffic_lights", "speed_limit", "pedestrians"})
        distributions = {p: Counter() for p in range(1, 6)}
        for ep in plan["episodes"]:
            for person in dataset.assignments(ep, cfg["proportions"]):
                distributions[person][(ep["town"], ep["weather"], ep["ego_model"])] += 1
        for person, distribution in distributions.items():
            normalized = {key: value / sum(distribution.values()) for key, value in distribution.items()}
            reference = distributions[1]
            self.assertEqual(normalized, {key: value / sum(reference.values()) for key, value in reference.items()})

    def test_invalid_plan_rejected(self):
        cfg = dict(self.cfg, total_images=200)
        with self.assertRaisesRegex(ValueError, "town/weather"):
            dataset.make_plan(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(self.cfg, images_per_episode=201)))
            with self.assertRaises(ValueError):
                dataset.load_config(path)

    def test_configured_proportions_assign_every_image_to_person_five(self):
        cfg = dict(self.cfg, proportions=[0, 0, 0, 0, 1])
        plan = dataset.make_plan(cfg)
        for ep in plan["episodes"]:
            owners = dataset.assignments(ep, cfg["proportions"])
            self.assertEqual(owners, [5] * ep["images"])

    def test_invalid_proportions_rejected(self):
        invalid = ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0.5],
                   [0, 0, 0, 0, -1], [0, 0, 0, 0, float("nan")])
        for proportions in invalid:
            with self.subTest(proportions=proportions), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(dict(self.cfg, proportions=proportions)))
                with self.assertRaisesRegex(ValueError, "proportions"):
                    dataset.load_config(path)

    def test_zero_traffic_light_durations_are_allowed(self):
        cfg = dict(self.cfg, traffic_light_green_seconds=0,
                   traffic_light_yellow_seconds=0, traffic_light_red_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg))
            loaded = dataset.load_config(path)
        self.assertEqual(loaded["traffic_light_green_seconds"], 0)
        self.assertEqual(loaded["traffic_light_yellow_seconds"], 0)
        self.assertEqual(loaded["traffic_light_red_seconds"], 0)

    def test_negative_traffic_light_duration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(dict(self.cfg, traffic_light_green_seconds=-1)))
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                dataset.load_config(path)

    def test_single_active_traffic_light_phase_is_held_and_restored(self):
        """Apply a constant phase to all lights and restore even after failure."""
        states = SimpleNamespace(Green="Green", Yellow="Yellow", Red="Red")
        scenarios = ((0, 1000, 0, "Yellow"), (1000, 0, 0, "Green"),
                     (0, 0, 1000, "Red"), (12, 18, 12, None),
                     (0, 0, 0, None))
        for green, yellow, red, expected in scenarios:
            with self.subTest(durations=(green, yellow, red)):
                cfg = dict(self.cfg, traffic_light_green_seconds=green,
                           traffic_light_yellow_seconds=yellow,
                           traffic_light_red_seconds=red)
                lights = [Mock(), Mock()]
                for light, state in zip(lights, ("Green", "Red")):
                    light.current_state, light.frozen = state, False
                    light.get_state.side_effect = lambda light=light: light.current_state
                    light.is_frozen.side_effect = lambda light=light: light.frozen
                    light.get_green_time.return_value = 12
                    light.get_yellow_time.return_value = 18
                    light.get_red_time.return_value = 12
                    light.set_state.side_effect = lambda state, light=light: setattr(light, "current_state", state)

                    def freeze_all(frozen):
                        # CARLA freezing can affect other lights in the group.
                        for member in lights:
                            member.frozen = frozen

                    light.freeze.side_effect = freeze_all
                world, client = Mock(), Mock()
                client.load_world.return_value = world
                world.get_actors.return_value = MagicMock()
                world.get_actors.return_value.filter.side_effect = (
                    lambda pattern: lights if pattern == "traffic.traffic_light*" else [Mock()])
                args = SimpleNamespace(load_timeout=300, timeout=60, tm_port=8000)
                ep = dataset.make_plan(cfg)["episodes"][0]

                def check_configured_lights(*unused):
                    """Check the scene before spawning, then exercise cleanup."""
                    for light in lights:
                        light.set_green_time.assert_called_with(green)
                        light.set_yellow_time.assert_called_with(yellow)
                        light.set_red_time.assert_called_with(red)
                        if expected is None:
                            light.set_state.assert_not_called()
                            light.freeze.assert_not_called()
                        else:
                            self.assertEqual(light.current_state, expected)
                            self.assertTrue(light.frozen)
                    raise RuntimeError("stop after configuring lights")

                with tempfile.TemporaryDirectory() as directory, \
                        patch.object(dataset, "pedestrian_pools", return_value=([], [])), \
                        patch.object(dataset, "weather_value"), \
                        patch.object(dataset, "ranked_spawns", side_effect=check_configured_lights), \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "stop after configuring lights"):
                        dataset.capture_episode(SimpleNamespace(TrafficLightState=states),
                                                client, args, Path(directory), cfg, ep)
                for light, original in zip(lights, ("Green", "Red")):
                    self.assertEqual(light.current_state, original)
                    self.assertFalse(light.frozen)
                    light.set_green_time.assert_called_with(12)
                    light.set_yellow_time.assert_called_with(18)
                    light.set_red_time.assert_called_with(12)

    def test_verify_complete_missing_wrong_size_and_extra_images(self):
        cfg = dict(self.cfg, total_images=10, images_per_episode=10,
                   towns=["Town01"], weather_presets=["ClearNoon"], width=2, height=2)
        plan = dataset.make_plan(cfg)
        ep = plan["episodes"][0]
        people = dataset.assignments(ep, cfg["proportions"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for i in range(ep["images"]):
                path = dataset.image_path(output, ep, i, people)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png(2, 2))
            for person in range(1, 6):
                paths = [str(dataset.image_path(output, ep, i, people).relative_to(output))
                         for i, owner in enumerate(people) if owner == person]
                folder = output / f"person_{person}"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "images.txt").write_text("\n".join(paths) + ("\n" if paths else ""))
            dataset.write_json(output / "episodes" / (ep["id"] + ".json"), ep)
            with contextlib.redirect_stdout(io.StringIO()):
                dataset.verify(output, plan)
            report = json.loads((output / "verification.json").read_text())
            self.assertEqual(report["valid_images"], 10)
            path = dataset.image_path(output, ep, 0, people)
            path.write_bytes(png(3, 2))
            self.assertFalse(dataset.valid_png(path, cfg))
            path.unlink()
            extra = output / "person_1" / ep["id"] / "extra.png"
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_bytes(png(2, 2))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                dataset.verify(output, plan)
            report = json.loads((output / "verification.json").read_text())
            self.assertEqual(report["missing_or_invalid"], 1)
            self.assertEqual(report["unexpected_images"], 1)

    def test_truncated_png_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(png(1280, 720)[:-5])
            self.assertFalse(dataset.valid_png(path, self.cfg))

    def test_full_png_verification_detects_interior_corruption(self):
        cfg = dict(self.cfg, width=2, height=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            data = bytearray(png(2, 2))
            path.write_bytes(data)
            self.assertTrue(dataset.valid_png(path, cfg, full=True))
            data[45] ^= 1
            path.write_bytes(data)
            self.assertTrue(dataset.valid_png(path, cfg))
            self.assertFalse(dataset.valid_png(path, cfg, full=True))

    def test_pedestrian_age_uses_attribute_value_not_debug_string(self):
        def blueprint(identifier, age):
            bp = Mock()
            bp.id = identifier
            bp.has_attribute.return_value = age is not None
            attribute = Mock()
            attribute.as_str.return_value = age
            bp.get_attribute.return_value = attribute
            return bp
        library = Mock()
        library.filter.return_value = [blueprint("walker.pedestrian.0009", "child"),
                                       blueprint("walker.pedestrian.0001", "adult"),
                                       blueprint("walker.pedestrian.0050", "teenager"),
                                       blueprint("walker.pedestrian.0049", None)]
        children, adults = dataset.pedestrian_pools(library)
        self.assertEqual([bp.id for bp in children], ["walker.pedestrian.0009", "walker.pedestrian.0049"])
        self.assertEqual([bp.id for bp in adults], ["walker.pedestrian.0001"])

    def test_connection_waits_for_world_readiness_then_restores_rpc_timeout(self):
        client = Mock()
        client.get_server_version.return_value = "matching"
        client.get_client_version.return_value = "matching"
        client.get_world.side_effect = [RuntimeError("map still loading"), Mock()]
        fake_carla = SimpleNamespace(Client=Mock(return_value=client))
        args = SimpleNamespace(host="127.0.0.1", port=2000, timeout=60, startup_timeout=600)
        with patch.dict("sys.modules", {"carla": fake_carla}), patch.object(dataset.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            _, connected = dataset.connect(args)
        self.assertIs(connected, client)
        self.assertEqual(client.get_world.call_count, 2)
        client.set_timeout.assert_called_with(60)

    def test_traffic_manager_removed_without_starting_async_worker(self):
        tm = Mock()
        dataset.stop_traffic_manager(tm)
        from unittest.mock import call
        self.assertEqual(tm.mock_calls, [call.set_synchronous_mode(True), call.shut_down()])

    def test_partial_verification_ignores_uncollected_episodes(self):
        cfg = dict(self.cfg, total_images=20, images_per_episode=10,
                   towns=["Town01"], weather_presets=["ClearNoon"], width=2, height=2)
        plan = dataset.make_plan(cfg)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for person in range(1, 6):
                paths = []
                for ep in plan["episodes"]:
                    owners = dataset.assignments(ep, plan["config"]["proportions"])
                    for i, owner in enumerate(owners):
                        if owner == person:
                            paths.append(str(dataset.image_path(output, ep, i, owners).relative_to(output)))
                folder = output / f"person_{person}"
                folder.mkdir()
                (folder / "images.txt").write_text("\n".join(paths) + ("\n" if paths else ""))
            ep = plan["episodes"][0]
            owners = dataset.assignments(ep, plan["config"]["proportions"])
            for i in range(10):
                path = dataset.image_path(output, ep, i, owners)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png(2, 2))
            dataset.write_json(output / "episodes" / (ep["id"] + ".json"), ep)
            dataset.write_json(output / "plan.json", plan)
            with contextlib.redirect_stdout(io.StringIO()):
                dataset.verify(output, plan, completed_only=True)
            report = json.loads((output / "verification_completed.json").read_text())
            self.assertEqual(report["expected_images"], 10)
            self.assertEqual(report["planned_images"], 20)
            archive = output / "partial.zip"
            with contextlib.redirect_stdout(io.StringIO()):
                dataset.export_completed(output, plan, archive)
            with dataset.zipfile.ZipFile(archive) as bundle:
                self.assertEqual(sum(name.endswith(".png") for name in bundle.namelist()), 10)
                expected_person_five = sum(
                    owner == 5 for owner in dataset.assignments(
                        plan["episodes"][0], plan["config"]["proportions"]))
                self.assertEqual(len(bundle.read("person_5/images.txt").decode().splitlines()),
                                 expected_person_five)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                dataset.verify(output, plan)


if __name__ == "__main__":
    unittest.main()

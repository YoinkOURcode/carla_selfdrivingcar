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
from unittest.mock import Mock, patch
from types import SimpleNamespace
import zlib


DATA = Path(__file__).resolve().parents[1]
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
        plan = dataset.make_plan(self.cfg)
        totals = Counter()
        self.assertEqual(len(plan["episodes"]), 100)
        for ep in plan["episodes"]:
            owners = dataset.assignments(ep)
            self.assertEqual(owners, dataset.assignments(ep))
            self.assertEqual(Counter(owners), {1: 20, 2: 20, 3: 40, 4: 40, 5: 80})
            for start in range(0, len(owners), 10):
                self.assertEqual(Counter(owners[start:start + 10]), {1: 1, 2: 1, 3: 2, 4: 2, 5: 4})
            totals.update(owners)
        self.assertEqual(totals, {1: 2000, 2: 2000, 3: 4000, 4: 4000, 5: 8000})

    def test_scenario_coverage_identical_between_people(self):
        plan = dataset.make_plan(self.cfg)
        pairs = Counter((ep["town"], ep["weather"]) for ep in plan["episodes"])
        self.assertEqual(len(pairs), 50)
        self.assertEqual(set(pairs.values()), {2})
        self.assertEqual(set(ep["ego_model"] for ep in plan["episodes"]), set(self.cfg["ego_models"]))
        self.assertEqual(set(ep["focus"] for ep in plan["episodes"]), {"traffic_lights", "speed_limit", "pedestrians"})
        distributions = {p: Counter() for p in range(1, 6)}
        for ep in plan["episodes"]:
            for person in dataset.assignments(ep):
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

    def test_verify_complete_missing_wrong_size_and_extra_images(self):
        cfg = dict(self.cfg, total_images=10, images_per_episode=10,
                   towns=["Town01"], weather_presets=["ClearNoon"], width=2, height=2)
        plan = dataset.make_plan(cfg)
        ep = plan["episodes"][0]
        people = dataset.assignments(ep)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for i in range(ep["images"]):
                path = dataset.image_path(output, ep, i, people)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png(2, 2))
            for person in range(1, 6):
                paths = [str(dataset.image_path(output, ep, i, people).relative_to(output))
                         for i, owner in enumerate(people) if owner == person]
                (output / f"person_{person}" / "images.txt").write_text("\n".join(paths) + "\n")
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
                    owners = dataset.assignments(ep)
                    for i, owner in enumerate(owners):
                        if owner == person:
                            paths.append(str(dataset.image_path(output, ep, i, owners).relative_to(output)))
                folder = output / f"person_{person}"
                folder.mkdir()
                (folder / "images.txt").write_text("\n".join(paths) + "\n")
            ep = plan["episodes"][0]
            owners = dataset.assignments(ep)
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
                self.assertEqual(len(bundle.read("person_5/images.txt").decode().splitlines()), 4)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                dataset.verify(output, plan)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from dreamerv3_attention.config import Config
from dreamerv3_attention.evaluation import (
    compare_results, episode_metrics, evaluate, scenario_catalog, scenario_suite,
)


class ScenarioCatalogTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(seed=17, split="train")

    def test_builtin_catalog_covers_splits_with_unique_seeds_and_routes(self):
        catalog = scenario_catalog(self.cfg)
        self.assertEqual({record["split"] for record in catalog},
                         {"train", "validation", "held_out"})
        self.assertEqual(len({record["seed"] for record in catalog}), len(catalog))
        self.assertEqual(len({(record["route_start"], record["route_end"])
                              for record in catalog}), len(catalog))
        categories = {record["category"] for record in catalog}
        for required in ("straight", "curved", "intersection", "moving_traffic",
                         "static_obstacle", "pedestrian_approaching", "pedestrian_crossing",
                         "child_pedestrian", "front_hazard", "side_hazard", "rear_hazard",
                         "occupied_crossing", "unoccupied_crossing", "speed_limit_transition",
                         "heavy_rain", "low_light", "combined_weather_hazard",
                         "missing_sensor", "stale_sensor"):
            self.assertIn(required, categories)
        supported_categories = {r["category"] for r in catalog if r["supported"]}
        self.assertTrue({"static_obstacle", "speed_limit_transition", "stale_sensor"} <= supported_categories)

    def test_suite_filters_requested_split(self):
        suite = scenario_suite(self.cfg)
        self.assertTrue(suite)
        self.assertTrue(all(record["split"] == "train" for record in suite))
        validation = scenario_suite(replace(self.cfg, split="validation"))
        self.assertTrue(all(record["split"] == "validation" for record in validation))

    def test_json_catalog_requires_unique_routes_seeds_and_explicit_split(self):
        records = [
            {"name": "same", "split": "train", "seed": 1, "route_start": 0,
             "route_end": 10, "category": "straight"},
            {"name": "same", "split": "validation", "seed": 2, "route_start": 20,
             "route_end": 30, "category": "straight"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scenarios.json"
            path.write_text(json.dumps(records))
            cfg = replace(self.cfg, scenario_file=str(path))
            self.assertEqual(len(scenario_suite(cfg)), 1)
            records[1]["seed"] = 1
            path.write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                scenario_suite(cfg)


class EpisodeMetricTests(unittest.TestCase):
    def test_metrics_count_edges_integrals_smoothness_and_missing_obs(self):
        infos = [
            {"timestamp": 0.0, "route_progress_m": 0, "route_length_m": 10,
             "speed_mps": 0, "speed_limit_mps": 5, "target_speed_mps": 0,
             "policy_inference_latency_s": 0.002,
             "lane_departure": True, "collision": False, "pedestrian_collision": False,
             "front_hazard_m": 8, "interventions": ["invalid_observation"],
             "observations_valid": False, "applied_action": [0, -1],
             "crossing_id": "c", "crossing_distance_m": 1, "crossing_phase": "stopped"},
            {"timestamp": 0.1, "route_progress_m": 0.5, "route_length_m": 10,
             "speed_mps": 0, "speed_limit_mps": 5, "target_speed_mps": 0,
             "policy_inference_latency_s": 0.004,
             "lane_departure": True, "collision": True, "pedestrian_collision": True,
             "front_hazard_m": 3, "interventions": ["crosswalk_stop"],
             "observations_valid": True, "applied_action": [0.2, -1],
             "crossing_id": "c", "crossing_distance_m": 0, "crossing_phase": "stopped"},
            {"timestamp": 0.2, "route_progress_m": 1.0, "route_length_m": 10,
             "speed_mps": 7, "speed_limit_mps": 5, "target_speed_mps": 5,
             "lane_departure": False, "collision": True, "pedestrian_collision": True,
             "red_light_violation": True, "crossing_violation": True,
             "front_hazard_m": 4, "interventions": [], "observations_valid": True,
             "applied_action": [0.6, 1], "crossing_id": None,
             "crossing_distance_m": None, "crossing_phase": "passed"},
        ]
        result = episode_metrics(infos, dt_s=0.1)
        self.assertEqual(result["lane_departures"], 1)
        self.assertEqual(result["collision_events"], 1)
        self.assertEqual(result["pedestrian_incidents"], 1)
        self.assertEqual(result["crossings_encountered"], 1)
        self.assertEqual(result["red_light_violations"], 1)
        self.assertEqual(result["crossing_violations"], 1)
        self.assertEqual(result["speed_violation_events"], 1)
        self.assertAlmostEqual(result["speed_violation_duration_s"], 0.1)
        self.assertAlmostEqual(result["minimum_relevant_separation_m"], 3)
        self.assertEqual(result["unnecessary_stopping_s"], 0)
        self.assertEqual(result["supervisor_intervention_steps"], 2)
        self.assertEqual(result["missing_observation_steps"], 1)
        self.assertEqual(result["missing_observation_fail_safe_steps"], 1)
        self.assertAlmostEqual(result["distance_traveled_m"], 0.7)
        self.assertAlmostEqual(result["route_completion"], 0.1)
        self.assertGreater(result["steering_change_mean"], 0)
        self.assertGreater(result["acceleration_rms_mps2"], 0)
        self.assertEqual(result["policy_inference_latency_p95_s"], 0.004)

    def test_unnecessary_stop_excludes_red_and_crosswalk_mandates(self):
        rows = [
            {"timestamp": 0, "speed_mps": 0, "red_light": False,
             "crossing_distance_m": None},
            {"timestamp": 1, "speed_mps": 0, "red_light": True,
             "crossing_distance_m": None},
            {"timestamp": 2, "speed_mps": 0, "red_light": False,
             "crossing_distance_m": 1},
            {"timestamp": 3, "speed_mps": 0, "crossing_distance_m": None},
        ]
        result = episode_metrics(rows, dt_s=1)
        self.assertEqual(result["unnecessary_stopping_s"], 2)
        self.assertEqual(result["unnecessary_stop_events"], 2)


class EvaluationAndComparisonTests(unittest.TestCase):
    def test_evaluate_runs_fake_environment_and_reports_metrics(self):
        class FakeEnv:
            def __init__(self, cfg, scenario):
                self.scenario = scenario
                self.closed = False

            def reset(self):
                return {"obs": 0}

            def step(self, action):
                return {"obs": 1}, 0.0, True, False, {
                    "timestamp": 0.1, "route_progress_m": 2, "route_length_m": 10,
                    "speed_mps": 2, "target_speed_mps": 2, "observations_valid": True,
                    "applied_action": action, "success": False, "route_start": 5, "route_end": 8,
                }

            def close(self):
                self.closed = True

        class FakeAgent:
            def initial(self):
                return 0

            def act(self, observation, carry, deterministic):
                return [0.0, 0.0], carry + 1

        cfg = Config(split="train")
        result = evaluate(cfg, FakeAgent(), FakeEnv, episodes=1)
        self.assertEqual(len(result["episodes"]), 1)
        self.assertEqual(result["episodes"][0]["name"], "train_straight")
        self.assertEqual(result["episodes"][0]["route"], [5, 8])
        self.assertEqual(result["episodes"][0]["scenario"]["route_start"], 0)
        self.assertEqual(result["episodes"][0]["metrics"]["route_completion"], 0.2)
        self.assertEqual(result["summary"]["episode_count"], 1)

    def test_evaluate_carries_executed_action_into_next_recurrent_step(self):
        class Carry:
            device = "cpu"

            def __init__(self):
                self.data = {}

            def set(self, key, value):
                self.data[key] = value

        class FakeEnv:
            def __init__(self, cfg, scenario):
                self.steps = 0

            def reset(self):
                return {}

            def step(self, action):
                self.steps += 1
                return {}, 0.0, self.steps == 2, False, {
                    "timestamp": self.steps * 0.1, "speed_mps": 0,
                    "applied_action": [0.1, -0.3],
                }

            def close(self):
                pass

        class FakeAgent:
            def initial(self):
                return Carry()

            def act(self, observation, carry, deterministic):
                if hasattr(self, "called"):
                    self.asserted_action = carry.data["previous_action"].tolist()
                    self.asserted_first = carry.data["is_init"].tolist()
                self.called = True
                return [0.2, 0.1], carry

        agent = FakeAgent()
        result = evaluate(Config(split="train"), agent, FakeEnv, episodes=1)
        self.assertAlmostEqual(agent.asserted_action[0], 0.1)
        self.assertAlmostEqual(agent.asserted_action[1], -0.3)
        self.assertEqual(agent.asserted_first, [False])
        metrics = result["episodes"][0]["metrics"]
        self.assertGreater(metrics["policy_inference_latency_mean_s"], 0)
        self.assertGreater(metrics["evaluation_steps_per_wall_s"], 0)

    def test_comparison_requires_matched_configuration_and_scenarios(self):
        ep = {"name": "straight", "seed": 1, "route": [0, 10],
              "metrics": {"distance_traveled_m": 5.0}}
        baseline = {"config": {"attention": True, "seed": 1, "device": "cpu"},
                    "episodes": [ep]}
        candidate = {"config": {"attention": False, "seed": 1, "device": "cpu"},
                     "episodes": [{**ep, "metrics": {"distance_traveled_m": 7.0}}]}
        comparison = compare_results(baseline, candidate)
        self.assertEqual(comparison["summary"]["distance_traveled_m_mean_delta"], 2.0)
        self.assertEqual(comparison["checkpoint_provenance"],
                         {"baseline": None, "candidate": None})
        mismatched = {**candidate, "config": {"attention": False, "seed": 2, "device": "cpu"}}
        with self.assertRaises(ValueError):
            compare_results(baseline, mismatched)
        unmatched = {**candidate, "episodes": []}
        with self.assertRaises(ValueError):
            compare_results(baseline, unmatched)

    def test_controller_actor_comparison_requires_matched_task(self):
        ep = {"name": "straight", "seed": 1, "route": [0, 10], "scenario": {},
              "metrics": {"distance_traveled_m": 2.0}}
        actor = {"config": {"attention": True, "seed": 1, "device": "cuda", "reward": {"lane": 1}},
                 "policy": {"type": "actor"}, "episodes": [ep]}
        controller = {"config": {"attention": True, "seed": 1, "device": "cpu", "reward": {"lane": 1}},
                      "policy": {"type": "route_controller", "target_speed_mps": 3.0},
                      "episodes": [{**ep, "metrics": {"distance_traveled_m": 4.0}}]}
        comparison = compare_results(actor, controller)
        self.assertEqual(comparison["summary"]["distance_traveled_m_mean_delta"], 2.0)
        legacy_actor = {**actor, "scope": "fixed_training_sanity"}
        legacy_controller = {**controller, "policy": None, "scope": "fixed_training_sanity",
                             "controller_speed_mps": 3.0}
        legacy_result = compare_results(legacy_actor, legacy_controller, fixed_scenario=True)
        self.assertEqual(legacy_result["policies"]["candidate"]["type"], "route_controller")
        changed = {**controller, "config": {**controller["config"], "seed": 9}}
        with self.assertRaisesRegex(ValueError, "identical task configs"):
            compare_results(actor, changed)
        changed = {**controller, "episodes": [{**ep, "scenario": {"weather": "rain"}}]}
        with self.assertRaisesRegex(ValueError, "same nonempty scenarios"):
            compare_results(actor, changed)
        changed = {**controller, "config": {**controller["config"], "reward": {"lane": 2}}}
        with self.assertRaisesRegex(ValueError, "identical task configs"):
            compare_results(actor, changed)
        for policy in ({"type": "untrusted"}, {"type": "route_controller", "target_speed_mps": float("nan")}):
            with self.assertRaises(ValueError):
                compare_results(actor, {**controller, "policy": policy})
        dynamic = {**controller, "policy": {"type": "route_controller", "speed_mode": "speed_limit"}}
        self.assertEqual(compare_results(actor, dynamic)["policies"]["candidate"]["speed_mode"], "speed_limit")
        with self.assertRaises(ValueError):
            compare_results(actor, {**controller, "policy": {"type": "route_controller", "speed_mode": "other"}})

    def test_trace_audit_separates_clear_idle_stops_and_missing_data(self):
        import json
        import tempfile
        from pathlib import Path
        from dreamerv3_attention.evaluation import audit_trace

        rows = [
            {"scenario": "straight", "speed_mps": 0.0, "target_speed_mps": 3.0,
             "policy_observations_valid": True, "sensor_valid": {"rgb": True},
             "requested_action": [0.1, 0.8], "applied_action": [0.1, 0.8],
             "interventions": [], "lane_error_m": 0.2, "heading_error_rad": 0.1,
             "timestamp": 1.0, "reward_components": {"progress": 0.2}},
            {"scenario": "straight", "speed_mps": 0.0, "target_speed_mps": 0.0,
             "policy_observations_valid": True, "sensor_valid": {"rgb": True},
             "requested_action": [0.0, 0.5], "applied_action": [0.0, -1.0],
             "interventions": ["red_light_stop"], "red_light": True,
             "timestamp": 2.0, "reward_components": {"progress": -0.1}},
            {"scenario": "invalid", "speed_mps": 0.0, "requested_action": [0.0, 0.0],
             "applied_action": [0.0, -1.0], "interventions": ["invalid_observation"],
             "sensor_valid": {"rgb": False}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot json\n")
            report = audit_trace(path)
        self.assertEqual(report["scenarios"]["straight"]["clear_road_stationary_steps"], 1)
        self.assertEqual(report["scenarios"]["straight"]["mandated_stop_steps"], 1)
        self.assertEqual(report["scenarios"]["invalid"]["invalid_observation_steps"], 1)
        self.assertEqual(len(report["malformed_lines"]), 1)
        self.assertEqual(report["overall"]["reward_component_totals"]["progress"], 0.1)

    def test_trace_audit_counts_weak_throttle_and_separates_brakes(self):
        from dreamerv3_attention.evaluation import audit_trace

        rows = [{"scenario": "idle", "speed_mps": 0.0, "target_speed_mps": 3.0,
                 "observations_valid": True, "requested_action": [0.0, value],
                 "applied_action": [0.0, value], "interventions": []}
                for value in (0.03, 0.07, 0.09)]
        rows.append({"scenario": "idle", "speed_mps": 0.0, "target_speed_mps": 0.0,
                     "observations_valid": True, "requested_action": [0.0, -0.5],
                     "applied_action": [0.0, -0.5], "interventions": ["red_light_stop"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = audit_trace(path)["overall"]
        self.assertEqual(result["weak_positive_throttle_steps"], 3)
        self.assertEqual(result["clear_road_stationary_weak_positive_throttle_steps"], 3)
        self.assertEqual(result["positive_throttle_request_steps"], 3)
        self.assertEqual(result["brake_request_steps"], 1)
        self.assertEqual(result["applied_minus_requested_longitudinal"]["median"], 0.0)
        self.assertAlmostEqual(result["requested_signed_longitudinal"]["median"], 0.05)
        self.assertAlmostEqual(result["applied_signed_longitudinal"]["median"], 0.05)

    def test_trace_audit_excludes_hazard_braking_from_clear_road_idle(self):
        from dreamerv3_attention.evaluation import audit_trace

        rows = [{"scenario": "hazards", "speed_mps": 0.0, "target_speed_mps": 3.0,
                 "policy_observations_valid": True, "requested_action": [0.0, 0.8],
                 "applied_action": [0.0, -1.0], "interventions": [reason]}
                for reason in ("pedestrian_conflict_stop", "front_hazard_brake")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = audit_trace(path)["overall"]
        self.assertEqual(result["stationary_steps"], 2)
        self.assertEqual(result["clear_road_stationary_steps"], 0)
        self.assertEqual(result["clear_road_observation_steps"], 0)
        self.assertEqual(result["safety_blocked_steps"], 2)

    def test_trace_audit_handles_invalid_fields_and_resets_acceleration(self):
        from dreamerv3_attention.evaluation import audit_trace

        lines = [
            json.dumps({"scenario": "one", "timestamp": 1.0, "speed_mps": 0.0, "frame": 8,
                        "observations_valid": True, "requested_action": [0.0, 0.5],
                        "applied_action": [0.0, 0.5]}),
            json.dumps({"scenario": "two", "timestamp": 1.1, "speed_mps": 2.0, "frame": 9,
                        "requested_action": 1, "applied_action": [0.0, 0.5]}),
            json.dumps({"scenario": "two", "timestamp": 1.2, "speed_mps": 3.0, "frame": 20,
                        "requested_action": [0.0, 0.5], "applied_action": [0.0, 0.5]}),
            '{"scenario":"two","timestamp":1.2,"speed_mps":"fast",'
            '"requested_action":[0.0,1e999],"applied_action":[0.0,0.5]}',
            '{"scenario":"bad","speed_mps":NaN}',
            '{"scenario":"bad","speed_mps":Infinity}',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(lines) + "\n")
            report = audit_trace(path)
        self.assertEqual(report["overall"]["unknown_observation_validity_steps"], 3)
        self.assertEqual(report["overall"]["known_observation_validity_steps"], 1)
        self.assertEqual(report["overall"]["acceleration_mps2"]["count"], 0)
        self.assertEqual(len(report["malformed_lines"]), 2)
        self.assertGreaterEqual(len(report["field_issues"]), 3)
        self.assertIn("speed_mps", [issue["field"] for issue in report["field_issues"]])


if __name__ == "__main__":
    unittest.main()

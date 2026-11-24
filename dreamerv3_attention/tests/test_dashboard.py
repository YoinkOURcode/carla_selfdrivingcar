import json

import pytest

from dreamerv3_attention import dashboard as d


def steps(n, **fields):
    """n steps at 0.05 s; list-valued fields give per-step values."""
    rows = []
    for i in range(n):
        row = {"episode": 1, "timestamp": .05*i, "speed_mps": 10., "route_length_m": 100.,
               "route_progress_m": float(i), "reward": 1., "terminated": i == n-1}
        row.update({k: v[i] if isinstance(v, list) else v for k, v in fields.items()})
        rows.append(row)
    return d.prepare(rows, d.TTC_THRESHOLD_S)


def test_ttc_uses_closing_rate_and_ignores_hazard_switches():
    # Closing at 10 m/s, then the hazard jumps from 10 m to 60 m (a different object).
    rows = steps(5, front_hazard_m=[12., 11.5, 11., 10.5, 60.])
    ttc = [r["_ttc"] for r in rows]
    assert ttc[0] is None and ttc[4] is None
    assert ttc[1:4] == pytest.approx([1.15, 1.1, 1.05])
    assert d.min_ttc(rows) == pytest.approx(1.05)
    assert d.tet(rows) == pytest.approx(.15)  # three steps under 1.5 s
    assert d.ttc_events(rows) == 1


def test_no_ttc_when_opening_or_no_hazard():
    rows = steps(3, front_hazard_m=[None, 10., 11.])
    assert d.min_ttc(rows) is None and d.tet(rows) == 0


def test_violations_count_once_per_event():
    rows = steps(6, red_light_violation=[False, True, True, False, True, True])
    assert d.red_light_violations(rows) == 2
    assert d.infraction_penalty(rows) == pytest.approx(.7**2)


def test_driving_score_with_typed_and_legacy_collisions():
    typed = steps(4, collision=[False, False, False, True],
                  collision_actor_types=[[], [], [], ["vehicle.tesla.model3"]])
    assert d.collisions_vehicle(typed) == 1 and d.collisions_type_assumed(typed) == 0
    assert d.driving_score(typed) == pytest.approx(3.*.6)  # 3 of 100 m

    legacy = steps(4, collision=[False, False, False, True])
    assert d.collisions_static(legacy) == 1 and d.collisions_type_assumed(legacy) == 1
    assert d.infraction_penalty(legacy) == pytest.approx(.65)

    pedestrian = steps(2, collision=[False, True], pedestrian_collision=[False, True])
    assert d.collisions_pedestrian(pedestrian) == 1 and d.infraction_penalty(pedestrian) == pytest.approx(.5)


def test_route_completion_excludes_offroad_progress_and_counts_success_as_complete():
    rows = steps(11, offroad=[False]*5 + [True]*3 + [False]*3)
    assert d.route_completion(rows) == pytest.approx(10. - 3.)
    done = steps(11, success=[False]*10 + [True])
    assert d.route_completion(done) == 100.


def test_interventions_and_rewards_breakdowns():
    rows = steps(4, interventions=[[], ["brake"], ["brake", "steer"], []],
                 reward_components=[{"progress": 1., "time": -.1}]*4)
    assert d.intervention_steps(rows) == 2 and d.intervention_events(rows) == 1
    assert d.intervention_types(rows) == {"brake": 2, "steer": 1}
    assert d.intervention_type_events(rows) == {"brake": 1, "steer": 1}
    assert d.reward_components(rows) == pytest.approx({"progress": 4., "time": -.4})


def test_summarize_keeps_complete_episodes_and_writes_metadata(tmp_path):
    path = tmp_path/"run"/"metrics.jsonl"
    path.parent.mkdir()
    rows = [{"episode": 1, "terminated": False}]  # cut off by a resume
    rows += [{**r, "episode": 2} for r in steps(3)]
    rows += [{"episode": 3, "terminated": False}]  # still running
    path.write_text("\n".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")})
                              for r in rows))
    summary = d.summarize(path)
    assert summary["name"] == f"{tmp_path.name}/run"
    assert summary["incomplete_episodes"] == 2 and len(summary["episodes"]) == 1
    keys = {m["key"] for m in summary["metrics"]}
    assert set(summary["episodes"][0]["m"]) == keys and "fn" not in summary["metrics"][0]
    json.dumps(summary, allow_nan=False)


def test_rss_min_gap_matches_formula_with_carla_defaults():
    # 8 m/s behind 8 m/s: 8*1 + 0.5*3.5*1 + 11.5^2/(2*4) - 8^2/(2*8)
    assert d.rss_min_gap(8., 8.) == pytest.approx(22.28125)
    assert d.rss_min_gap(0., 10.) == 0.  # a fast lead never makes a stopped ego unsafe


def test_rss_flags_tailgating_that_ttc_misses():
    # Same speed as the lead (gap constant at 10 m, 10 m/s): TTC never exists.
    tailgate = steps(5, front_hazard_m=10.)
    assert d.min_ttc(tailgate) is None and d.tet(tailgate) == 0
    assert d.rss_unsafe_steps(tailgate) == 4  # first step has no closing speed
    assert d.rss_violations(tailgate) == 1
    assert d.rss_min_margin(tailgate) == pytest.approx(10. - 28.28125)

    safe = steps(5, front_hazard_m=40.)
    assert d.rss_unsafe_steps(safe) == 0 and d.rss_min_margin(safe) > 0


def test_rss_parameters_are_configurable():
    rows = d.prepare([{**r} for r in steps(3, front_hazard_m=10.)], d.TTC_THRESHOLD_S,
                     {**d.RSS, "response_s": 0., "brake_min": 8.})
    # No response time and equal braking: same speed is exactly safe at any gap.
    assert d.rss_unsafe_steps(rows) == 0

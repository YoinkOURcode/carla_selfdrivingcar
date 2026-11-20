# Vision observations

Branch: `vision_integration`, based on published main `e475b55d9`.
The Town02 evaluation now uses the existing, frozen RGB models. No vision or
driving model is trained and checkpoint files are unchanged.

Unknown speed limits remain `None` with confidence 0. The driving fallback is
30 km/h (8.333 m/s), and the Town02 wrapper sets that fallback in new runtime
configs. This does not change the CARLA vehicle's reported speed limit or force
its speed. Confirmed OCR limits and existing safety reductions still apply.

Visibility now uses CARLA fog density in every perception mode:
`min(60, max(5, 100 * (1 - fog_density / 110)))` metres. The environment overrides
the vision provider's legacy 30 m placeholder before building observations or
running safety control. At a 30 km/h base speed, 60 m visibility gives a
28 km/h target, before other hazards reduce it. This is a simulator-weather
heuristic, not a visibility estimate learned from images.

## Source changes

- `dreamerv3_attention/vision.py`: loads the joint road checkpoint and the
  seven-class perception checkpoint once per environment. Converts their
  outputs to the existing 16-feature observation contract.
- `dreamerv3_attention/environment.py`: the provider's `camera_only` flag
  restricts its input to RGB and static camera calibration. Depth, radar,
  collision events and other sensor calibration are excluded. Radar tokens
  and map-filtered radar safety hazards are disabled for this provider;
  camera validity does not depend on radar availability. Camera-only
  providers require `external` mode. Other providers retain their behavior.
- `joint_road_model/road_perception.py`: package-relative imports, while
  preserving its standalone command-line interface. Model inference is
  otherwise unchanged.
- Depth camera and speed-limit OCR (branch `vision-lane-stall-fix`):
  `VisionPerceptionProvider.uses_depth_camera` lets `environment.py` pass the
  depth image and calibration from the depth camera mounted with the RGB camera,
  a stand-in for a real depth sensor. Radar, events and other calibration stay
  excluded. `vision.py` uses that depth for object and lamp positions and
  choosing which sign to read; `speed_sign.py` reads and
  votes on sign numbers. `reset()` clears the lane hold and the speed limit
  each episode.
- Lane acceptance (same branch): `VISION_MAX_EXTRAPOLATION` (default 0.7)
  replaces the fixed 0.5 low-confidence limit, and `VISION_LANE_HOLD_S`
  (default 2.0, 0 disables) holds the last valid lane estimate through dash
  gaps, shifted by the ego's motion relative to the planned route.
- `dreamerv3_attention/requirements-vision.txt`: inference dependencies pinned
  to work with this project's existing torch 2.7.1 and NumPy 1.26.4.
- `dreamerv3_attention/tests/test_vision.py` and `tests/test_environment.py`:
  regression coverage for geometry, channel order, class mapping, missing
  lanes, light conflicts and exclusion of privileged sensors from control.
- `runs/experiment_27/eval_town02{,_all}.sh`: generate external-vision config
  copies, run both existing policies in four conditions, preserve previous
  results, and return a failure code if any evaluation process fails.
  Restart assistance is disabled because signal identity/phase tracking is
  unavailable. Historical input configs and scenario definitions are retained.

CARLA actor/map/weather queries still build `oracle_state` for evaluation,
termination and rewards. Fog-derived visibility is also intentionally supplied
to the safety state; actor/map hazards remain excluded from vision control.
Ego speed/pose and planned route tokens remain as odometry/GPS/navigation
equivalents, as agreed. This is not camera-only localization.

## Known issues (depth/OCR, 2026-11-17, not yet fixed)

A live check on Town02 (Traffic Manager driving the eval route with traffic,
first 129 m only) found:

- **Red-light distance is wrong near junctions.** While the true red stop line
  was 40 m ahead, vision reported a red light at 0 m on all 175 frames where it
  gave a distance. The likely cause is a light the car has already passed:
  anything closer than the stop-line offset is clamped to 0 m, which means stop
  here. Suggested fix: ignore a light whose stop line is behind the car
  (forward position minus offset minus bumper well below 0), then recheck
  against ground truth.
- **Speed-limit OCR is not verified in simulation.** No sign was read on that
  stretch. The map limit there is 30 km/h throughout. The unit tests cover only
  the voting and passing logic.
- **Object positions from depth look good.** The median forward error against
  the true vehicle positions was 0.42 m over 1,215 frames.
- **Visibility:** the environment now uses the capped fog-density formula above.
  Speed scaling linearly interpolates through zero at 0 m, 25/30 at 30 m and
  full speed at 80 m. A 30 km/h fallback gives a 25 km/h target at 30 m,
  16.67 km/h at 20 m and 8.33 km/h at 10 m,
  before other hazards reduce it. This does not estimate actual visibility;
  junctions without lane paint still
  stall the car at about 74 m; launch CARLA without `-quality-level=Low`
  (it crashes at about 98 m).

## Observation limitations

Step metrics include `policy_lane_source` and `policy_observation_invalid_reason`
for the new observation. The corresponding fields under `safety_before`
describe the observation used to choose the applied action.

**These models are insufficient to replace the full original observation.**

| Original information | Vision behavior and remaining gap |
| --- | --- |
| Lane position and heading | Project predicted lane-centre pixels onto a flat road and fit a metric line. From 15 m before a route junction to its exit, use the planned path and ego odometry. Elsewhere, hold the last usable estimate for up to `VISION_LANE_HOLD_S` (2 s by default), shifted by the change in the route-relative lateral offset and heading since that estimate (unchanged when no route is available); a frozen value let the policy drive up to 2.3 m across the lane in experiment_28 eval2. The route heading uses the path's first 5 m segment, so it turns slightly before a bend (eval4 episodes 2 and 3 cut a bend during a hold). Odometry-yaw heading was tried and was worse: it ignores the bend itself (eval6, 0/6). After the hold expires, use a fresh route estimate if the ego is within 2 m of the local route start and aligned within 40°, so a car that drifted while vision was lost recovers instead of braking until the inactivity timeout. This fallback caps target speed at 1.5 m/s and holds behind a nearby lead vehicle on red. A vision offset more than 2 m from the aligned route's offset (or, with no aligned route, from the held estimate) is rejected and handled like a lost lane: on Town02 vision sometimes locks onto the neighbouring lane and reads 3-6 m off at the same spots every run, while correct readings stay within 0.75 m and the route within 0.14 m. The route, not the hold, is the reference so that one wrong reading cannot seed a hold that rejects the correct ones. An unavailable lane reference or other invalid perception still triggers full braking. |
| Object positions | Bottom centre of each vehicle/pedestrian box placed at the median CARLA depth of the box (`associate_depth`), with measured depth and its spread in the depth/uncertainty channels. Without depth: flat-road projection as before, and truncated boxes or feet outside the projection invalidate the observation. |
| Object velocity | Unavailable; velocity channels are zero placeholders. No tracker or motion model is introduced. |
| Speed-limit value | The nearest sign on our side is read with RapidOCR (`speed_sign.py`, taken from `perception/live_depth`), at most every 0.1 s. A value needs twice the readings of any other value and at least 3 readings to lower the limit, or 6 to raise it or set the first one (a false raise means speeding; a missed raise only means driving slower). It applies once the sign leaves view within 15 m (passed). Confidence is the vote share. Before the first sign, or without `rapidocr_onnxruntime`, the value stays unknown and the default speed applies. |
| Traffic lights | The camera decides colour; the stop position comes from navigation. `environment.py` passes `route_stop_lines`: a transverse segment at each signalised stop on the planned path (the stop stations the route already uses for violation checks), in ego coordinates, never signal state. Each station moves CARLA's stop waypoint (the trigger area, 7-13 m before the junction on Town01, median 9.1 m, which left the car about 11 m short of the intersection) towards the junction entry, but no closer than 2 m before the signal pole: Town01 heads stand 1.6-2.9 m before the junction and the 90 degree camera loses them under ~1.6 m ahead of the bumper, and a held red is only released by seeing green. The car now rests about 4 m before the head, 5.6-6.6 m before the junction. Distance is measured along the path from the actual front bumper (bounding box). Paint detection (`stop_lines.extract_stop_lines`) is no longer called: Town01 signalised stops have no painted line, and its fixed brightness threshold marks most of a sunlit road as paint. `traffic_signals.py` picks the lamp head nearest the path and the stop line, and retires the line and its light once the bumper passes it. A tracked red is not released by another lamp's green, but the lock yields to a head that fits the stop clearly better (association score margin 1.0; an off-path lock never fits): far out, the tracker can lock onto a red head on the far side of the junction or on another approach where the route turns, and the old absolute rule then held red through every later green (experiment_28: 7 of 19 green waits timed out; replay of the recorded detections releases all 7, red frames unchanged). A red/yellow with no route stop on the planned path (80 m) is treated as another road's light; previously it meant an unknown range and an immediate full stop, which halted the car up to ~75 m early. Without route data the range stays unknown and red still stops. Light and stop-line rows go to the policy as typed tokens (column 15 = -2 light, -3 stop line), excluded from hazards; the frozen policy is not trained on them. Restart on green is not inferred. |
| Crossings | Bounding-box ground projection provides approximate distance, with a conservative fixed 2.5 m bumper allowance. Pedestrian proximity estimates occupancy. A single visible-crossing identifier is used; reliable multi-crossing tracking is unavailable. |
| Side/rear hazards | Front camera only; no complete side/rear coverage and no radar fallback. |
| Weather and visibility | Models have no weather estimator. Rain/wetness/night retain neutral placeholders. The environment supplies visibility from CARLA fog density, bounded to 5–60 m, overriding the provider's legacy placeholder. |
| Metric lane width | Existing nominal 3.5 m state default remains; not a measured width. |

Ground projection assumes a level front camera, known intrinsics/mount and a
flat road at ego z=0. The mount height is relative to the vehicle origin, so
vehicle ride height and pitch introduce additional error. Nonzero configured
camera rotation is rejected. The supplied lane calibration assumes the
original camera geometry; it is not recalibrated or retrained here.

Object tokens retain their shape: estimated xyz/100, zero velocity, pedestrian
and child labels, position-valid flag, estimated range/100 and detector
confidence. Measured depth and uncertainty come from the depth camera when it is available. The existing
driving policy was trained with different observation semantics, so checkpoint
compatibility does not imply good driving performance.

The perception package reports provisional class mapping, weak yellow-light
detection, possible conflicting light colours and no clean independent test
score. Detection absence does not prove a clear road or a green light.

## Reproduce

From the repository root, with the existing `venv` and matching CARLA build:

```bash
venv/bin/python -m pip install -r dreamerv3_attention/requirements-vision.txt
venv/bin/python -m pytest dreamerv3_attention/tests/test_vision.py dreamerv3_attention/tests/test_environment.py dreamerv3_attention/tests/test_sensors.py -q
```

Required files, supplied separately where ignored by the repository:

- `joint_road_model/joint_road_model.pt`
- `joint_road_model/lane_center_calibration.json`
- `perception/carla_yolo26_demo_results/mac_demo/best.pt`
- `runs/experiment_26/curriculum/stage7_run_10/latest.pt` (existing driving policy)

Use a dedicated, idle server loaded with Town02. For a new server:

```bash
UE4_ROOT=/path/to/UnrealEngine_4.26 bash data/start_server.sh Town02 -RenderOffScreen -nosound
```

Do not start another server on occupied ports. The wrapper expects Town02 on
port 2000; `CARLA_PORT=2010` selects another already-running Town02 server.

```bash
bash runs/experiment_27/eval_town02_all.sh
```

The default output is a fresh timestamped directory under
`runs/vision_integration/`. To select it explicitly:

```bash
EVAL_OUTPUT_ROOT=runs/vision_integration/my_vision_eval bash runs/experiment_27/eval_town02_all.sh
```

The directory must not already exist. Each policy (`results_controller`,
`results_run_10`) gets `clear`, `wet`, `night`, `lowvis` results, logs, traces
and generated config copies. Existing per-condition budgets remain 1500 wall
seconds; override with `EVAL_BUDGET_SECONDS`. `VISION_DEVICE=cpu` forces CPU
vision inference; default is CUDA when available. No training command runs.

For other evaluation entry points, use a config with
`"perception": "external"` and
`"provider": "dreamerv3_attention.vision:VisionPerceptionProvider"`.
Historical configs elsewhere still explicitly select ground truth; this change
does not silently rewrite historical training experiments.

## Validation and evaluation

- Focused observation/provider/sensor tests: 141 passed, including a real
  `SensorFrameBuffer` calibration round-trip regression.
- Saved sample frame: both checkpoints loaded and inference completed; road
  parameters frozen. Observed lane geometry and red-light output were finite.
- Final full offline suite with host socket access: 1254 passed, 1 skipped,
  3 failures. All three also fail against the committed main configuration
  module and reference historical curriculum fixtures: non-increasing step
  counts, differing restart
  settings and missing `stage5_pedestrian_runtime.json`. The viewer test passes
  with host socket access. No existing assertions were weakened.
- `bash -n` on both evaluation scripts and `git diff --check`: passed.
- `graphify update .`: completed AST update. Existing parser warnings remain
  for 61 C++/other source files and one empty `pyproject.toml` extraction.

Final Town02 run: `runs/vision_integration/town02_vision_002`. The exact command
was:

```bash
EVAL_OUTPUT_ROOT=runs/vision_integration/town02_vision_002 YOLO_CONFIG_DIR=/tmp MPLCONFIGDIR=/tmp/carla-vision-mpl bash runs/experiment_27/eval_town02_all.sh
```

All eight evaluation processes completed. **None completed the route**; all
ended with `inactivity_timeout` after invalid lane observations triggered
braking. There were no collisions, but the car never reached the first authored
encounter at 30 m, so these results do not validate obstacle/light handling.

| Condition | Controller completion | Saved policy completion |
| --- | ---: | ---: |
| Clear | 1.050% | 1.057% |
| Wet | 1.050% | 1.047% |
| Night | 1.049% | 1.048% |
| Low visibility | 0.024% | 0.024% |

The car travelled about 7.2 m in clear/wet/night and under 0.05 m in low
visibility. Re-inference of the saved clear final-frame JPEG produced a
`left_only`, `low_confidence=True` lane estimate with extrapolated fraction
0.540. The joint road estimator flags extrapolation beyond half the road area
as low confidence. That is a model-confidence limitation, not a reason to
restore simulator observations. Vision weights and thresholds were unchanged.

`summary.json`, per-condition `result.json`, `trace.jsonl`, `live.json` and logs
are retained locally under the final output directory. Traces confirm external
vision configuration, empty radar safety hazards, and unknown numeric speed
limits throughout all eight episodes. `ALL_DONE` means processes finished,
not successful driving.

The earlier `town02_vision_001` run is retained for diagnosis only: a calibration
nesting error initially made all observations invalid. It was fixed and the
complete evaluation was rerun; use `_002` for final results.

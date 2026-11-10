# Vision observations

Branch: `vision_integration`, based on published main `e475b55d9`.
The Town02 evaluation now uses the existing, frozen RGB models. No vision or
driving model is trained and checkpoint files are unchanged.

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
termination and rewards. They do not supply the vision provider or its safety
state. Ego speed/pose and planned route tokens remain as odometry/GPS/navigation
equivalents, as agreed. This is not camera-only localization.

## Observation limitations

**These models are insufficient to replace the full original observation.**

| Original information | Vision behavior and remaining gap |
| --- | --- |
| Lane position and heading | Project predicted lane-centre pixels onto a flat road and fit a metric line. Missing or low-confidence lanes invalidate the observation and trigger full braking; there is no map fallback. Curves, hills and intersections remain difficult. |
| Object positions | Project the bottom centre of vehicle/pedestrian boxes onto a flat road. These are approximate positions, not measured depth. Truncated boxes or feet outside the supported projection invalidate observations. |
| Object velocity | Unavailable; velocity channels are zero placeholders. No tracker or motion model is introduced. |
| Speed-limit value | Detector only recognizes the sign class, not its number. Value stays unknown (`None`, confidence 0); existing default-speed behavior applies. |
| Traffic lights | Any detected red takes priority over yellow/green. Stop-line distance and route/lane association are unknown. Existing safety logic stops on an identified red/yellow with unknown range, including potentially irrelevant lights. No green-light restart is inferred. Missed lights can still cause violations. |
| Crossings | Bounding-box ground projection provides approximate distance, with a conservative fixed 2.5 m bumper allowance. Pedestrian proximity estimates occupancy. A single visible-crossing identifier is used; reliable multi-crossing tracking is unavailable. |
| Side/rear hazards | Front camera only; no complete side/rear coverage and no radar fallback. |
| Weather and visibility | Models have no weather estimator. Rain/wetness/night retain neutral placeholders; visibility uses a fixed conservative 30 m assumption, not simulator weather. |
| Metric lane width | Existing nominal 3.5 m state default remains; not a measured width. |

Ground projection assumes a level front camera, known intrinsics/mount and a
flat road at ego z=0. The mount height is relative to the vehicle origin, so
vehicle ride height and pitch introduce additional error. Nonzero configured
camera rotation is rejected. The supplied lane calibration assumes the
original camera geometry; it is not recalibrated or retrained here.

Object tokens retain their shape: estimated xyz/100, zero velocity, pedestrian
and child labels, position-valid flag, estimated range/100 and detector
confidence. Measured-depth/uncertainty channels remain unavailable. The existing
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

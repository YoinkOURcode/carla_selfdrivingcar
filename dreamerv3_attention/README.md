# CARLA DreamerV3 with attention

This package composes the installed TorchRL DreamerV3 implementation with a
small multimodal encoder, CARLA sensors, route navigation, a safety supervisor,
replay, checkpoints and scenario evaluation. It controls CARLA actors only.
The deleted legacy `world_model_rl` source and existing simulator/data edits
are preserved.

## Verified status

CPU tests verify sensor math and synchronization, perception contracts, safety
and rewards, real TorchRL model/actor/critic gradient updates, replay, exact
optimizer continuation, evaluation metrics and the command-line smoke cycle.
The synthetic fixture is a test, not a trained driving agent.

Live Town01 RGB/depth/radar delivery, synchronized frames and actor cleanup
have passed the opt-in integration test. A bounded three-episode collection
recorded 384 transitions with valid continuous sensors and no collisions.
Resuming that replay for three further episodes completed 96 world-model-only
updates, with finite losses and unchanged actor/critic/target parameters. The
checkpoint contains 768 real transitions. This short run was mostly stationary
and does not validate driving competence or model generalization. Artifacts:
`runs/dreamerv3_luna_20261029_1534/`.
GPU inspection requires host access in this sandboxed session.
Map annotation coverage, scenario placements, physical stopping distances,
learned driving, reward calibration and attention benefit remain unverified.
Full acceptance requires further approved live experiments. Unsupported map,
route, child-blueprint and scenario capabilities are reported as such.

## Setup and offline checks

The tested installation is Python 3.10.22, CARLA 0.9.16, PyTorch 2.7.1
(CUDA 12.6 build), TorchRL 0.14.1 and TensorDict 0.14.3. Keep client/server
CARLA builds identical. These pins describe the installed environment; the
CPU tests establish compatibility for the exercised eager execution path.

```bash
venv/bin/python -m pip install -r dreamerv3_attention/requirements.txt
venv/bin/python -m dreamerv3_attention preflight
venv/bin/python -m pytest dreamerv3_attention/tests -q
venv/bin/python -m dreamerv3_attention smoke --config dreamerv3_attention/configs/smoke.json
```

Smoke writes a unique directory under `/tmp`, with configuration, JSONL
metrics, a replay-inclusive checkpoint and a result JSON. It collects 16
synthetic transitions and performs three real Dreamer updates on CPU.

## Command reference

Run commands from the repository root with `venv/bin/python -m
dreamerv3_attention`. The complete CLI is:

```text
python -m dreamerv3_attention {smoke,train,collect,validate,evaluate,infer,compare,preflight,viewer,audit-controls}
  [--config CONFIG] [--output OUTPUT] [--checkpoint CHECKPOINT]
  [--world-model-only] [--baseline BASELINE] [--candidate CANDIDATE]
  [--data-checkpoint DATA_CHECKPOINT]
  [--budget-seconds SECONDS] [--ue4-root UE4_ROOT] [--web-port PORT]
```

| Command | Purpose | Required or relevant arguments |
| --- | --- | --- |
| `preflight` | Print dependency versions, CUDA availability and the live-validation status. Does not start CARLA or create a run. | None |
| `smoke` | Run a bounded CPU-only synthetic collection and learner update; verify the saved checkpoint. This is a software smoke check, not a driving evaluation. | `--config configs/smoke.json`; optional `--output`, `--checkpoint` |
| `train` | Collect from CARLA and perform joint Dreamer updates. | `--config`; `--budget-seconds`; optional `--checkpoint`, `--output`, `--ue4-root`; `--world-model-only` trains only the world model |
| `collect` | Collect CARLA transitions and save replay/checkpoints without learner updates. | `--config`; `--budget-seconds`; optional `--output`, `--ue4-root` |
| `validate` | Compute one held-out world-model loss batch on CPU, without optimizing. | `--checkpoint` for the model and `--data-checkpoint` from a separate `validation` or `held_out` collection |
| `evaluate` | Evaluate a checkpoint on the configured validation/held-out scenarios, or run the diagnostic route controller on the same suite. | `--config`, `--budget-seconds`; actor requires `--checkpoint`; controller uses `--controller` and optional `--controller-speed`; optional `--output`, `--ue4-root` |
| `infer` | Run one checkpoint-loaded actor evaluation episode. | `--config`, `--checkpoint`, `--budget-seconds`; optional `--output`, `--ue4-root` (`--controller` is evaluate-only) |
| `viewer` | Observe active training or evaluation; optionally launch a separate checkpoint drive. | Monitoring: optional `--training-run`, `--web-port` (8080). Driving: `--checkpoint`, matching `--config`, `--budget-seconds`; optional `--ue4-root` |
| `compare` | Compare two evaluation result JSON files and print the comparison JSON. | `--baseline` and `--candidate`; optional `--output` to save the comparison |
| `audit-controls` | Summarize an existing JSONL control trace offline. | `--trace`; optional new `--output` path |

`--config` defaults to `dreamerv3_attention/configs/attention.json`.
Live commands require `--budget-seconds` (greater than 0 and at most 86400).
The viewer requires a time budget for driving; clicking **Start drive** starts
the live run.
`--ue4-root` starts and owns a server; without it, the command
connects to an existing idle dedicated server. Outputs default to a unique
directory under `/tmp` for `smoke`, and under `runs/` for live work. Pass
`--output` to choose a Vpath; populated output directories are never overwritten.
`--world-model-only` is valid only for `train`. `--baseline`, `--candidate`,
and `--data-checkpoint` apply only to their corresponding commands.

Examples of non-live commands:

```bash
venv/bin/python -m dreamerv3_attention --help
venv/bin/python -m dreamerv3_attention preflight
venv/bin/python -m dreamerv3_attention smoke --config dreamerv3_attention/configs/smoke.json
venv/bin/python -m pytest dreamerv3_attention/tests -q
```

The opt-in live integration test is separately gated and skipped by default:

```bash
DREAMERV3_APPROVED_LIVE_TEST=1 venv/bin/python -m pytest dreamerv3_attention/tests/test_live.py -q
```

## Application workflow

### Local web viewer

Start the dashboard, then open **http://localhost:8080**. **Live run**
automatically follows active CARLA training, collection or evaluation under `runs/`:

```bash
venv/bin/python -m dreamerv3_attention viewer
```

Run `evaluate` or `infer` in another terminal with its usual checkpoint, config
and budget. The viewer shows the evaluation drive, switches scenarios
automatically, and keeps the final camera frame when the evaluation ends.

This mode needs no checkpoint or driving budget. It observes the running
process's RGB camera and applied controls without connecting another client
to CARLA. It shows route progress, episode return, reward, learner updates,
losses and action source (actor, controller warmup/refresh or random warmup).
Snapshots update at up to five frames per second between learner updates;
the page reports feed age when updates delay new driving frames. Episode
resets are followed automatically. Attention overlays are unavailable on
training frames because inference used the preceding observation.

Use `--training-run runs/NAME` to select a specific training or evaluation run, including runs outside
`runs/`. With no active publisher the page reports idle or shows the last
captured frame labeled as ended. Monitoring requires a run started with
this version: an already-running older process cannot publish camera frames
until its next approved launch. The viewer never starts or stops the monitored run.
Each new live run keeps only its latest snapshot in `live.json`, beside a
held `live.lock`; historical metrics and checkpoints are unchanged.

To launch a separate checkpoint drive, select **Destination drive** and start
the viewer with a checkpoint and a driving budget:

```bash
venv/bin/python -m dreamerv3_attention viewer \
  --config runs/dreamerv3_train_20261029_204710_12171/config.json \
  --checkpoint runs/dreamerv3_train_20261029_204710_12171/latest.pt \
  --budget-seconds 180
```

For a generic drive, choose distinct map spawn indices for the start and
destination, then click **Start drive**. **Stop** ends the drive and removes
its actors; Ctrl+C also
closes the dashboard. `--web-port` changes the HTTP port.
Without a time budget or a checkpoint the page
opens for inspection but driving is disabled. Use an
idle dedicated server, or add `--ue4-root /path/to/UnrealEngine_4.26` to launch
one. Each drive holds the shared workload lock, uses the supplied wall-clock
budget including initialization, and records config/final telemetry in a
unique `runs/dreamerv3_viewer_*/` directory. No training is performed.

With a named scenario config such as `curriculum/evaluate_lights.json`, the
viewer resolves and runs exactly that catalog record, including its actor,
traffic and weather settings. It uses the scenario's route instead of picker
overrides, then reports the actual selected endpoints after reset. Driving
requires one selected case; use an `evaluate_*.json` config rather than a
multi-task stage config. Restart an existing viewer to load code changes.

Demo configs keep the picker endpoints and validate that route against the
demo's length, curve and traffic-light requirements. An unsupported route
reports an error instead of substituting endpoints. The Town02 demo defaults
to start 2 and destination 56. Final telemetry includes the termination or
truncation reason, active safety events and measured lane offset; the dashboard
shows the reason when an episode ends.

The viewer uses the checkpoint's Dreamer policy on CPU and the existing safety
supervisor. It shows front RGB, projected object centres and classifications,
optional per-object attention, speed/limit in km/h, measured traveled distance,
route distance remaining, actual steering/throttle/brake, and interventions.
Choose the checkpoint's config so observation sizes and attention match.
The example checkpoint disables attention; an attention-enabled checkpoint
shows actual fusion weights when **Show object attention** is checked.

Ground-truth objects are labeled simulator data, can include occluded actors,
and are not camera detections. External/mixed perception uses the configured
provider. The table lists up to 20 valid policy object tokens, including those
outside the camera; overlays show projected centers only. Radar tokens are
listed separately by distance and attention, without object identities. The
viewer also reports red/yellow signal state from policy inputs; camera-based
traffic-light recognition is not yet provided. Slot numbers are not persistent
tracking IDs. Projection markers
are not detection bounding boxes. Attention is unavailable when the model has
it disabled. Destination arrival is reported only when the environment marks
success; the viewer does not establish that a checkpoint can complete a route.

### Training and evaluation

1. **Install and check the environment.** Install `requirements.txt`, run
   `preflight`, then run the offline tests and synthetic `smoke`. These do not
   require a CARLA server.
2. **Choose a configuration.** `configs/baseline.json` disables attention;
   `configs/attention.json` enables it. They are matched in other settings.
   `configs/evaluation.json` selects held-out evaluation data, while
   `configs/smoke.json` is deliberately bounded for CPU use. Configurations
   control the observation/model sizes, route/scenario split, sensor and
   perception choices, training duration and output behavior.
3. **Prepare a live run.** Use an idle dedicated CARLA server with matching
   client/server builds, or provide `--ue4-root` to launch one. Select a unique
   output directory and set its wall-clock budget. The workload lock serializes
   simulator and GPU use; preflight checks happen before run data is created.
4. **Collect or train.** Use `collect` to build replay without optimization,
   or `train` for collection plus joint model, actor and critic learning. A
   later `train --checkpoint ...` resumes a checkpoint. For world-model-only
   fitting, collect a train-split replay first and use `train
   --world-model-only` with a compatible config and checkpoint.
5. **Validate learned dynamics.** Collect separate data with the config split
   set to `validation` or `held_out`; pass that checkpoint to `validate` as
   `--data-checkpoint`, alongside the trained model checkpoint. Validation is
   CPU-only and does not change model parameters.
6. **Evaluate and compare.** Run `evaluate` for the configured suite or `infer`
   for one episode. For an attention ablation, evaluate baseline and attention
   checkpoints on matching scenarios, seeds, routes and configs, with only
   attention changed; then pass their `result.json` files to `compare`.
7. **Inspect artifacts.** Each run directory contains its config, metrics,
   checkpoints and `result.json`. Live experiment registrations are appended
   to `world_model_rl/orchestration/experiment_registry.json`. Preserve those
   artifacts; use a new output path for another run.

This workflow describes the available software path, not evidence that a
policy drives well. Simulator results, held-out routes and matched experiments
are needed to assess driving performance or an attention benefit.

For a matched controller check, run the actor and controller with the same
validation config, suite, budget and safety settings, then compare their result
files. The controller run does not load a checkpoint:

```bash
venv/bin/python -m dreamerv3_attention evaluate --config dreamerv3_attention/configs/evaluation.json \
  --checkpoint runs/ACTOR/latest.pt --budget-seconds 600 --output runs/actor_eval
venv/bin/python -m dreamerv3_attention evaluate --config dreamerv3_attention/configs/evaluation.json \
  --controller --budget-seconds 600 --output runs/controller_eval
venv/bin/python -m dreamerv3_attention compare --baseline runs/controller_eval/result.json \
  --candidate runs/actor_eval/result.json --output /tmp/controller_actor_comparison.json
```

By default the controller follows trusted speed limits, with the configured
fallback when a limit is unavailable. Add `--controller-speed 3` only when
the matched comparison should use a 3 m/s cap.

An offline audit reads prior traces without CARLA or a checkpoint:

```bash
venv/bin/python -m dreamerv3_attention audit-controls \
  --trace runs/EXISTING_EVAL/trace.jsonl --output /tmp/control_audit.json
```

The audit separates clear-road inactivity from invalid observations and
mandated stops, and summarizes commands, overrides, target speed, lane and
heading error, reward components and acceleration when recorded. Missing fields
remain unavailable. No-override or weak-command evidence does not establish a
throttle or vehicle-physics cause. If the controller also fails, investigate
route geometry, control and vehicle response first. If it succeeds while the
actor stalls, inspect corrective-supervision coverage and alignment on the
actor's own valid clear-road states; assisted runs should remain distinct from
actor-only evaluation.

## Architecture and upstream boundary

```text
1280x720 RGB + aligned depth + four radars + measured ego + assigned route
  -> exact-frame bundle + calibration + validity
  -> GT / external / explicitly mixed perception
  -> resized uint8 RGB + normalized navigation/object/radar tokens
  -> upstream image encoder + one-query masked attention (or masked mean)
  -> upstream categorical RSSM -> image/state reconstruction, reward, continuation
  -> upstream imagined rollouts -> actor + two-hot critic + slow target
  -> requested action -> safety supervisor -> CARLA
                              | executed action + next observation
                              v
                 TorchRL sequence replay -> upstream Dreamer learner
```

`model.py` reuses TorchRL's image encoder/decoder, RMS MLPs, categorical
prior/posterior/rollout, online state estimator, two-hot heads, model/actor/value
losses, optimizer, target updater and optimization stepper. The reference is
[TorchRL DreamerV3](https://docs.pytorch.org/rl/main/reference/dreamer_v3.html)
and its linked official recipe. Separate dynamics/representation KL weights
(1.0/0.1), 1 nat free bits, 0.01 unimix, continuation prediction, return
normalization and a slow distributional critic are enabled explicitly.

Project deviations: a bounded TanhNormal continuous actor; a small configurable
GRU model; single-query multimodal pooling; normalized structured reconstruction
alongside RGB; the optional upstream replay-value auxiliary term disabled;
eager float32 optimization without compilation or mixed precision. A short
custom collection loop stores the *executed* supervisor action and per-step
safety metadata. Training samples warm-start from the stored current posterior;
latents are not refreshed in replay. Episode identifiers prevent sequences
crossing resets. These are research choices, not a paper-reproduction claim.

Single-query cross attention avoids a full token-to-token attention matrix.
Masked mean pooling keeps identical sensors, projections, image processing,
latent sizes, objectives and schedules for the baseline. A temporal transformer
would duplicate the RSSM's role and cost more memory; several attention stacks
would add untested complexity. Attention weights (`agent.encoder.last_attention`)
are diagnostics, not causal explanations. Fusion usefulness requires matched
multi-seed driving measurements.

## Observation and perception contracts

- World/ego axes follow CARLA: ego x forward, y right, z up. Positions are
  metres; speeds m/s; timestamps simulation seconds; camera/radar mount angles
  degrees; radar detection angles radians before conversion. CARLA speed limits
  are converted from km/h once at acquisition.
- RGB and depth share the configurable pose/FOV/tick and native **1280x720**
  resolution. Model images default to **64x64 CHW uint8**. Intrinsics,
  extrinsics, per-sensor frame/time and validity are retained in the bundle.
- Depth decodes packed CARLA values to metres. Projected actor boxes associate
  aligned depth, subject to a geometry consistency check; invalid/occluded
  associations remain missing. Median crops alone cannot prove visibility.
- Radar uses native supported front/left/right/rear sensors and full CARLA
  mount transforms. Rows contain ego xyz and radial velocity; radial velocity
  is not a full velocity vector. Default 30-degree horizontal FOVs leave large
  blind sectors: `effective_radar_coverage()` reports them. Nearby hits are
  sorted, bounded and deduplicated within each radar; cross-radar fusion does
  not assert object identity.
- Model input is `pixels`, `pixels_valid`, `tokens[N,16]`, `mask[N]` and
  `radar_valid[4]`, with `N = 36 + max_objects`. Tokens are context, three
  navigation points 5/15/30 m ahead, bounded objects, then 8 hits per radar.
  Missing token rows are zero/masked; finite checks run before projections.
- Context fields: speed/30, limit/30, limit confidence, lane error/4,
  heading error/pi, remaining route/1000, rain, wetness, night, visibility/100,
  crossing distance/100 (1 if absent), occupied crossing, red light,
  pedestrian risk, authored child risk, observation validity.
- Object fields: xyz/100, relative xyz velocity/30, pedestrian flag,
  authored-child flag, position validity, GT range/100, confidence, reserved,
  measured depth/100, depth validity, uncertainty metres/100, reserved.
  Navigation tokens use xyz/100 and field 15 = -1. Fields 3 and 4 carry red
  stop-line distance/100 and its validity; fields 5, 6 and 7 carry yellow
  presence, stop-line distance/100 (-1 if unknown), and confidence. These fields
  change observations for existing checkpoints. Radar xyz/100 and radial
  velocity/30 use the first four fields. Padding has no classification meaning.

`sensors.PerceptionProvider.predict(FrameBundle)` returns `PerceptionPrediction`
with exact frame/time, normalized object rows, validity, optional metre depth
and uncertainty arrays, and `driving` replacements named after `DrivingState`.
The provider receives sensor data and calibration, never an oracle object set.
Construct it through a no-argument `module:factory` in `Config.provider`.
No detector framework or weights are bundled or downloaded.

Choose `perception: ground_truth`, `external` or `mixed`. Full external mode
must supply speed limit/confidence, lane/heading error, crossing ID/distance/
occupancy, red light state/distance, `yellow_light` (bool),
`yellow_light_distance_m` (front-bumper to route stop line in metres, or None),
`yellow_light_confidence` (0 to 1), visibility and observation validity;
missing fields force a stop. A reported yellow requires confidence greater than
zero; invalid confidence, distance, or simultaneous red and yellow fails safe.
All positive confidence values receive the same response; no uncalibrated
confidence threshold is assumed. Providers must associate the light with the
route and supply stop-line distance, not distance to the light itself. Object detections also drive safety hazard state.
Mixed mode requires a nonempty explicit `mixed_fields` list; other state fields
remain GT and are labeled mixed. Objects come from the external provider.
External depth can replace/supplement object depth without changing the model.
`MockPerceptionProvider` produces no detections and no safety replacements, so
full external mode stops conservatively. GT object geometry is privileged and
includes occluded actors; it is not labeled camera detection.

Only exact current frames enter observations. Slower sensors have explicitly
invalid skipped frames; older frames are not reused. Camera or directional
radar loss triggers braking. Depth loss is represented per object; GT geometry
still provides distances. Late collision events are retained separately and
attributed when received. Callbacks do not leak future events. Original raw
RGB/depth arrays are not duplicated into replay. Configure `drop_sensors`,
scenario `stale_sensors`, or metre-valued `sensor_noise_std` for robustness.

## Live commands

Every live command requires an explicit wall budget.
GPU preflight allows desktop compositors and known desktop GPU renderer
processes. When connecting to an existing server, it allows the CARLA game
process only when its command identifies this repository's `.uproject` and
the configured local RPC port (default 2000). A different Unreal project,
server port or training process still blocks the run. `--ue4-root` launches
a new server and therefore cannot reuse an already running simulator.
The guard preserves unrelated processes and holds
`/tmp/carla-training-gpu.lock` using `flock`. The lock file is never deleted.
GPU preflight failures do not create an output directory; an empty directory
left by an earlier failed preflight may be reused. Populated output directories
are always preserved and require a different run path.
Only use a dedicated server with no other ticking client. A timeout checkpoints
training and cleans owned actors. Resume begins a fresh simulator episode and
keeps replay, parameters, targets, optimizer, counters and RNG; it does not
promise the same subsequent CARLA trajectory.

The existing `data/start_server.sh` adds `-carla-rgb-only`. In this checkout the
flag disables semantic/instance annotation proxies; live testing confirmed RGB,
scene depth and radar still work. It does not establish semantic-camera support.
Use an idle dedicated server, or the runner's owned-server option, which omits
the flag. `--ue4-root` points to the built Unreal Engine directory. Startup and
all simulation/learning share one workload lock. Never run a second job.

```bash
# Run only after approving the experiment and budget; unique output required.
venv/bin/python -m dreamerv3_attention train --config dreamerv3_attention/configs/attention.json --output runs/dreamer_attention_001 --budget-seconds 600 --ue4-root /path/to/UnrealEngine

# Collection uses the same replay/checkpoint contract, with no learner updates.
venv/bin/python -m dreamerv3_attention collect --config dreamerv3_attention/configs/attention.json --output runs/dreamer_collection_001 --budget-seconds 120

# Resume collected replay and update only the upstream world-model objective.
# extended.json must keep the collection config unchanged except a higher steps limit.
venv/bin/python -m dreamerv3_attention train --world-model-only --config /tmp/extended.json --checkpoint runs/dreamer_collection_001/latest.pt --output runs/dreamer_world_model_001 --budget-seconds 600

# Raise steps in a copied config to extend a run; use a fresh result directory.
venv/bin/python -m dreamerv3_attention train --config /tmp/extended.json --checkpoint runs/dreamer_attention_001/latest.pt --output runs/dreamer_attention_002 --budget-seconds 600

venv/bin/python -m dreamerv3_attention evaluate --config dreamerv3_attention/configs/evaluation.json --checkpoint runs/dreamer_attention_001/latest.pt --output runs/dreamer_eval_001 --budget-seconds 600
```

For `collect` and `train`, add a `scenarios` list to a copy of your config
to cycle through conditions in order, switching after each completed episode:

```json
"scenarios": ["train_heavy_rain", "train_low_light", "train_combined_weather_hazard"]
```

These use `HardRainNoon`, `ClearNight`, and `HardRainNight`, respectively.
Keep `scenario` at its default, `"basic"`, when using the list. Names must
belong to the configured split; a custom `scenario_file` can supply your own
weather presets, routes and actor placements. Check that the authored spawn
indices exist on your map. Training collection varies routes by default:
each scenario cycles through up to 16 seeded spawn/destination pairs, chosen
from at most 128 candidates (2,048 for traffic-light tasks) and checked against
its geometry, crossing and speed-limit requirements. Routes must be at least
30 m long. The pool repeats;
actor placements, blueprints and native navigation seeds vary per episode.
Authored hazard offsets and weather retain the scenario's intent. If only one
route qualifies, collection warns; if none qualify, it reports unsupported.
Set `"randomize_routes": false` to keep authored routes. Evaluation and
validation/held-out collection keep their fixed routes. Training excludes
other splits' catalog endpoint pairs and their reverses; this does not ensure
that the routes use different roads.
The list repeats, sharing one replay buffer and one total step/time budget.
`metrics.jsonl` records each transition's scenario, weather, actual `route_start`,
`route_end` and `episode_seed`. The completed-episode count determines the
route variant and actor seed, including on resume. Episode cycling
does not guarantee equal frame counts when episodes have different lengths.
Resume with the same list and config and a larger `steps` limit: completed
episodes advance the cycle, while an interrupted episode restarts its scenario.
An empty or omitted list retains the single `scenario` behavior. Evaluation
continues to use the configured split's full scenario suite.

`infer` runs one checkpoint-loaded evaluation episode. `compare --baseline
BASE/result.json --candidate CAND/result.json` requires measured matching
scenarios/seeds/routes/configs with only attention differing. For the baseline,
use `configs/baseline.json` and a copy of evaluation.json with attention false.
Records include actual checkpoint paths. Live runs append to the shared
`world_model_rl/orchestration/experiment_registry.json`, without restoring any
legacy source, and create unique `runs/` outputs.

For a fixed training task, `baseline` runs a pure-pursuit controller using the
packed policy observations and the same safety controller as Dreamer. It does
not fit a model. Both commands below evaluate exactly the configured collection
scenario; `--fixed-scenario` labels the result as a training sanity check,
not held-out performance. Multiple configured scenarios are rejected.

```bash
venv/bin/python -m dreamerv3_attention baseline --config runs/experiment_17/pilot_config.json --output runs/experiment_17/controller_baseline --budget-seconds 180
venv/bin/python -m dreamerv3_attention train --config runs/experiment_17/pilot_config.json --baseline-result runs/experiment_17/controller_baseline/result.json --output runs/experiment_17/pilot --budget-seconds 900
venv/bin/python -m dreamerv3_attention evaluate --fixed-scenario --config runs/experiment_17/pilot_config.json --checkpoint runs/experiment_17/pilot/latest.pt --output runs/experiment_17/pilot_evaluation --budget-seconds 180
```

Run live commands only with an approved experiment budget. The pilot gate
requires a successful, collision-free controller result with matching settings.
If the baseline fails, inspect its `trace.jsonl` before starting training.
Traces and training metrics include `safety_before` (the actual controller
input) and `radar_before` (selected relevant radar point and clearance per
sensor). All raw radar detections remain in the policy observations.
Radar hazard selection excludes returns within 0.1 m of the mapped driving-road
surface, using its local plane to account for slopes and vehicle pitch. Unknown
road geometry retains the return. This tolerance can also exclude very low debris;
actor and external-object hazards still apply independently.
Oracle hazard fields remain separate. Fixed evaluations report stopped time,
actual action changes, blocked throttle requests and required average speed.
Use `compare --fixed-scenario --baseline OLD/result.json --candidate NEW/result.json`
for checkpoints evaluated on the same fixed task, including the same rewards.

New configs and checkpoints store resolved reward, driving and sensor defaults.
Resume compares only resolved reward settings; scenarios, speed caps and other
settings can change without extra flags. Incomplete saved reward metadata still
blocks resume with an explicit config; reading it without a config remains
possible. Basic config validation and strict model loading remain active.
Schema and dependency versions are recorded but no longer checked on restore.

Resume retains model, optimizer, RNG, replay and progress. Old replay keeps its
original observations, rewards and supervision labels, including removed
scenarios. Increasing `capacity` reallocates replay and retains its valid rows
in chronological order. A capacity smaller than the retained transition count
is rejected. Checkpoints saved after a replay allocation mismatch are repaired
using the rows actually present. Shape changes can fail during loading;
observation-meaning changes can load but make old experience unsuitable.
Replay sampling uses the resumed config's `batch` and `sequence` sizes.
Matching reward settings do not guarantee the same objective after timestep,
discount or speed-target changes.
Saved optimizer settings, including learning rate, are restored. Use fresh runs
for clean comparisons and keep evaluation data separate from training replay.

Rewards add 50 for each newly crossed route checkpoint, spaced 15 m apart.
Lane-centering defaults to +1 per metre of new forward progress within 0.5 m
of the intended lane center, tapering to zero at the lane edge. Lateral error
penalties also start beyond 0.5 m; heading and lane-departure penalties still
apply. Departure detection uses the planned route's lane width, with an extra
0.5 m tolerance on segments entering or leaving a junction. Physical off-road
checks remain separate. Unsafe or repeated progress earns neither reward. Collision and off-road each add a -100 penalty
component; other negative components may also apply. Off-road means the
vehicle center is outside a mapped driving lane; adjacent driving lanes are
not off-road. Explicit reward values in a config still override these
defaults; historical run configs and artifacts are unchanged. Validation here
is offline; it does not establish CARLA driving performance.

If the fixed controller succeeds but the actor stalls with few safety
overrides, an optional moving-data bootstrap uses `warmup_policy: "controller"`
and a larger `warmup_frames` (experiment 17 uses 2048). These initial actions
come from the controller through the normal safety path and populate the same
replay buffer. `world_model_warmup_updates: 200` trains only the model on the
first 200 learner updates, then enables joint actor/critic training. This is
not behavior cloning. Defaults retain random warmup and immediate joint
updates. Logs identify `action_source` and `learner_mode`; resume preserves
the schedule through frame/update counts. Fixed comparisons allow these
training schedule differences while requiring matching task/reward settings.
Budget expiry during training saves a checkpoint and reports
`stop_reason: "budget_exhausted"` in the result.

An exception during the training loop also saves `latest.pt`, including model,
optimizer, replay, RNG state and progress counters. Live failures save the full
traceback in `error.txt`. Resume with `--checkpoint` pointing to the most recent
run's `latest.pt` and its matching config. The command prints the restored counts
and destination checkpoint path. Without `--output`, every invocation creates
a new `runs/dreamerv3_train_TIMESTAMP_PID` folder; it does not update the input
checkpoint. Use a new explicit `--output` folder to make continuations easy to
find. An interrupted simulator episode restarts on resume.

`actor_bc_weight` optionally anchors the actor to controller actions retained
in replay (`imitation_config.json` uses 1.0). It defaults to zero and requires
controller warmup. Controller transitions are explicitly marked as expert;
actor transitions are never imitated. The supervised term uses current-model
detached posterior features at t+1 and the expert action at t+1, skipping
episode boundaries. It updates only the actor alongside the normal Dreamer
objectives; world-model-only updates still leave the actor untouched. Logs
include `actor_bc_loss`. This objective can change on resume with matching rewards.

`expert_episode_interval` optionally refreshes controller-labelled episodes
on the same route/weather (for example 5 schedules one in five episodes).
Default 0 disables refresh. Logs distinguish `controller_refresh` from actor
actions; actor-only evaluation never uses this controller. The interval can
be changed explicitly on resume without changing the reward objective.

`actor_steering_bc_weight` additionally supervises steering on all valid replay
states using the same pure-pursuit reference computed from their navigation
points. This supplies corrective targets at drifted actor states; it does not
supervise throttle, change dynamics/rewards or bypass runtime safety. Default
0 disables it. Changes on resume automatically record `supervision_fork` in
new checkpoints.

`actor_longitudinal_bc_weight` adds throttle/brake supervision on valid next
policy states, including states visited by the actor. It defaults to 0;
experiment 25 enables it at 4. Targets combine proportional speed feedback
with a copy of the current safety supervisor, preserving its signal/crossing
state without advancing the live supervisor. The targets are replay labels;
the actor's actual applied actions still drive the simulator and world model.
The loss uses detached posterior features and supervises only the longitudinal
distribution. Terminal and invalid states are excluded. Metrics include
`actor_longitudinal_bc_loss`.

Model, optimizer, RNG, replay and progress are retained when either corrective
loss changes on resume. Old replay rows without longitudinal labels are marked
invalid for this term; newly collected rows provide the targets. Experiment 25 uses
`expert_episode_interval: 5`, which rotates controller refresh through the
original 12-scenario cycle and the reduced 11-scenario cycle.

The separate `runs/experiment_25/motion_pilot_config.json` is a fresh 20,000-step
motion pilot with the same reward weights, clear weather, a 2 m/s speed cap and
no automatic background NPCs. Its scenario catalog is beside the config. It
pins the first supported approximately 40 m straight, nonjunction route from
a seeded search of at most 128 endpoint pairs. Route planning fails explicitly
if none is supported. Map-reported speed limits remain visible; the optional
`driving.max_speed_mps` caps target speed (0 leaves existing behavior unchanged).
Authored scenario actors remain active when `background_traffic` is false.
Signals, collisions and all safety checks remain active.

Experiment 26's `motion_pilot_config.json` removes the 2 m/s cap. To resume
its existing pilot checkpoint at the supervisor's road-limit target, use:

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/motion_pilot_config.json \
  --checkpoint runs/experiment_26/output/latest.pt \
  --output runs/experiment_26/speed_limit_resume --budget-seconds 1800
```

Future collection uses the new cap; old transitions remain in replay.
Checkpoints and results automatically record `speed_cap_fork`.

From the repository root, run the pilot only when ready:

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_25/motion_pilot_config.json \
  --output runs/experiment_25/motion_pilot --budget-seconds 1800
```

The pilot changes the route/speed task and starts fresh. To continue the newer
156-episode checkpoint with longitudinal supervision and a reduced scenario list:

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_25/config.json \
  --checkpoint runs/experiment_25/resume_134/latest.pt \
  --output runs/experiment_25/curriculum_resume --budget-seconds 21600
```

Changes to `scenario` or `scenarios` automatically record the source checkpoint
as `scenario_fork`. The new list controls future collection; older transitions,
including removed scenarios, remain in replay.

Experiment 26 has continuation stages in `runs/experiment_26/curriculum/`:

| Config | Episode cycle | Total frame target |
| --- | --- | --- |
| `stage1_straights.json` | Original fixed 40 m pilot, varied 80 m straights | 600,000 |
| `stage2_turns.json` | Both earlier tasks, then varied 80 m turn routes | 800,000 |
| `stage3_lights.json` | Earlier tasks plus straight routes through traffic lights | 1,000,000 |
| `stage4_pedestrians.json` | Earlier tasks plus pedestrian-only and pedestrian/lead-vehicle cases | 1,200,000 |
| `stage5_wet_weather.json` | Earlier tasks plus wet straights, rain turns and heavy-rain lights | 1,400,000 |
| `stage6_low_visibility.json` | Earlier tasks plus fog, dusk, night, foggy lights and rainy-night hazards | 2,000,000 |
| `stage7_town01_variety.json` | Stages 1–6 plus 36 variants of routes, weather and actor placements | 4,000,000 |

All keep the pilot's rewards, model, safety settings and supervision weights.
Stages 1–2 use ClearNoon with no background or authored traffic. The pilot stays
fixed; new tasks cycle up to 16 supported routes with distinct starting spawn
indices. Route searches stop after 128 candidates, or 2,048 for traffic-light
tasks whose constrained routes are rare among arbitrary endpoints. Cropped
routes exclude junctions. Town01's turns are tight corners rather than gentle bends; stage 2
limits total heading spread to 1.8 radians and rejects straight or larger-turn
routes. This bound does not guarantee a minimum bend radius.

Run stage 1 after the actor can complete the pilot:

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage1_straights.json \
  --checkpoint runs/experiment_26/fixed_controller_speed/latest.pt \
  --output runs/experiment_26/curriculum/stage1_run --budget-seconds 1800
```

The frame targets are cumulative, not additional frames. A wall-clock budget
can end a run earlier; resume its latest checkpoint into a new output directory.
Use a newer pilot checkpoint if one exists. Stage 2 resumes stage 1 with
`stage2_turns.json` and a new output such as `curriculum/stage2_run`.
Controller refresh interval 5 cycles across both the two- and three-task lists.
Promotion is manual; adding a task does not prove that the actor improved.

Continue with lights, then pedestrians, then wet weather only after the preceding
stage passes actor evaluation. The frame targets are ceilings, not required
training durations; save evaluated checkpoints and advance when progress levels
off. Stage 3 has four tasks and refresh interval 5; stages 4–5 have six/nine
tasks and interval 7, preventing demonstrations from being pinned to one task.

Stages 6–7 also stay in Town01 and retain stage 5's model, rewards, driving
settings and supervision weights. Their separate `scenarios_stage6.json` and
`scenarios_stage7.json` catalogs preserve earlier scenario definitions. Stage 6
cycles through 15 tasks: the previous nine plus foggy straights, dusk turns,
night straights, foggy traffic lights, night pedestrians and rainy-night mixed
hazards. Stage 7 retains those 15 and adds six variants for each skill group:
straight driving, turns, lights, pedestrian/vehicle encounters, wet weather
and low visibility. Its 51 tasks use independent route seeds; the new cropped
routes range from 60 to 120 m, while light cases use full routes. Mixed-hazard
variants change lead-vehicle distance/speed, pedestrian side/speed and include
two child cases. The unchanged refresh interval 7 reaches every task in both
lists. Task order repeats; weather and actor offsets are explicit variants,
not continuously randomized ranges.

Fog uses scenario `fog_density` (0–100) and `fog_distance` (nonnegative metres);
`sun_altitude_angle` accepts -90–90 degrees. Overrides copy the weather preset
without changing it. These configs retain ground-truth perception: low visibility
changes rendered RGB, weather context and speed targets, while object tokens
remain simulator observations. They do not train a camera-only detector.
Route availability and hazard spawning still need live Town01 checks; a valid
config does not establish driving performance. No training starts by creating
these files. Example continuation commands, each with a 3,600-second budget:

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage6_low_visibility.json \
  --checkpoint runs/experiment_26/curriculum/stage5_run_3/latest.pt \
  --output runs/experiment_26/curriculum/stage6_run --budget-seconds 3600

venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage7_town01_variety.json \
  --checkpoint runs/experiment_26/curriculum/stage6_run/latest.pt \
  --output runs/experiment_26/curriculum/stage7_run --budget-seconds 3600
```

The step targets above are cumulative across checkpoint resumes. Evaluate each
stage before promotion; use a new output directory for each continuation.

```bash
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage3_lights.json \
  --checkpoint runs/experiment_26/curriculum/stage2_run_3/latest.pt \
  --output runs/experiment_26/curriculum/stage3_run --budget-seconds 1800
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage4_pedestrians.json \
  --checkpoint runs/experiment_26/curriculum/stage3_run/latest.pt \
  --output runs/experiment_26/curriculum/stage4_run --budget-seconds 1800
venv/bin/python -m dreamerv3_attention train \
  --config runs/experiment_26/curriculum/stage5_wet_weather.json \
  --checkpoint runs/experiment_26/curriculum/stage4_run/latest.pt \
  --output runs/experiment_26/curriculum/stage5_run --budget-seconds 1800
```

Run each command separately after reviewing the prior stage, using its best
evaluated checkpoint if it differs from the example. No background vehicles or
unrequested pedestrians are added. Signal tasks require a lane/direction-matched
stop line at least 30 m into the route and at least 10 m before its end. They
use full routes through junctions, preserving natural red/yellow/green timing.
`red_light_observation_steps` and `yellow_light_observation_steps` report actual
exposure; an all-green success does not establish stopping and restarting.

The pedestrian case requires one adult walker on nearby navigation within the
initial awareness corridor. Required walkers search up to 16,384 map-wide
navigation samples for nearby visible candidates and try alternate candidates
when a spawn is blocked. Missing navigation, spawn, destination or moving
controller fails explicitly instead of silently running an empty case. Walkers
use native navigation and avoid jaywalking. This is sidewalk awareness and
appropriate speed response; Town01 has no crosswalk annotations in the checked
OpenDRIVE file, so the stage does not claim controlled crossing coverage.
`pedestrian_encounter_steps` counts nonzero observed pedestrian risk.

Stage 4 also includes `train_pedestrians_vehicles`: one lead vehicle starts
22 m ahead at a requested 3 m/s, together with the nearby moving pedestrian.
The vehicle uses the existing rule-following Traffic Manager. Both actors are
required; unavailable spawns fail explicitly. This mixed case is retained in
stage 5 alongside the pedestrian-only case. Automatic background traffic stays
off, so no unrelated actors are added. Use `evaluate_pedestrians_vehicles.json`
to test the same mixed case after training on it.

For a runtime-only adjustment, `driving.pedestrian_speed_factor` controls the
remaining speed fraction near adults or pedestrians without trusted age labels;
its default is 0.8. `driving.child_pedestrian_speed_factor` defaults to 0.15,
retaining the strong slowdown near children with authored labels. On a clear
30 km/h road these targets are 24 km/h and 4.5 km/h respectively. Partial risk
interpolates each fraction; overlapping risks use the strongest reduction once.
Road, weather and following limits can reduce these targets further. The route
controller and supervisor share these settings. Rain, wetness, darkness,
visibility and actor awareness each cap the same base speed (trusted limit or
default, bounded by `driving.max_speed_mps`). The lowest cap wins; reductions
are never added or multiplied together. Following and signal/crossing limits
apply independently to that target. `driving.vehicle_speed_factor`
defaults to 0.8 and applies when a vehicle is detected ahead within 80 m and
2.5 m of the ego centreline. Gap-based following limits still apply; vehicle
and pedestrian factors use the strongest reduction once. Side/rear vehicles
and unclassified radar points do not trigger this vehicle factor. Pedestrian-conflict,
occupied-crossing, close-front-hazard and signal stops remain enforced.
Policy observations and weights are unchanged, and actor-requested braking
is still respected.

Existing configs with an explicit `pedestrian_speed_factor` retain that override.
`stage5_pedestrian_runtime.json` copies `stage5_run_2/config.json` with only the
two pedestrian factors set to 0.5 and 0.15. Use it on the next resume or inference
to adopt the age-specific targets with the existing checkpoint. Running Python
processes must reload the code and configuration; historical run files are kept.

`pedestrian_runtime_eval.json` copies the mixed-case evaluation with only this
factor changed to 0.5. Evaluate an existing checkpoint without retraining:

```bash
venv/bin/python -m dreamerv3_attention infer \
  --config runs/experiment_26/curriculum/pedestrian_runtime_eval.json \
  --checkpoint runs/experiment_26/curriculum/stage4_run/latest.pt \
  --output runs/experiment_26/curriculum/pedestrian_runtime_infer --budget-seconds 120
```

This changes the runtime speed target; it cannot guarantee that a learned
actor stops requesting brake. Use requested/applied actions and `action_source`
to distinguish actor behavior, controller-refresh episodes and safety overrides.

Wet-weather cases add WetNoon straights with wetness 60%, SoftRainNoon turns
with wetness 70%, and HardRainNoon traffic lights with wetness 90%. Presets are
copied before applying wetness: the installed presets otherwise report zero
wetness. Existing rain/wetness speed reductions remain active. CARLA weather
does not automatically reduce tire friction; this stage covers weather
perception and conservative speed selection, without a custom low-grip model.

New single-task evaluation configs are `evaluate_lights.json`,
`evaluate_pedestrians.json`, `evaluate_wet_straight.json`,
`evaluate_rain_turns.json` and `evaluate_heavy_rain_lights.json`. Use the same
actor/controller command pattern below with a matching
checkpoint, and fresh output paths. Retest all earlier skills after each stage.
Inspect exposure counters, violations, stop release and supervisor overrides;
the supervisor can mask an actor that has not learned to stop independently.

All curriculum evaluation configs select exactly their named training task.
`evaluate` and `infer` honor this selection without needing `--fixed-scenario`;
they do not expand it to the shared catalog's later tasks. Named training cases
are automatically labeled as training-sanity checks. Default validation/held-out
configs without named selectors still evaluate their full split suite.
The light/turn/straight/weather cases have no background or authored NPCs.
`evaluate_pedestrians.json` adds one walker; `evaluate_pedestrians_vehicles.json`
adds one lead vehicle and one walker. Other evaluation cases remain traffic-free.

The three `evaluate_{pilot,straight,turns}.json` files select one task each.
Check the controller first, then the actor with the same config and safety:

```bash
venv/bin/python -m dreamerv3_attention evaluate \
  --config runs/experiment_26/curriculum/evaluate_straight.json \
  --controller \
  --output runs/experiment_26/curriculum/straight_controller --budget-seconds 600
venv/bin/python -m dreamerv3_attention evaluate \
  --config runs/experiment_26/curriculum/evaluate_straight.json \
  --checkpoint runs/experiment_26/curriculum/stage1_run/latest.pt \
  --output runs/experiment_26/curriculum/straight_actor --budget-seconds 600
```

Repeat the pilot check after stage 1; check all three tasks after stage 2.
These are single-episode training-scenario sanity checks, not a held-out suite
or a 20–30-trial promotion assessment. Evaluation can select different routes
from collection because of its scenario seed; results record actual selected
endpoints. For promotion, collect additional route/seed trials separately and
inspect completion, collisions, signal violations, inactivity and supervisor
interventions. A provisional gate is at least 90% completion across 20–30 actor
trials without collisions or signal violations, plus retained earlier skills.
Controller-refresh training episodes do not count as actor successes.

Offline Town01 OpenDRIVE checks found supported straight and turn paths using
generated waypoint transforms as proxy spawns. Actual simulator spawn coverage
and driving performance still require live validation; unsupported route
searches fail explicitly. These commands were not run during implementation.

Output paths must be new or empty. These commands were not run during
implementation. Pilot motion and learning still require live validation.

`train --world-model-only` uses TorchRL's model loss and the checkpointed
optimizer, while leaving actor, critic, return-normalization and target state
unchanged. It continues collection during fitting and records `training_mode`
in checkpoints, metrics and results. Plain `train` retains joint Dreamer updates.

Separately collect with `split: validation` or `held_out` for world-model
validation, then run `validate --checkpoint MODEL/latest.pt --data-checkpoint
VALIDATION/latest.pt`. It loads on CPU and computes one held-out model-loss
batch without optimization. Training commands require `split: train`, but
resume no longer rejects a source checkpoint from another split. Keep validation
and held-out replay out of training. Only load trusted project checkpoints:
their replay/RNG state uses PyTorch's full checkpoint format.

For an explicitly approved bounded live integration check against an idle
dedicated server: `DREAMERV3_APPROVED_LIVE_TEST=1 venv/bin/python -m pytest
dreamerv3_attention/tests/test_live.py -q`. It is skipped by default.

An episode terminates when its accumulated reward drops below -500. The
crossing step keeps its computed reward and reports `reward_limit` as the
termination reason; the return resets at the next episode.

The inactivity watchdog truncates an episode after 15 simulated seconds with
less than 0.5 m net route progress while movement is safe and sensing is valid.
Blocked movement and invalid sensing use an independent 60-second
stagnation window. A red/yellow signal wait requires speed below `stop_speed_mps`
(default 0.2 m/s) and front-bumper distance within 2.5 m of the stop line.
The approach buffer remains `crossing_buffer_m` (default 2 m).
Connected stationary queues and stops at a nearby crosswalk
also qualify. These waits use a zero speed target and brake hold, and pause both
watchdogs, the episode time limit, and time, idle and speed-tracking penalties.
Lane and motion-change costs remain active. A 2-second startup grace applies
before either window begins. The watchdog adds no failure reward; genuine environment termination
takes precedence, and `max_episode_steps` can end an episode sooner. Tune
`inactivity_timeout_s`, `inactivity_min_progress_m`, `blocked_timeout_s`, and
`inactivity_grace_s` in the config.

## Curriculum, scenarios and metrics

`evaluation.scenario_catalog()` supplies split-specific seeds/routes for every
requested category. Built-in route indices are candidate definitions, not
claims of valid placement or map capabilities. Use `scenario_file` with
`{"scenarios": [...]}` to author map-verified routes and actors. Each record
requires name, split, seed, distinct route_start/route_end spawn indices and
category. Curves, intersections, crosswalks, speed-limit landmarks and child
blueprints are checked at runtime. Every scenario requests a vehicle, an adult,
and a child pedestrian; unavailable blueprints raise an error. Rejected actor
placements are skipped with a warning so the episode can continue. A skipped
authored actor means its intended hazard is absent from that episode.
Each episode also requests 1–2 extra vehicles and 5–10 extra pedestrians,
using seeded placements and bounded retries. Fewer actors may spawn when
valid space or sidewalk navigation is unavailable, with a warning reporting
the shortfall.

Vehicles align to nearby driving lanes and avoid spawning in junctions.
Moving vehicles use synchronous, seeded CARLA Traffic Manager autopilot:
lights, stop signs, vehicles and pedestrians are never deliberately ignored;
lane changes are disabled; the following gap is 5 m. Their requested speed is
capped at 80% of the current speed limit on every step. Authored stationary
vehicle obstacles remain braked.

Roaming pedestrians spawn on nearby sidewalk navigation points and use native
walker AI at walking speed, capped at 1.4 m/s. Road crossing probability is
zero: CARLA's navigation filter excludes ordinary road areas while permitting
marked crosswalk areas. Native navigation handles avoidance and checking for
traffic before crossing. Without a connected crosswalk they stay on their
sidewalk network; without nearby sidewalk navigation they are skipped. This
depends on correct map/navmesh tagging. Explicit zero-speed `at_crossing`
actors remain stationary occupied-crossing hazards.

Authored XY coordinates now select the vicinity of a valid lane or sidewalk;
authored yaw no longer forces NPC motion. This changes the traffic and hazard
placement distribution from historical runs. Traffic Manager has approximate
priority handling at unsignaled junctions and cannot guarantee accident-free
traffic. Restart the environment process to load these changes; old replay and
run artifacts retain their original behavior.

Child labels use authored age attributes or the documented CARLA child blueprints. External
child flags require `child_labels_authored: true`; other child flags are
discarded and the same conservative pedestrian protections still apply.

Training selects `scenario: basic` or a split-qualified catalog name, e.g.
`train_pedestrian_crossing`. A sensible staged sequence is basic lane motion,
route/speed control, nearby vehicles, pedestrians, mandatory crossing stops,
rain/night, combined hazards, then sensor uncertainty/external perception.
Use separate unique runs and explicit measured promotion criteria (completion,
no collisions/violations, intervention rate) before increasing difficulty.
There is no automatic curriculum promotion or unvalidated transfer between
reward/observation contracts. Scenario/seed changes on resume are allowed when
reward settings match; old replay is retained.

The supervisor handles mandatory stop/dwell/clear/resume before route-filtered
crossings, pedestrians, speed/weather envelopes, front braking and side
steering constraints. Rear hazards are observed; braking is never suppressed
because of a rear object. It is a simulation research safeguard, not a proven
safety guarantee. Reward preferences are separate, component-logged and
configurable; default weights are provisional, uncalibrated by driving trials.
Speed-envelope corrections use proportional feedback (0.5 times speed error,
clipped to the action range) rather than a fixed 40% brake for small overshoots.
Throttle is bounded by the speed error and the remaining next-tick speed margin.
This also permits gentle starts when weather or risk leaves a small positive
target. Stronger actor brakes and mandatory hazard/signal stops retain priority.

Front radar hazards follow the planned route corridor through bends; obstacles
inside the current body corridor and stopping distance remain relevant even
when the route turns away. Side clearances use the vehicle width. Known side
hazards limit steering toward the obstacle while allowing correction away;
very close or undirected hazards retain limits in both directions. This avoids
stopping for roadside barricades or blocking the required turn beside them.
`left_hazard_m` and `right_hazard_m` record the directional clearances.

The route expert uses trusted speed-limit observations, reduced for weather,
visibility and pedestrian risk, with the configured default speed when the
limit is unknown. `--controller-speed` optionally adds a positive cruise cap.
Steering follows the existing 5 m route point; the 15 m point previously
cut across tight corners. The actor's corrective steering loss uses the same
5 m reference. Existing learned weights and stored expert actions retain
their previous behavior until further training with refreshed demonstrations.
The safety supervisor remains responsible for hazards and stop/dwell/resume
behavior. Validation uses offline fake plants and unit tests; it does not
establish CARLA driving competence. Traffic-light stop waypoints are matched
to route lanes and direction before entering a light's trigger. Red/yellow
distances follow route arc length to the front bumper; phases are read each
tick. Yellow has a separate phase and confidence. When a controlled stop is
feasible, the controller and supervisor reduce speed and hold the brake near
the stop line. If yellow first appears too late for that stop, they commit to
continuing while yellow persists; red is still handled by the red stop rule.
The decision resets when yellow disappears. Red and yellow stopping use
`driving.red_light_deceleration_mps2` (default 1.5) and a tick allowance. This
conservative assumption has offline coverage and is not calibrated CARLA
braking performance. The supervisor rechecks the stopping envelope each tick,
releasing excess braking if the vehicle slows faster than expected. The speed
target remains positive during approach. The shared stationary stop area above
sets a zero target and full-brake hold; unknown signal distance and the final
buffer plus one low-speed tick also enforce a hold. Earlier braking does not
latch the vehicle far from the line.
The route controller also rechecks yellow braking each tick; its carry retains
the yellow stop/go decision without retaining a full-brake command.

### Temporary restart assistance

To reuse a checkpoint with a bounded red-to-green restart assist, copy the
current run configuration and merge `"restart_assist_enabled": true` into its
existing `driving` object. Keep the other driving and reward settings. The
default is false, including when loading older configurations. Restart the
environment process to load the code and configuration; existing processes
and historical artifacts are not changed automatically.

The assist supports `perception: "ground_truth"` only. It confirms a stop at
at most 0.2 m/s within 6 m of a known red stop line for 0.5 s, then waits at
least 0.5 s after the same light explicitly turns green. An actor already
making sustained progress retains control. Otherwise, proportional speed
feedback temporarily overrides the actor's throttle/brake command, with a
1.5 m/s target and at most 0.4 throttle. Steering stays with the actor.

Invalid observations/actions, unknown signals, red/yellow signals, hazards,
required crossing stops and poor lane alignment cancel assistance. Normal
braking and speed constraints retain final authority. Handover requires
speed at least 1 m/s, at least 0.5 m of forward progress and actor throttle
at least 0.1 for 0.5 s. After at most 5 s of assistance, control returns to
the actor even if recovery fails. Another attempt requires a fresh verified
red stop; remaining stopped on green cannot trigger unlimited retries.

The configurable `driving` limits are `restart_stop_distance_m` (6),
`restart_stop_confirm_s` (0.5), `restart_green_grace_s` (0.5),
`restart_target_speed_mps` (1.5), `restart_max_throttle` (0.4),
`restart_timeout_s` (5) and `restart_handover_s` (0.5). Durations use
simulation time. The stopped-speed threshold uses existing `stop_speed_mps`.

Signal metadata stays outside policy tokens, and mixed/external perception
cannot inject it. Rewards and corrective longitudinal targets keep their
existing speed reference. Applied actions remain the source for replay and
recurrent state. Metrics record `restart_assist_phase`, `restart_assist_event`
and `restart_assist_elapsed_s`, alongside requested/applied actions and
intervention reasons. Report assisted performance separately from the raw
actor. Offline tests do not establish live CARLA recovery or reliable driving
after handover.

Evaluation records route completion and progress, traveled distance, event
counts, separation, crossing/speed compliance, missing-input behavior,
intervention steps/reasons, tracking error, idle behavior, control smoothness
and policy latency. Training logs upstream losses, update duration, throughput
and peak CUDA allocation. Use held-out routes and multiple seeds for comparison.
GPU inference timing and attention overhead need synchronized profiling on the
host GPU; CPU smoke timings cannot establish GPU efficiency or driving gain.

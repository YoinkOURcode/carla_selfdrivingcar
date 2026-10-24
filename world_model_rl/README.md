# CARLA attention/RSSM + SAC

Point-to-point driving in Town01 using structured CARLA observations, masked attention,
a Gaussian RSSM with GRU memory, and SAC with real and validated imagined transitions.

The current contract is `carla-structured-v2` with `decision-comfort-v3` rewards.
`environment.max_steps` counts active driving decisions. Stopped waits in the
0–5 m area before a relevant red/yellow line pause the deadline; green lights and
early stops spend it. Expiry ends the task with `-reward.time_limit` (default 100).
Observation index 15 is remaining active seconds, and traffic light presence has
one mask. Geometric V3/V4 teachers decelerate toward a 2.5 m front-bumper margin;
the former V4 blanket 25 m braking rule is removed.

These changes require fresh compatible recordings, a new training-only normalizer,
attention/RSSM and prediction heads, then new SAC actor/critics. Old experiment
artifacts remain historical and cannot be loaded directly into this contract.
Generate a current default configuration and start a fresh run when ready:

```bash
venv/bin/python -c 'from world_model_rl.config import Config; from world_model_rl.data import write_json; write_json("runs/light_deadline_v2/config.json", Config().to_dict())'
venv/bin/python -m world_model_rl run --config runs/light_deadline_v2/config.json --output runs/light_deadline_v2/training
```

The stopping geometry and future CV provider interface are described in
[DESIGN.md](DESIGN.md). Historical profiles below describe earlier experiment
artifacts; their saved configurations and datasets require explicit conversion
before reuse. The code update does not launch training.
[


The server launcher requires `UE4_ROOT` to point to the built Unreal Engine.
Client and server must use the same CARLA build. Do not share a synchronous server
with another ticking client.

From the repository root, start the complete training pipeline:

```bash
bash world_model_rl/train_from_scratch.sh
```

Each invocation chooses a new `runs/from_scratch_<timestamp>_<suffix>` directory.
To specify the output, SAC update budget, and optionally a configuration:

```bash
bash world_model_rl/train_from_scratch.sh runs/experiment4 100000
bash world_model_rl/train_from_scratch.sh runs/experiment5 50000 my_config.json
```

The script rejects an existing output directory or log. It does not launch training
until you run it. It streams progress to your terminal and `OUTPUT_DIR.log`, preserving
the training command's failure exit status. Configuration paths and relative output
paths are resolved from the repository root.

At the default configuration, a fresh run:

1. Collects 20 new episodes with alternating BasicAgent/noisy-BasicAgent controllers,
   new episode seeds/routes, and the configured day/night/rain/fog conditions.
   Checks training and fixed validation-split event coverage before any gradient updates.
   Missing task endings or nonterminal support stops the run and saves a coverage report.
2. Fits normalization using only the new training split. Randomly initializes attention,
   RSSM, all prediction heads, and the model optimizer; performs 10,000 joint updates
   followed by 2,000 termination-readout updates on full real histories.
3. Saves `world_model.pt` and evaluates a held-out prediction/termination suite.
4. If validation passes, initializes fresh SAC actor/critics/optimizers and performs
   100,000 policy updates in 10,000-update blocks, using 75% real and 25% imagined replay.
5. Between policy blocks, collects two episodes using the current SAC actor,
   rebuilds latent replay with the frozen world model, and validates before further SAC updates.

`run` requires a new or empty output directory and at least ten bootstrap episodes.
It never reuses data or weights from an existing experiment. The configuration is saved
in the output before collection. Existing experiments are retained as separate artifacts.
The launcher always starts a fresh run; it has no checkpoint input.

Collection and environment evaluation require CARLA; fitting and SAC optimizer blocks
use recorded replay without connecting to CARLA. `training.device=auto` selects CUDA
when available, otherwise CPU. Use `cuda` in a copied config to require the GPU.

## Watch driving episodes

While collection/training is running, open another terminal in the repository root:

```bash
venv/bin/python -m world_model_rl.viewer --config runs/experiment4/config.json
```

Open http://127.0.0.1:8080 in your browser. Use the config path for your current run.
The viewer attaches its own chase camera to the training vehicle and follows new
vehicles across bootstrap and SAC collection episodes. It also works with the offscreen
server. It never advances simulation ticks, changes world settings or controls the car.
It waits between driving sessions; fitting/SAC replay updates have no car to follow.
Extra camera rendering uses some GPU time. Ctrl+C stops the viewer and removes its camera.
No additional packages are required. Use `--port 8081` if the default browser port is busy.
This is a live view; existing NPZ recordings do not contain camera videos for past episodes.

The dashboard shows current speed in km/h and m/s, recognized object slots (class,
actor ID, ego-relative position and provider confidence), and the accumulated reward
for the current episode. It also compares actual speed against the observation's cautious
target speed, shows route completion and distance remaining along the route, and displays
the applied steering, throttle and brake values after action limits. The reward breakdown
lists each component for the last decision and its accumulated episode total, including
negative costs. Rewards and their component totals reset at each episode. With the default
provider, objects come from CARLA ground truth with geometric visibility filtering;
they are not camera-based detector predictions.

During SAC/DAgger driving, the attention panel shows the actual encoder weights over
ego state, route, weather, traffic controls, five waypoints and present object slots.
Choose an individual head or the mean of all heads. Empty slots are masked. These
weights describe the structured input attention, not a pixel heatmap or proof of why
the policy chose an action. Bootstrap/expert-only episodes do not run the encoder,
so their attention is unavailable.

Restart the viewer and the driving process after updating to this version. Telemetry
from a process started before the expanded panels was added lacks their fields; the
viewer explains when saving progress and restarting collection/training is required.
Restarting only the viewer or refreshing the browser cannot add those missing values.
Telemetry
is shared through an atomic local temporary JSON snapshot keyed by CARLA host/port;
both processes must run as the same user on the same machine and use the same CARLA
endpoint. No new CLI options or dependencies are needed. State updates happen once
per driving decision; the chase camera is a separate sensor and may show a different
simulation frame. State older than five wall-clock seconds is marked stale. Offline
fitting and replay have no live driving state to display.

## Validation and budgets

A failed validation gate stops SAC with exit code 2 and writes `validation.json`.
More independent ending/nonterminal data or changes to observable hazard coverage may
be necessary. Twenty ordinary collection episodes do not guarantee a passing gate.
Review event support, precision/recall, per-ending diagnostics, and prediction errors;
the program does not lower thresholds or substitute real-only policy training.
Optional policy experiments can set `training.behavior_cloning_updates` (default 0),
`behavior_cloning_weight` (default 0), and `policy_observation_skip` (default false).
With the `recorded` teacher, warm-up initializes the fresh actor from training-split
`autopilot` requests or separately recorded DAgger expert labels;
it never loads an old SAC policy. Warm-up counts are separate from SAC updates.
The demonstration penalty uses deterministic actor requests and Q normalization;
demonstration batches balance turns/braking with cruising. The observed-state skip
adds normalized structured observations alongside RSSM features. Real replay and live
driving use current observations; imagined skips use the model's predicted state,
masks, classes and light, with known controls/weather supplied causally. These are
explicit algorithm variants, not automatic recovery or an expert override at runtime.

`training.behavior_cloning_teacher` defaults to `recorded`, which uses actual expert
requests or the separate DAgger labels. The optional `geometry_v1` experiment instead
computes counterfactual pursuit/speed requests from current structured training states:
curvature-limited cornering, observed light stopping, and observed-object clearance.
It may use any training collector's states, including recovery states; validation/test
states remain forbidden. These are teacher-generated labels, not measured simulator
actions. Dynamics/critic replay still uses the measured executed controls and rewards.
The teacher is not called during policy evaluation. This variant is calibrated for the
Tesla Model3 configuration and must establish its driving quality in held-out rollouts.
`geometry_v2` keeps those longitudinal labels and calibrates pursuit steering using
the measured Tesla wheelbase (approximately 3.0 m), 70-degree front-wheel steering,
and CARLA's speed-dependent steering curve in km/h. Each teacher version stays explicit
in configuration/checkpoints so an earlier experiment's label rule is reproducible.
`geometry_v3` additionally holds at least 0.35 brake with zero throttle when the
teacher's desired speed is at most 0.3 m/s. This prevents the speed servo's cruise
feed-forward from teaching creep through a stopped lead vehicle or red light.
It does not guarantee the learned actor stops: candidate 007 still recorded ten
red-light violations over its ten development episodes.

`behavior_cloning_traffic_fraction` defaults to 0. At 0.5, half each demonstration
batch focuses on near red/yellow braking states, split three quarters moving and one
quarter waiting; the remainder retains manoeuvre/uniform sampling. Candidate 008
uses this option. `behavior_cloning_permute_objects` defaults false and requires
`policy_observation_skip`. When enabled, demonstration geometry/masks/classes are
permuted together within vehicle/pedestrian slot groups, preserving teacher labels.
It affects actor demonstration fitting only; no trained checkpoint uses it yet.

For additional recovery-state demonstrations, `collect --policy-checkpoint agent.pt
--expert-probability 0.5` executes a seeded mixture of the deterministic policy and
BasicAgent while recording separate `expert_actions` at every visited state. These
episodes are labelled `dagger`. Dynamics/Q replay always use the executed requests and
applied actions; demonstration fitting uses the separate expert labels. Only training
splits supply demonstrations. Use an isolated dataset copy when preserving a resumable
agent's recording frontier; never collect development or final evaluation seeds for fitting.

For the structured ground-truth provider, `environment.no_rendering_mode: true`
disables viewport drawing to improve simulation throughput. Physics, timing and reward
settings stay fixed. Existing structured recordings/models remain compatible across
this rendering switch. Camera-based providers require rendering, and the live camera
viewer needs it enabled. The previous world setting is restored when the environment closes.

`evaluate --plan PATH` accepts a fixed JSON scenario plan with format 1, town, and a
`scenarios` list containing unique `episode_id`, `seed`, and scheduled `weather` values.
The plan overrides `--episodes` and scenario seeds from `--seed-offset`. Evaluation
never saves its trajectories into the training dataset. Results are saved after each
episode, with completion status and per-weather metrics. Existing result files cannot
be overwritten. Prototype acceptance requires all 50 planned episodes, ten in each of
five configured weathers, at least 80% successes, and at most 10% collision episodes.
A partial run or a smaller development suite cannot establish acceptance.
Reports also include red-light violation totals, violating-episode counts and rates
overall and per weather. A route arrival can still contain traffic violations.
`evaluate --record-traces` additionally saves diagnostic NPZ/JSON traces alongside
the result in a separate directory. These carry `split: evaluation` and explicit
never-train provenance; dataset save/load rejects them. They may be inspected for
failure analysis but must not be used for model or policy gradient updates.
`termination.by_episode` breaks event metrics down by episode. `termination.false_positive_groups`
describes the first up to 20 contiguous false-alarm stretches in time order, including
weather, transition range, duration, probabilities, observed speed/route distance and
visible-object counts. The total group count and all false positives remain in the metrics.
These are diagnostic contexts, not demonstrated causes or permission to fit on validation.

Before fitting, `validate --for-training` checks recorded task-ending/nonterminal
support in training and the first up to `training.validation_episodes` validation
episodes. `fit` performs the same check before allocating a model and writes
`CHECKPOINT.coverage.json`. Stuck truncations count as nonterminal transitions;
task terminations, including active timeouts, success and collision, are positives.
The check preserves all split assignments and does not select episodes by outcome.
Adequate label support alone does not establish prediction quality; the later model
validation gate remains required.

For an experiment blocked before SAC because held-out task endings are absent:

```bash
venv/bin/python -m world_model_rl validate --config runs/experiment4/config.json --data runs/experiment4/data --for-training
bash world_model_rl/recover_from_scratch.sh runs/experiment4 20 100000
```

With a dedicated CARLA server running, the recovery script appends exactly 20 new
episodes to that dataset, checks coverage, fits randomly initialized world-model
weights in a new `runs/experiment4/recovery_<timestamp>_<suffix>` directory, and starts
a fresh SAC policy only if validation passes. Existing checkpoints and recorded
episodes remain intact. No old weights are loaded. Progress is saved to the new
output's sibling `.log`. It stops before fitting if the fixed additional batch still
lacks coverage; it never loops until favorable validation outcomes appear. A run that
already has `agent.pt` is refused to protect its dataset/checkpoint frontier.
If the validation suite has already reached its configured episode cap, appending more
episodes does not replace its existing members; plan a broader suite explicitly before
fitting a new experiment rather than selecting favorable endings or using the test split.

The recovery output uses the original run's external dataset. To recover an interrupted
SAC session there, use `train --data runs/experiment4/data --resume OUTPUT_DIR/agent.pt
--output OUTPUT_DIR --config OUTPUT_DIR/config.json` with your update budget.

`--policy-updates` counts SAC optimizer updates. `--episodes` instead counts real SAC
collection episodes, in addition to bootstrap episodes; these flags are mutually exclusive.
Imagination spans one decision (0.1 s) by default and is regenerated every 100 SAC updates.
The normalizer stays fixed during a run; replay is rebuilt when the model changes.

## Individual stages

Use a new directory and collect fresh data:

```bash
venv/bin/python -m world_model_rl collect --data runs/manual/data --episodes 20
venv/bin/python -m world_model_rl validate --data runs/manual/data
venv/bin/python -m world_model_rl fit --data runs/manual/data --checkpoint runs/manual/world_model.pt --updates 10000
venv/bin/python -m world_model_rl train --data runs/manual/data --model runs/manual/world_model.pt --output runs/manual --policy-updates 100000
```

`fit` always creates a new model, normalizer and optimizer, and refuses to overwrite its
checkpoint. `train --model` accepts a fitted world-model checkpoint and starts a new SAC
policy. `fit-termination` remains available to refit only a current world model's ending
readout; it does not import an actor or critics.

Explicit recovery of the *same interrupted run* is separate from starting a new run:

```bash
venv/bin/python -m world_model_rl run --config runs/manual/config.json --output runs/manual --resume --policy-updates 10000
```

This requires `agent.pt`, restores all networks/optimizers/RNG state and phase counters,
and adds 10,000 updates. Dataset/checkpoint frontier mismatches stop for inspection.
An unfinished simulator episode restarts from its seed. The from-scratch launcher
never requests resume. Checkpoints are saved every 1,000 SAC updates and at phase boundaries.

## Evaluation and artifacts

```bash
venv/bin/python -m world_model_rl evaluate-model --config runs/manual/config.json --data runs/manual/data --checkpoint runs/manual/agent.pt --split test --output runs/manual/test_model.json
venv/bin/python -m world_model_rl evaluate --config runs/manual/config.json --checkpoint runs/manual/agent.pt --episodes 10 --output runs/manual/evaluation.json
```

`data/` stores numeric NPZ episodes, provenance and fixed episode splits. `world_model.pt`
stores the initial fitted model, optimizer and normalizer; `agent.pt` also contains SAC
state and the same frozen model. `training.jsonl` records policy progress and replay
rebuilds. Explicit fitting uses `world_model_training.jsonl` and `termination_training.jsonl`.
The launcher's sibling `.log` contains collection summaries and printed stage progress.
No automatic completion notification is configured.

The observation provider defaults to structured CARLA ground truth. A perception
provider can be selected through `environment.observation_provider`; a changed observation
contract requires a new compatible dataset and model. See DESIGN.md.

## Offline verification

```bash
venv/bin/python -m pytest world_model_rl/tests -q
```

Tests use generated arrays, small networks and fake environments. They do not start
CARLA or run full training. See [VALIDATION.md](VALIDATION.md) for evidence and limitations.

# World model RL command reference

Evaluation summaries now keep `event_totals` (crossing multiplicities) separately
from `event_decision_counts` (decisions containing an event). Red-light totals use
the former; violating-episode rates remain separate. Legacy summaries lacking
totals retain their historical decision-count interpretation. This reporting-only
correction preserves the fitting/SAC function ASTs, thresholds, data and settings;
its phase-specific source snapshot is recorded in the active experiment provenance.

For the exact Town01 artifact paths, warm-start explanation and reproduction/transfer
workflow, see [PROTOTYPE.md](PROTOTYPE.md). No final 50-episode success is claimed.

This reference covers all ten `python -m world_model_rl` subcommands, the live
viewer, and the shell launchers. It describes the interface implemented in
[`__main__.py`](__main__.py), [`runner.py`](runner.py), and
[`viewer.py`](viewer.py). For an introduction, see [README.md](README.md); for
the observation and model contracts, see [DESIGN.md](DESIGN.md).

For the consolidated collected-data workflow, see
[COLLECTED_TRAINING.md](COLLECTED_TRAINING.md). It documents the active dataset/model,
the `train_collected.sh` launcher, imports, and model adaptation after new collection.

## Fresh Town01 experiment profile, 17 October 2026

`runs/town01_experiment_001/experiment.json` now declares a fresh world model and
fresh SAC policy, using the preserved data plus one fixed 20-episode collection
batch. Copied prototype metadata is preserved in `historical_import/`; it is not
evidence of training for this request. `selection_plan.json` freezes candidate
times and development ranking before training. The final/development plans are
unchanged and their seeds remain reserved. The production five-hour clock begins
with fitting, after collection, and is shared with warm-up and SAC.

New **optional training** settings (package defaults in parentheses) are:

| Setting | This experiment | Function |
| --- | --- | --- |
| `red_light_objective_multiplier` (1) | 10 | Learning penalty 200 instead of recorded 20 per red crossing. |
| `collision_objective_multiplier` (1) | 5 | Learning penalty 500 instead of recorded 100 per collision. |
| `gamma` (0.99) | 0.997 | About 33 simulated seconds of discounted horizon at 10 Hz. |
| `behavior_cloning_teacher` (`recorded`) | `geometry_v4` | Historical profile used blanket 25 m braking; current V4 uses V3's distance-based approach and hold. |
| `behavior_cloning_applied_weight` (0) | 2 | Add shared-adapter applied-action MSE to request MSE in warm-up and SAC anchoring. |
| `behavior_cloning_permute_objects` (false) | true | Train with scene-preserving within-class current-slot permutations. |
| `deterministic_algorithms` (false) | true | Strict PyTorch deterministic checks; CUDA BLAS workspace and cuDNN settings are fixed. |

Raw NPZ rewards, reward components, requested/applied actions and expert labels
remain immutable. `learning_rewards` creates an in-memory, versioned event-cost
view for sequence fitting, real SAC replay and held-out reward prediction errors.
The learned reward head then supplies that same objective in imagination. The
checkpoint stores `learning_reward_contract`; loading a model with another
objective is refused. This is not an in-place data conversion, and no old model
is substituted. Driving summaries still report raw simulator return, while red
violations and arrivals without recorded violations are reported separately.

The v4 teacher is called only on training-split recordings. It never controls an
evaluation vehicle. Its conservative braking may increase waits/timeouts, which
must be measured. Slot permutation was previously untrained; enabling it is an
experiment, not established improvement. Neither a passing gate nor these code
changes establish the 40/50 arrival and <=5 collision result. Source, pinned
environment, commands, live logs and later results are kept under the experiment.

After the timed fitting/SAC commands have stopped at the deadline, the authoritative
post-training workflow is:

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=20261016 \
  venv/bin/python -u -m world_model_rl.select_checkpoint \
  --experiment runs/town01_experiment_001
```

It requires an existing expired clock and acquires its lock, saves the frozen world
model again with deadline metadata, and evaluates the preregistered SAC candidates
on the development plan. Candidates must have actual mixed-replay SAC updates;
supervised-only snapshots cannot replace the requested approach. The extra baseline
is the first completed SAC update after warm-up, plus the fixed hourly checkpoints
and actual final deadline snapshot. Selection ranks arrivals without recorded
violations, arrivals, fewer collision episodes, fewer red violations, route completion,
then earlier archive index. Every development plan must be complete.

It preserves all candidate archives, copies the selected checkpoint to `agent.pt`,
writes `selection_report.json` before final results exist, then drives the complete
reserved 50 episodes with protected traces and saves `final_evaluation.json` and
`final_metrics.json`. SHA-256/config/plan checks bind reports to their actual weights.
An incomplete existing evaluation is preserved and stops this workflow; it cannot
silently rerun favorable subsets. Model/timer behavior is tested independently from
real driving, and the final acceptance threshold remains unchanged.

## Command index

| Command | Purpose | Running CARLA server required? |
| --- | --- | --- |
| [`inspect`](#inspect) | Print client/server versions and map information. | Yes |
| [`collect`](#collect) | Create or append real driving episodes. | Yes |
| [`merge-data`](#scenario-adaptation-and-dataset-import) | Import unique compatible recordings while preserving source splits. | No |
| [`validate`](#validate) | Check recorded episodes and fitting coverage. | No |
| [`fit`](#fit) | Fit a new world model, or fine-tune with `--warm-start`. | No |
| [`fit-termination`](#fit-termination) | Refit only a world model's task-ending readout. | No |
| [`evaluate-model`](#evaluate-model) | Measure held-out model predictions. | No |
| [`train`](#train) | Train or resume SAC using a fitted world model. | During collection blocks |
| [`evaluate`](#evaluate) | Drive with a trained SAC policy and save metrics. | Yes |
| [`run`](#run) | Collect, fit, and train a complete fresh experiment, or resume it. | During collection |
| [`world_model_rl.viewer`](#live-viewer) | Watch the current vehicle in a local browser. | Yes, with another client advancing the simulation |
| [`train_from_scratch.sh`](#train_from_scratchsh) | Launch a fresh experiment with terminal and file logging. | During collection |
| [`recover_from_scratch.sh`](#recover_from_scratchsh) | Append a fixed data batch to a pre-SAC run, then fit fresh model/policy weights. | During collection |
| [`train_collected.sh`](COLLECTED_TRAINING.md) | Train fresh SAC from the consolidated data and preserved model. | No with default profile |

Additional sections: [setup and shared options](#setup-and-shared-options),
[configuration and budgets](#configuration-and-budgets),
[validation and exit status](#validation-and-exit-status),
[artifacts](#artifacts), and [offline verification](#offline-verification).

## Setup and shared options

### Shared five-hour offline deadline

`fit` and `train` accept `--training-clock PATH`, `--training-seconds SECONDS`
(default 18000) and `--checkpoint-seconds SECONDS` (default 300). Without a
clock path, existing update-budget behavior is unchanged. Use the **same clock
path and interval settings** for fitting and policy training. The first valid
training stage starts the persisted wall-clock deadline; subsequent stages and
resumes cannot reset it. Time spent between commands counts. A clock lock
rejects concurrent training commands sharing the same budget.

The clock covers joint world-model fitting, termination-head fitting, actor
warm-up, and SAC. It is checked between complete optimizer updates and after
imagination. At expiry, save completed state and stop further gradients. A long
inference/encoding operation or optimizer step can delay observing the deadline;
the clock never interrupts an optimizer half way through. Prediction validation
and driving evaluation may run after training stops. A failed imagination gate
still blocks SAC; the deadline does not create a real-only fallback.

Timed policy training requires explicit `collection_episodes=0`, with no pending
collection phase. The legacy `refresh_updates` setting is ignored. Collect real data
in fixed batches separately. Timed replay updates never connect to CARLA. The
clock options apply to individual `fit`/`train` stages, not the integrated `run`
command. This is an explicit offline training profile, not a change to package
defaults.

Alongside the latest checkpoint, `checkpoints/` retains immutable candidates at
the interval and at important phase boundaries. The clock JSON lists their
SHA-256 hashes, actual counters, stages and save times. World-model snapshots
include the fitted normalizer, joint optimizer, RNG and any active termination
optimizer; policy snapshots include SAC networks, targets, optimizers, RNG and
partial warm-up counters. Explicit policy resume finishes only the remaining
warm-up updates. Warm-up and SAC logs remain separate. The deadline alone does
not make an untrained or failed-gate checkpoint a valid driving policy.

Command shapes for an existing compatible dataset and a reviewed offline config
are below. They have **not** been launched for `town01_experiment_001`; its host
CARLA/GPU access is still blocked. The large SAC update cap lets the wall-clock
deadline stop the run before that cap. It is not a promised update count.

```bash
venv/bin/python -m world_model_rl fit \
  --config runs/town01_experiment_001/config.json \
  --data runs/town01_experiment_001/data \
  --checkpoint runs/town01_experiment_001/world_model.pt \
  --updates 10000 \
  --training-clock runs/town01_experiment_001/training_clock.json \
  --training-seconds 18000 --checkpoint-seconds 300

venv/bin/python -m world_model_rl train \
  --config runs/town01_experiment_001/config.json \
  --data runs/town01_experiment_001/data \
  --model runs/town01_experiment_001/world_model.pt \
  --output runs/town01_experiment_001 --policy-updates 1000000000 \
  --training-clock runs/town01_experiment_001/training_clock.json \
  --training-seconds 18000 --checkpoint-seconds 300
```

After five hours, use only the reserved development plan to select among saved
neural policy candidates, then evaluate the selected checkpoint using the full
fixed final plan. Record the selection rule and development evidence before
opening final results. No final trajectory may enter fitting, demonstrations,
policy gradients or candidate selection. Runtime access, final experiment
settings and actual full-scope results remain outstanding.

Run the examples from the repository root. They use the existing `venv/bin/python`.
The package uses Python 3.10 or newer; the Python version must also match the CARLA
client wheel. Runtime dependencies are listed in
[`requirements.txt`](requirements.txt): NumPy, PyTorch, and NetworkX. To install
them into an already-created repository environment:

```bash
venv/bin/python -m pip install -r world_model_rl/requirements.txt
```

For commands that connect to CARLA, install this checkout's matching CARLA client
wheel in that same environment and start a dedicated matching server. The repository
server launcher requires `UE4_ROOT` to point to a built Unreal Engine:

```bash
export UE4_ROOT=/absolute/path/to/your/built/UnrealEngine
bash data/start_server.sh
```

Keep the server terminal running. This launcher starts Town01, offscreen rendering,
and RPC port 2000. Collection and policy evaluation own synchronous simulation ticks;
use one driving client per server. The viewer can attach alongside that client.
See [`data/README.md`](../data/README.md) for the server and client setup details.

The main CLI has this form:

```text
venv/bin/python -m world_model_rl COMMAND [OPTIONS]
```

Every main subcommand and the viewer accept these options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | The package's `world_model_rl/config.json` | Load JSON configuration, filling omitted fields with the defaults in `config.py`. |
| `-h`, `--help` | — | Print help and exit without executing the command. |

Place subcommand options **after** the subcommand. There is no top-level `--config`
option. Top-level `-h`/`--help` lists the ten commands:

```bash
venv/bin/python -m world_model_rl --help
venv/bin/python -m world_model_rl train --help
venv/bin/python -m world_model_rl.viewer --help
bash world_model_rl/train_from_scratch.sh --help
bash world_model_rl/recover_from_scratch.sh --help
```

The main CLI's help paths do not import CARLA or PyTorch. Viewer help requires NumPy.
Running a main subcommand imports the runtime dependencies, including for offline work.

Relative paths passed to Python commands resolve from the current working directory.
They do not resolve from the configuration file's directory. Both shell launchers
change to the repository root before resolving their arguments. Quote paths containing
spaces. In the usage blocks below, brackets mean an optional argument; do not type them.

Use the experiment's saved configuration when working with existing data or weights:
`--config runs/EXPERIMENT/config.json`. Commands do not automatically discover this
file from `--data`, `--checkpoint`, `--output`, or `--resume`.

## Scenario adaptation and dataset import

All subcommands except `inspect` accept `--allow-scenario-variation`, setting
`training.allow_scenario_variation=true` for that invocation and saved checkpoints.
It permits different towns, weather, traffic, routes and connection settings while
retaining strict ego, physics/timing, action, observation, termination and reward
checks. The default remains strict. Fitting/policy training requires validation
support for every training scenario, uses all validation episodes and gates each
town/weather/traffic/route configuration separately.

`collect` additionally accepts `--town NAME`, `--weather NAME [NAME ...]`,
`--vehicles N`, and `--pedestrians N`, overriding those collection settings. Episodes
record the actual environment configuration. Collection appends data and does not
update model weights.

```bash
venv/bin/python -m world_model_rl merge-data \
  --config world_model_rl/configs/collected.json \
  --data runs/world_model_rl/data --allow-scenario-variation \
  --source runs/another_collection/data
```

`merge-data --source PATH [PATH ...]` imports unique compatible real episodes into
`--data` (default `runs/default/data`). It preflights all sources, rejects evaluation
traces and cross-split town/seed reuse, preserves source files and split assignments,
reassigns conflicting IDs, and records source provenance in `imported_from` sidecar
metadata. Recording-content checksums deduplicate independently of NPZ compression
or numeric episode IDs. `DATA/imports.jsonl` records counts. Repeated imports add
no duplicate episodes. See [COLLECTED_TRAINING.md](COLLECTED_TRAINING.md) for adaptation
and retraining after import.

## `inspect`

Connect to CARLA and print a JSON summary of the current server. This is a useful
connection check before collection. It queries the current map without loading the
configured town or advancing simulation ticks.

```text
venv/bin/python -m world_model_rl inspect [--config PATH]
```

Only the [shared options](#setup-and-shared-options) apply.

```bash
venv/bin/python -m world_model_rl inspect --config world_model_rl/config.json
```

**Output:** stdout JSON containing `client`, `server`, `map`, `available_maps`,
and `spawn_points`. It writes no dataset or checkpoint. Compare the printed versions;
the driving environment rejects differing client/server versions.

## `collect`

Record complete real episodes using the route-following expert controller. Create a
dataset if necessary, or append episodes with new IDs to a compatible existing dataset.

```text
venv/bin/python -m world_model_rl collect [--config PATH] [--data DIR]
    [--episodes N] [--controller {mixed,autopilot,noisy}]
    [--policy-checkpoint FILE] [--expert-probability P]
    [--perturbations] [--staged-hazards] [--exposure-fraction P]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Configuration for this dataset and CARLA session. |
| `--data DIR` | `runs/default/data` | Dataset directory to create or append to. |
| `--episodes N` | `training.bootstrap_episodes` = `20` | Number of additional episodes to record; must be positive. |
| `--controller MODE` | `mixed` | `autopilot`: BasicAgent expert; `noisy`: expert with action noise; `mixed`: expert on even episode IDs and noisy expert on odd IDs. |
| `--policy-checkpoint FILE` | None | Agent for expert-labelled mixed policy collection. Replaces controller mode; deterministic policy/expert execution is selected independently per decision. |
| `--expert-probability P` | `0.5` | Expert execution probability for labelled policy collection, in `[0,1]`. Every decision still receives a separate expert action label. |
| `--perturbations` | Off | Add repeated sustained steering or acceleration disturbances during exposed episodes. |
| `--staged-hazards` | Off | Add one parked vehicle, adult crossing, or child crossing during exposed episodes. |
| `--exposure-fraction P` | `0.8` | Probability that an episode uses the enabled exposure options, in `[0,1]`. |

Here `autopilot` names the package's BasicAgent controller. It is not a request to
enable Traffic Manager autopilot on the ego vehicle. Mixed collection uses absolute
dataset episode IDs, so alternation continues when appending.

The exposure options are independent. With both enabled, the same exposed episodes
receive both perturbations and hazards; the remaining episodes use the selected
baseline controller. The default is an 80% exposure probability, not a guaranteed
episode proportion or termination rate. Choices use each episode's seed, so
appending and restarting do not restart the exposure schedule. These options apply
only to explicit `collect`, and cannot be combined with `--policy-checkpoint`.

Perturbations start after 3 to 8 simulated seconds and last 2 to 5 seconds, with
3 to 8 seconds of baseline recovery between them. Steering disturbances choose
left or right with magnitude 0.35 to 0.9; acceleration disturbances retain baseline
steering. All disturbances request throttle 0.5 to 1.0 and suppress braking. Below
2 m/s, only acceleration disturbances start. The usual action rate limits still
apply, and requested and applied actions are both recorded.

Staged hazards sample equally among parked vehicles, adults, and children. They
use a non-junction route segment 20 to 50 metres ahead, with varied models and
offsets. Pedestrians wait beside the road, then cross at 0.8 to 1.8 m/s when the
ego approaches within 8 to 20 metres. Each hazard is one additional real actor
beyond the configured background actor budgets, including when those budgets are
zero. Placement checks ground and actor clearance; after 20 unsuccessful candidate
placements collection raises an error. Fixed routes shorter than 30 metres cannot
support staging. All staged actors are cleaned up with the episode.

Neither option changes rewards or termination labels. Actual collisions, sustained
off-road driving, success, and active time limits remain terminal; stuck endings
remain truncations. All completed episodes are retained, including avoided hazards
and recoveries. `collection_exposure` in episode metadata and `collection.jsonl`
records the options, exposure choice, realized disturbance steps, hazard placement,
and crossing trigger. Existing observation and action arrays keep their schema.

Append a fixed batch to a compatible dataset and inspect its coverage:

```bash
venv/bin/python -m world_model_rl collect \
  --config runs/manual/config.json --data runs/manual/data \
  --episodes 40 --perturbations --staged-hazards

venv/bin/python -m world_model_rl validate \
  --config runs/manual/config.json --data runs/manual/data --for-training
```

Use either flag alone to enable just that exposure type. Exposure does not change
the saved split assignments. `termination_support` reports terminal density,
truncations, and ending reasons across every split, while `fitting_coverage`
reports the actual training and fixed validation episodes used by fitting.
Appending data may leave new validation episodes outside the existing cap of
`training.validation_episodes`; inspect the reported IDs before fitting. To expand
that cap, increase it explicitly in the configuration before a fresh fit, without
selecting episodes by their outcomes.

```bash
venv/bin/python -m world_model_rl collect \
  --config world_model_rl/config.json --data runs/manual/data \
  --episodes 20 --controller mixed
```

**Requires:** a dedicated running CARLA server. Collection can load the configured
town and applies synchronous settings for the session. Weather, route selection,
traffic, pedestrians, and step limits come from the configuration.

**Output:** `dataset.json`, `episode_XXXXXX.npz`, matching JSON metadata,
`splits.json`, and appended `collection.jsonl` summaries under `--data`. Existing
episodes are preserved. Configuration/schema/reward/split-seed mismatches and an
interrupted episode sidecar cause failure rather than reuse of incompatible data.

## `validate`

Check stored episodes without connecting to CARLA or fitting a model. Validation
checks shapes, numeric values, actions, timing, rewards, ending flags, schema,
metadata, and split assignments. It also reports whether training and the fixed
validation suite contain enough event support to begin fitting.

```text
venv/bin/python -m world_model_rl validate [--config PATH] [--data DIR]
    [--for-training]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Must be compatible with the dataset. |
| `--data DIR` | `runs/default/data` | Existing recorded dataset. |
| `--for-training` | Off | Exit with code `2` if the reported fitting coverage is insufficient. |

```bash
venv/bin/python -m world_model_rl validate \
  --config world_model_rl/config.json --data runs/manual/data --for-training
```

**Output:** stdout JSON with episode/transition totals, split counts,
`event_decision_counts`, `termination_support`, and `fitting_coverage`. The command does not save a report
file itself. To retain the JSON, redirect stdout to a file in an existing directory:

```bash
venv/bin/python -m world_model_rl validate \
  --config world_model_rl/config.json --data runs/manual/data --for-training \
  > runs/manual/coverage.json
```

Without `--for-training`, insufficient event coverage is reported as
`fitting_coverage.ready=false` but does not by itself cause a nonzero exit status.
Malformed or incompatible recordings still fail. This command checks data support;
it does not measure a model's prediction quality.

## `fit`

By default, fit a randomly initialized attention/RSSM world model, a training-only
normalizer and a new optimizer. `--warm-start FILE` instead fine-tunes an existing
world-model checkpoint while retaining its normalizer and optimizer. Joint fitting
is followed by termination-readout fitting and held-out model evaluation. Model
and termination counters continue from the source; no SAC weights are imported.

```text
venv/bin/python -m world_model_rl fit [--config PATH] [--data DIR]
    --checkpoint FILE [--updates N] [--warm-start FILE] [--allow-scenario-variation]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Dataset-compatible configuration and fitting settings. |
| `--data DIR` | `runs/default/data` | Existing training/validation recordings. |
| `--checkpoint FILE` | **Required** | New output world-model checkpoint file, usually ending in `.pt`. |
| `--updates N` | `training.pretrain_updates` = `10000` | Positive number of joint model optimizer updates. |
| `--warm-start FILE` | Unset | Existing world-model checkpoint to fine-tune; retain its normalizer/optimizer and save to the new `--checkpoint` path. |

```bash
venv/bin/python -m world_model_rl fit \
  --config world_model_rl/config.json --data runs/manual/data \
  --checkpoint runs/manual/world_model.pt --updates 10000
```

The termination stage adds `training.termination_fit_updates` updates (`2000` by
default); these are additional to `--updates`. Only the training split contributes
to normalization or gradients. The validation suite is the first up to
`training.validation_episodes` validation episodes, preserving split assignments.

**Output:** for `--checkpoint runs/manual/world_model.pt`:

| Artifact | Contents |
| --- | --- |
| `runs/manual/world_model.coverage.json` | Pre-fit training/validation support report, also written when coverage blocks fitting. |
| `runs/manual/world_model.pt` | World-model weights, normalizer, optimizer, configuration, counters, validation, and fitting metadata. |
| `runs/manual/world_model.validation.json` | Held-out prediction and termination metrics. |
| `runs/manual/world_model_training.jsonl` | Appended joint fitting progress and timings. |
| `runs/manual/termination_training.jsonl` | Appended termination fitting progress and timings. |

The report names **replace** the checkpoint's suffix: `model.pt` produces
`model.coverage.json` and `model.validation.json`.

An existing checkpoint is refused. Missing fitting coverage exits with code `2`
before device/model allocation or gradient updates. Once fitting completes, a
failed prediction gate is printed and the model/report are still saved; `fit`
does not fail solely because that later gate fails. `train` enforces the gate
before policy updates or further driving collection.

## `fit-termination`

Refit only the task-ending readout of an existing fitted world model. The dynamics,
encoder, reward predictions, and normalizer stay fixed. Class-balanced training
uses full real histories and a correction based on training event prevalence.

```text
venv/bin/python -m world_model_rl fit-termination [--config PATH] [--data DIR]
    --checkpoint FILE --output FILE [--updates N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Must match the source checkpoint and dataset. |
| `--data DIR` | `runs/default/data` | Existing real training and validation episodes. |
| `--checkpoint FILE` | **Required** | Input checkpoint of kind `world_model`; an SAC agent is rejected. |
| `--output FILE` | **Required** | New output world-model checkpoint file. |
| `--updates N` | `training.termination_fit_updates` = `2000` | Positive number of termination-readout updates. |

```bash
venv/bin/python -m world_model_rl fit-termination \
  --config world_model_rl/config.json --data runs/manual/data \
  --checkpoint runs/manual/world_model.pt \
  --output runs/manual/world_model_termination.pt --updates 2000
```

**Requires:** nonempty training and validation splits, both task-ending and
nonterminal training transitions, and `training.batch_size >= 2`. It does not run
the separate `fit` coverage preflight. Missing validation endings can still make
its post-fit prediction gate fail.

**Output:** the new world-model checkpoint, `OUTPUT_STEM.validation.json`, and
`termination_training.jsonl` beside the output checkpoint. Existing output files
are refused; the input checkpoint is preserved. The model version and termination
update counter increase; the joint model update counter is retained. No SAC state
is imported. As with `fit`, a failed post-fit prediction gate is reported but does
not by itself fail the command.

Use the resulting checkpoint with `train --model` to start a fresh SAC policy.
Additional head updates cannot supply missing held-out labels, and repeated
head-only fitting may not resolve a training/held-out generalization gap.

## `evaluate-model`

Measure the world model against recorded held-out episodes. This is offline
prediction evaluation; it does not drive a vehicle or update weights.

```text
venv/bin/python -m world_model_rl evaluate-model [--config PATH] [--data DIR]
    --checkpoint FILE [--split {validation,test}] [--output FILE]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Must match the checkpoint and dataset. |
| `--data DIR` | `runs/default/data` | Existing dataset containing the selected split. |
| `--checkpoint FILE` | **Required** | A world-model or agent checkpoint; agent checkpoints contain the frozen model used by SAC. |
| `--split SPLIT` | `test` | Select `validation` or `test`; there is no training-split option. |
| `--output FILE` | `runs/default/test_model.json` | Destination report file; an existing report is replaced. |

```bash
venv/bin/python -m world_model_rl evaluate-model \
  --config runs/manual/config.json --data runs/manual/data \
  --checkpoint runs/manual/agent.pt --split test \
  --output runs/manual/test_model.json
```

**Output:** the same report is printed as JSON and written to `--output`, creating
parent directories. It includes prediction errors by horizon, reward errors,
object/light metrics, event support, termination precision/recall/Brier score,
per-episode diagnostics, and bounded false-alarm groups. If the split is empty,
the report has `available=false` and a reason.

This command evaluates **all** episodes in the selected split. It does not apply
the capped fixed validation suite used by fitting and SAC training, and it does
not enforce a pass/fail gate through its exit status. `--output` remains
`runs/default/test_model.json` even when `--split validation` is selected, unless
you override it. Keep the test split for final assessment rather than fitting or
choosing changes to pass validation.

## `train`

Train SAC from a fitted world model, or recover the same interrupted SAC session
from an agent checkpoint. Supply exactly one of `--model FILE` or `--resume FILE`.
These are checked at runtime even though argparse lists them as optional.

```text
venv/bin/python -m world_model_rl train [--config PATH] [--data DIR]
    (--model FILE | --resume FILE) [--output DIR]
    [--policy-updates N | --episodes N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Compatible dataset/model settings; resume also checks training settings. |
| `--data DIR` | `runs/default/data` | Existing real dataset; new policy collection episodes are appended here. |
| `--model FILE` | Unset | Input checkpoint of kind `world_model`; initialize fresh SAC actor, critics, and optimizers. |
| `--resume FILE` | Unset | Input checkpoint of kind `agent`; restore the same training session. Requires a file path. |
| `--output DIR` | `runs/default` | Directory for `agent.pt`, saved config, reports, and progress logs. |
| `--policy-updates N` | `training.policy_updates` = `100000` when neither budget flag is supplied | Additional SAC optimizer updates for this invocation; positive and mutually exclusive with `--episodes`. |
| `--episodes N` | Unset | Additional real SAC collection episodes, across dataset splits; positive and mutually exclusive with `--policy-updates`. |

Start a new SAC policy from the fitted world model:

```bash
venv/bin/python -m world_model_rl train \
  --config world_model_rl/config.json --data runs/manual/data \
  --model runs/manual/world_model.pt --output runs/manual \
  --policy-updates 100000
```

Resume that session, adding 10,000 optimizer updates:

```bash
venv/bin/python -m world_model_rl train \
  --config runs/manual/config.json --data runs/manual/data \
  --resume runs/manual/agent.pt --output runs/manual --policy-updates 10000
```

Alternatively, finish after four additional real SAC collection episodes:

```bash
venv/bin/python -m world_model_rl train \
  --config runs/manual/config.json --data runs/manual/data \
  --resume runs/manual/agent.pt --output runs/manual --episodes 4
```

**Behavior:** encode real training replay, evaluate the fixed validation suite,
and require the imagination gate before policy updates or collection. The default
schedule performs 10,000-update policy blocks, then collects two SAC episodes,
rebuilds replay using the frozen world model, and validates again. Neither joint
RSSM fitting nor termination-head fitting occurs during SAC, even when an older
configuration has a positive `refresh_updates` value. All model parameters,
normalization, model versions, and model/termination update counters stay fixed.
An update budget can finish before another collection block; an episode budget
includes the intervening optimizer blocks and replay rebuilds. Replay optimizer
blocks themselves do not connect to CARLA.

With defaults, SAC batches use 75% real and 25% imagined transitions. Imagination
is refreshed every 100 optimizer updates and spans one decision (`0.1` simulated
seconds). The normalizer stays fixed. Failed validation blocks further SAC with
exit code `2`; there is no real-only fallback.

Optional actor warm-up and demonstration anchoring are selected in configuration,
not new CLI flags. Warm-up uses training-split labels only, initializes a new actor,
leaves critics fixed and does not count toward `--policy-updates`. `recorded` accepts
autopilot/DAgger labels; geometric teachers can label any training collector's current
states. A geometric teacher is never called during evaluation. See the configuration
table below and PROTOTYPE.md for the exact loss and candidate 008 settings.

**Resume requirements:** restore networks, optimizers, RNG state, model version,
and phase counters. Environment/reward/model configuration must match; training
settings must match apart from device, update budgets, and the ignored legacy
`refresh_updates` value. Old checkpoints with `refresh_pending=true` rebuild replay
without fitting the model. New checkpoints record `replay_pending`,
`world_model_frozen=true`, and `refresh_pending=false`. The dataset's next
episode ID must match the checkpoint frontier. Do not append episodes separately
to a dataset that belongs to a resumable agent. An unfinished, unsaved simulator
episode restarts from its seed; partial simulator state is not restored.

**Output:** `agent.pt`, `config.json`, `validation.json`, and `training.jsonl`.
SAC creates no world-model or termination fitting logs. The dataset remains at
`--data`; setting `--output` does **not** change `--data`. A fresh `train` refuses
an output containing `agent.pt`; it can use the directory containing the initial
world model. Resume updates the agent checkpoint at `--output/agent.pt`.
Checkpoints are saved every `training.checkpoint_updates` updates (`1000` by
default) and at phase boundaries. Validation failures save `validation.json`;
the initial gate can fail before a new `agent.pt` or output config is written.

## `evaluate`

Drive using the deterministic SAC actor from an agent checkpoint. No fitting or
policy updates occur, and evaluation episodes are not appended to a dataset.

```text
venv/bin/python -m world_model_rl evaluate [--config PATH] --checkpoint FILE
    [--episodes N] [--output FILE] [--seed-offset N] [--plan FILE] [--record-traces]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Must match the agent checkpoint's environment/reward/model settings. |
| `--checkpoint FILE` | **Required** | Agent checkpoint containing SAC state; a world-model-only checkpoint is rejected. |
| `--episodes N` | `10` | Positive number of live evaluation episodes. |
| `--output FILE` | `runs/default/evaluation.json` | Destination JSON report; an existing report is refused. |
| `--seed-offset N` | `100000` | Integer offset added to `training.seed`; episode `i` uses `training.seed + seed_offset + i`. |
| `--plan FILE` | None | Fixed JSON scenarios (`format: 1`, `town`, and `scenarios` with episode ID, seed, weather). Overrides episode count and scenario seeds. |
| `--record-traces` | Off | Save diagnostic NPZ/JSON trajectories to a sibling `OUTPUT_STEM_traces` directory; existing paths are refused. Traces are explicitly excluded from training. |

```bash
venv/bin/python -m world_model_rl evaluate \
  --config runs/manual/config.json --checkpoint runs/manual/agent.pt \
  --episodes 10 --seed-offset 100000 --output runs/manual/evaluation.json
```

**Requires:** a dedicated running CARLA server and the matching client.

**Output:** a JSON file with `episodes`, `seed_offset`, `mean_return`,
`success_rate`, and `mean_route_completion`. Episode summaries include route completion,
ending reason, event counts, collision rate, lane/heading errors, and comfort
metrics. Episode progress is printed to the terminal; the aggregate JSON is
written to the file rather than printed. Parent directories are created. Reports are
saved after each episode, include `complete`, `per_weather`, `collision_episode_rate`,
and `prototype_acceptance`, and cannot pass acceptance while incomplete. The prototype
criterion requires 50 episodes balanced ten per configured weather across five weathers,
at least 80% successes, and at most 10% collision episodes. A development suite is not a
substitute for this final evaluation. Freeze the plan before training and keep its seeds
and trajectories out of model/policy fitting.
Reports additionally contain `red_light_violations`, `red_light_episodes` and
`red_light_episode_rate`, overall and per weather. These supplement the stated
success/collision acceptance rule; success alone does not establish traffic compliance.
Recorded evaluation traces carry `split: evaluation` and never-train provenance;
dataset save/load rejects them even if wrapped in a compatible dataset manifest.

## `run`

Run the full pipeline in a fresh experiment directory, or resume an existing
experiment's agent. This command owns its data/checkpoint paths under `--output`.

```text
venv/bin/python -m world_model_rl run [--config PATH] [--output DIR]
    [--resume] [--policy-updates N | --episodes N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Bootstrap, model, SAC, and environment settings. Pass the saved config explicitly for resume. |
| `--output DIR` | `runs/default` | Fresh experiment directory, or the experiment to resume. |
| `--resume` | Off | Resume `OUTPUT/agent.pt` with `OUTPUT/data`. This is a boolean flag, with no path argument. |
| `--policy-updates N` | `training.policy_updates` = `100000` when neither budget flag is supplied | Positive additional SAC optimizer-update budget. |
| `--episodes N` | Unset | Positive additional SAC collection-episode budget; mutually exclusive with `--policy-updates`. |

Start a fresh experiment:

```bash
venv/bin/python -m world_model_rl run \
  --config world_model_rl/config.json --output runs/experiment \
  --policy-updates 100000
```

The fresh sequence is:

1. Require a new or empty output directory and save `config.json`.
2. Collect `training.bootstrap_episodes` new episodes into `OUTPUT/data`.
3. Check coverage, initialize a new model/normalizer/optimizer, and fit
   `OUTPUT/world_model.pt`.
4. Validate and start fresh SAC weights, writing `OUTPUT/agent.pt` as training proceeds.

At least ten bootstrap episodes and a positive `training.pretrain_updates` budget
are required. `--episodes` changes the subsequent SAC collection budget; it does
not change bootstrap collection. There is no `run --data` or `run --model` option.

Resume the same interrupted experiment:

```bash
venv/bin/python -m world_model_rl run \
  --config runs/experiment/config.json --output runs/experiment \
  --resume --policy-updates 10000
```

Resume requires `OUTPUT/agent.pt` and applies the [train resume checks](#train).
It does not resume partially completed bootstrap collection or initial fitting.
If no agent exists, inspect the recorded artifacts and use the individual stages
or the recovery launcher as appropriate. For an external dataset, such as one used
by the recovery launcher, use `train --data ... --resume FILE`.

## Live viewer

The viewer is a separate Python module, not a main-CLI subcommand:

```text
venv/bin/python -m world_model_rl.viewer [--config PATH] [--port N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PATH` | Packaged config | Uses `environment.host` and `environment.port` to locate CARLA. |
| `--port N` | `8080` | Local HTTP port, from `1` through `65535`; independent of CARLA's RPC port. |
| `-h`, `--help` | — | Print help and exit. |

In another terminal while collection or training is running:

```bash
venv/bin/python -m world_model_rl.viewer \
  --config runs/experiment/config.json --port 8080
```

Open <http://127.0.0.1:8080>. The HTTP server binds to loopback only. It attaches
a 960 × 540 chase camera to the vehicle with role `world_model_rl`, follows new
vehicles across episodes, and waits during offline fitting or replay updates.
It also works with the offscreen server. The viewer does not tick, change world
settings, or control the vehicle; extra rendering uses GPU time.

Use a different `--port` if the local HTTP port is occupied. The CARLA connection
uses a fixed three-second timeout in the viewer. Ctrl+C stops the viewer and
cleans up its camera. It produces no saved video; recorded NPZ episodes contain
no camera footage for playback. Its local HTTP endpoints are `/` (page),
`/status` (JSON), and `/frame.png` (latest frame, or HTTP 204 while none is available).

The dashboard includes speed (km/h and m/s), the current observation's object slots
(class, actor ID, position and confidence), and accumulated reward for the current
episode. Rewards reset at episode start. The observation source is displayed explicitly:
the default objects are CARLA ground truth with geometric visibility filtering.
SAC/DAgger episodes also show encoder attention weights by input and by head;
expert-only bootstrap episodes have no model attention. This is structured token
attention, not a camera heatmap or a causal explanation of policy actions.

Additional panels show actual versus cautious target speed, route completion and distance
remaining along the route, and steering/throttle/brake values applied after action limits.
The reward table shows signed components for the last decision and their episode totals;
all accumulated reward values reset at the next episode. Steering uses CARLA's signed
`[-1, 1]` scale (left to right); throttle and brake are displayed as percentages.

Restart both driving and viewer processes after upgrading. They share an atomic
temporary snapshot on the same machine/user, keyed by `environment.host` and
`environment.port`; use matching endpoints. `/status` includes a `telemetry` object
or `null` when unavailable. Data older than five seconds is flagged stale. The
state's simulation frame is displayed because camera frames are independently
sampled. Telemetry failures warn once in the driving process and do not stop driving.

## `train_from_scratch.sh`

Launch `run` with a fresh directory, unbuffered Python output, and a sibling log.

```text
bash world_model_rl/train_from_scratch.sh [OUTPUT_DIR] [POLICY_UPDATES] [CONFIG]
```

| Positional argument | Default | Meaning |
| --- | --- | --- |
| `OUTPUT_DIR` | `runs/from_scratch_<YYYYMMDD_HHMMSS>_<random>` | New experiment directory. |
| `POLICY_UPDATES` | `100000` | Positive SAC update budget, passed explicitly as `--policy-updates`. |
| `CONFIG` | `world_model_rl/config.json` | Existing JSON configuration file. |

```bash
bash world_model_rl/train_from_scratch.sh
bash world_model_rl/train_from_scratch.sh runs/experiment5 50000 my_config.json
```

Arguments are positional: to provide `CONFIG`, provide the first two arguments
as well. `-h` or `--help` as the first argument prints help and exits.

The script changes to the repository root, checks the repository interpreter and
configuration, and refuses any existing output directory, file, symlink, or
`OUTPUT_DIR.log`. Unlike direct fresh `run`, it also refuses an empty existing
directory. It then executes the equivalent of:

```text
venv/bin/python -u -m world_model_rl run --config CONFIG --output OUTPUT_DIR
    --policy-updates POLICY_UPDATES
```

**Output:** the normal `run` artifacts plus combined stdout/stderr streamed to
the terminal and `OUTPUT_DIR.log`. The script has no resume mode and does not
start CARLA. Its explicit update budget overrides `training.policy_updates` in
the config. Argument/preflight failures exit with code `2`; pipeline failures
remain nonzero through `tee` (normally the Python command's failure status).

## `recover_from_scratch.sh`

Recover an experiment blocked before SAC by recording a fixed additional batch,
checking coverage, and fitting fresh world-model and policy weights. Existing
recordings are retained and extended; existing checkpoint weights are not loaded.

```text
bash world_model_rl/recover_from_scratch.sh RUN_DIR [NEW_EPISODES]
    [POLICY_UPDATES] [OUTPUT_DIR]
```

| Positional argument | Default | Meaning |
| --- | --- | --- |
| `RUN_DIR` | **Required** | Source run containing `config.json` and `data/dataset.json`, without `agent.pt`. |
| `NEW_EPISODES` | `20` | Positive fixed number of additional mixed-controller episodes to record. |
| `POLICY_UPDATES` | `100000` | Positive SAC update budget after successful fitting/validation. |
| `OUTPUT_DIR` | `RUN_DIR/recovery_<YYYYMMDD_HHMMSS>_<random>` | New directory for the fresh model, policy, and copied config. |

```bash
bash world_model_rl/recover_from_scratch.sh runs/experiment4 20 100000
bash world_model_rl/recover_from_scratch.sh \
  runs/experiment4 20 50000 runs/recovered_experiment4
```

`-h` or `--help` as the first argument prints help and exits. All paths resolve
from the repository root. The launcher requires the repository interpreter, refuses
a source with `agent.pt`, and refuses an existing output path or sibling log.

The sequence is:

1. Create `OUTPUT_DIR` and copy `RUN_DIR/config.json` there.
2. Run `collect` once, appending exactly `NEW_EPISODES` if collection completes.
3. Run `validate --for-training`; stop on inadequate coverage.
4. Run fresh `fit`, writing `OUTPUT_DIR/world_model.pt`.
5. Run `train --model`, writing fresh SAC state under `OUTPUT_DIR` if its gate passes.

It does not loop until favorable validation outcomes appear. Data are appended to
**`RUN_DIR/data`**, not `OUTPUT_DIR/data`; old episode/split assignments are preserved.
The configured validation cap can mean that newly appended validation episodes do
not join an already-full fitting/training suite. Neither more head fitting nor
appending data guarantees a passing gate.

**Output:** fresh checkpoints/reports/logs under `OUTPUT_DIR`, the extended source
dataset, and terminal/file progress in `OUTPUT_DIR.log`. It stops at a failed
stage and retains completed work. Argument/preflight failures exit with code `2`;
stage failures propagate through the pipeline. A running dedicated CARLA server
is required for collection; the script does not launch it.

To resume an agent produced by this script, pass its external dataset explicitly:

```bash
venv/bin/python -m world_model_rl train \
  --config runs/recovered_experiment4/config.json --data runs/experiment4/data \
  --resume runs/recovered_experiment4/agent.pt \
  --output runs/recovered_experiment4 --policy-updates 10000
```

## Configuration and budgets

Host, town, weather, model sizes, device, and most training settings are JSON
configuration fields rather than CLI flags. Copy
[`config.json`](config.json) to a new path for a new experiment, edit it, and
select it using `--config PATH`. Only the `environment`, `reward`, `model`, and
`training` sections and their declared fields are accepted; see
[`config.py`](config.py) for all fields and validation rules.

The settings most directly related to command behavior are:

| Field | Default | Effect |
| --- | --- | --- |
| `environment.host` / `environment.port` | `127.0.0.1` / `2000` | CARLA connection. Viewer `--port` controls a separate HTTP listener. |
| `environment.tm_port` | `8000` | Traffic Manager port for driving sessions. |
| `environment.timeout` | `30.0` seconds | Driving/inspect client timeout; viewer uses its own timeout. |
| `environment.town` | `Town01` | Map loaded by driving sessions if needed. |
| `environment.fixed_delta_seconds` / `action_repeat` | `0.05` / `2` | One decision spans `0.1` simulated seconds. |
| `environment.max_steps` | `1800` | Active decisions before timeout task failure; legitimate waits at lights pause this budget. |
| `reward.time_limit` | `100.0` | One final negative cost on active deadline expiry. |
| `environment.no_rendering_mode` | `false` | Disable viewport drawing for structured ground-truth runs; camera providers require rendering. Prior settings are restored on close. |
| `training.seed` | `20261016` | Seeds collection, training, and fixed dataset split assignments. |
| `training.device` | `auto` | CUDA when available, otherwise CPU; `cpu` forces CPU and `cuda` requires CUDA. |
| `training.bootstrap_episodes` | `20` | Default standalone collection count and fresh `run` bootstrap count. |
| `training.pretrain_updates` | `10000` | Default joint updates in `fit` and fresh `run`. |
| `training.termination_fit_updates` | `2000` | Readout updates after explicit model fitting and default for `fit-termination`; unused by SAC. |
| `training.policy_updates` | `100000` | Default additional SAC update budget for direct `train`/`run`. |
| `training.policy_block_updates` | `10000` | SAC updates between collection phases. |
| `training.collection_episodes` | `2` | SAC episodes collected per collection phase; must be positive for an episode budget. |
| `training.refresh_updates` | `0` | Deprecated compatibility field; accepted from saved configs but ignored during SAC. |
| `training.minimum_real_steps` | `500` | Minimum real training transitions for fitting/SAC. |
| `training.replay_steps` | `100000` | Recent real training replay budget, loaded as whole episodes. |
| `training.synthetic_fraction` | `0.25` | Fraction of SAC samples from imagined replay. |
| `training.imagination_horizon` | `1` | Imagined decisions per branch; supported values are `1`, `3`, or `5`. |
| `training.imagination_refresh_updates` | `100` | SAC update interval between imagined replay generations. |
| `training.validation_episodes` | `10` | Maximum whole episodes in the fixed fitting/SAC validation suite. |
| `training.checkpoint_updates` | `1000` | Periodic SAC checkpoint interval; phase boundaries also save checkpoints. |
| `training.behavior_cloning_updates` | `0` | Actor-only supervised warm-up updates before fresh SAC training; not repeated by resume. |
| `training.behavior_cloning_weight` | `0.0` | Continued demonstration penalty in SAC actor updates; zero disables anchoring. |
| `training.behavior_cloning_teacher` | `recorded` | `recorded`, `geometry_v1`, `geometry_v2` or `geometry_v3`; geometric teachers require the Tesla Model3 ego. |
| `training.policy_observation_skip` | `false` | Add 169 current/predicted structured values beside latent features; 336 total with the default model. |
| `training.behavior_cloning_traffic_fraction` | `0.0` | Demonstration fraction reserved for near red/yellow stopping: 3/4 moving, 1/4 waiting; remainder keeps manoeuvre/uniform balance. |
| `training.behavior_cloning_permute_objects` | `false` | Training-only demonstration object geometry/mask/class permutation; requires the skip. Implemented but no trained checkpoint uses it yet. |

`--episodes`, `--updates`, and `--policy-updates` accept positive integers when
present. `--updates` is a model-fitting budget; `--policy-updates` counts SAC
optimizer updates. `train`/`run --episodes` counts additional live SAC episodes
and overrides the configured update-budget stopping rule. It excludes fresh-run
bootstrap episodes. Resume budgets are additional work, not lifetime totals.
`evaluate --episodes` defaults to ten independently of the training configuration.

Datasets enforce compatible environment/reward/model configuration and the original
split seed. Checkpoints enforce environment/reward/model compatibility. Same-run
SAC resume additionally checks training settings apart from device and budgets.
Use the saved configuration rather than changing dynamics or model dimensions
under existing artifacts. A new observation contract requires compatible fresh data
and model weights.

## Validation and exit status

Two different checks govern training:

1. **Fitting coverage:** enough real training transitions, at least one training
   task ending and one nonterminal transition, plus configured validation support.
   Defaults require at least one validation ending and twenty nonterminal
   validation transitions. `validate --for-training` exposes this check;
   `fit` performs it before gradients.
2. **Prediction gate:** held-out normalized prediction, reward, speed, lane,
   heading, and termination metrics must pass before SAC updates or additional
   SAC collection. The same fixed suite is retained across replay rebuilds and resume.

Stuck endings are truncations and count as nonterminal labels. Active deadline
expiry, success, collision and off-road termination supply positive task-ending labels.
Whole episodes receive fixed seeded 8/1/1 train/validation/test assignments per
complete block of ten IDs; commands do not move episodes to improve coverage.

Default prediction-gate thresholds are:

| `training` field | Requirement |
| --- | --- |
| `gate_normalized_rmse` | Normalized prediction RMSE ≤ `0.75`. |
| `gate_reward_mae` | Reward-sum MAE ≤ `10.0`. |
| `gate_speed_mae` | Forward-speed MAE ≤ `2.0` m/s. |
| `gate_lane_mae` | Lane-offset MAE ≤ `0.5` m. |
| `gate_heading_mae` | Heading MAE ≤ `0.2` radians. |
| `gate_terminal_brier` | Termination Brier score ≤ `0.2`. |
| `gate_terminal_recall` | Termination recall ≥ `0.5`. |
| `gate_terminal_precision` | Termination precision ≥ `0.5`. |
| `gate_min_terminal_positive` | At least `1` positive validation label. |
| `gate_min_terminal_negative` | At least `20` negative validation labels. |

Prediction checks use the configured imagination horizon. Missing held-out episodes,
insufficient sequence lengths, nonfinite metrics, or insufficient ending support
also block SAC. Twenty collected episodes do not guarantee that the gate passes.
Inspect `termination.endings`, `termination.by_episode`, and
`termination.false_positive_groups` to understand failures. False-alarm groups
describe the first twenty contiguous groups in time order; total metrics still
include every transition. These contexts do not establish the cause of an error.

| Situation | Exit behavior |
| --- | --- |
| Successful command, or help | `0`. |
| Missing/unknown CLI arguments, unsupported command, invalid explicit budget, or invalid viewer port | Argparse exits `2`. |
| `validate --for-training` or `fit` fails fitting coverage | `2`, with an explanation; fitting has not begun. |
| `train` or `run` fails the prediction gate | `2`; inspect `OUTPUT/validation.json`. |
| `fit` or `fit-termination` completes but prediction gate fails | Still `0`; the saved model/report can be inspected, but SAC will enforce the gate. |
| `evaluate-model` reports poor predictions or an empty split | Still `0` if evaluation/report writing completes; inspect the JSON. |
| Other runtime failures, such as file conflicts, incompatible configs, unavailable requested CUDA, connection failures, or invalid recordings | Nonzero, generally a Python traceback; these are not all normalized to `2`. |

## Artifacts

For a complete `run --output runs/experiment`, the principal artifacts are:

| Path | Meaning |
| --- | --- |
| `config.json` | Experiment configuration, written before fresh collection. |
| `data/dataset.json` | Dataset schema, configuration, reward version, and split rule. |
| `data/episode_XXXXXX.npz` / `.json` | Numeric trajectory and its provenance/summary metadata. |
| `data/splits.json` | Whole-episode split membership. |
| `data/collection.jsonl` | Expert bootstrap/standalone collection summaries. SAC collection is logged in `training.jsonl`. |
| `world_model.pt` | Initial fitted world model, kept frozen throughout SAC. |
| `world_model.coverage.json` | Pre-fit label and transition support. |
| `world_model.validation.json` | Initial fitted model evaluation. |
| `agent.pt` | Latest SAC state, frozen world model, optimizer/RNG state, and resume counters. |
| `validation.json` | Most recent SAC validation report, including one that blocks training. |
| `training.jsonl` | SAC progress, collection summaries, replay rebuilds, and timings. |
| `world_model_training.jsonl` | Explicit joint model fitting progress. |
| `termination_training.jsonl` | Explicit termination fitting progress. |

The launchers additionally write `OUTPUT_DIR.log` **beside** the output directory.
Direct Python commands do not create that terminal transcript automatically.
`evaluate-model` and `evaluate` write their separately selected JSON output files.
Most diagnostic JSON reports replace existing files; evaluation refuses an existing
result/trace path, and fitting refuses existing checkpoint paths. JSONL logs append. A refused or failed stage
can leave only the artifacts completed before it stopped.

## Offline verification

Run the existing package tests:

```bash
venv/bin/python -m pytest world_model_rl/tests -q
```

Pytest is a development dependency and is not listed in the runtime requirements.
The tests use generated arrays, small networks, and fake environments; they do not
start CARLA or perform full live training. To check launcher syntax without
executing it:

```bash
bash -n world_model_rl/train_from_scratch.sh
bash -n world_model_rl/recover_from_scratch.sh
```

See [VALIDATION.md](VALIDATION.md) for recorded verification evidence and the
remaining live simulator checks.

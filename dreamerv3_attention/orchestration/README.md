# Passive run watcher

Run from the CARLA repository root:

```sh
python -m dreamerv3_attention.orchestration.watch --run runs/<run> --once
python -m dreamerv3_attention.orchestration.watch --simulate
```

Without `--run`, the newest directory below `runs/` containing `metrics.jsonl`
or `live.json` is selected, including nested output directories. `--once`
prints one JSON snapshot and exits; interactive use can poll at `--interval`
seconds. The watcher only reads files. It never writes to run directories,
connects to CARLA, restarts a process, or trains. `nvidia-smi` is optional and
queried read-only with a three-second timeout.

Incidents are compact objects with `type`, `severity`, `detail`, and `evidence`.
The watcher detects non-finite loss, a drop between consecutive windows of
completed episode returns (five episodes per window; default drop is the larger
of 1 return point or 50% of the prior mean), gradient norm over 100 when such a
metric exists (nonfinite norms also alert), unchanged active steps/live metadata, stale active-run
checkpoints, CARLA disconnect/error text, and GPU memory use at or above 95%.
Run activity uses recent `live.json` status and a nonblocking shared-lock
probe of the existing `live.lock`. A lock alone counts as active only when no
status file exists; an `evaluating` status never counts as training. Stale
checkpoints are reported only while training is active. The lock is opened
read-only and is not held beyond the probe.
Current training rows do not log gradient norm, so that detector is dormant
unless the metric is added by a separate change. Reward collapse uses sums of
per-step rewards between terminal flags when the row format does not provide a
completed episode return. Nonfinite loss and gradient incidents remain active
until a later finite corresponding metric arrives. Other incidents clear when
their signal recovers. Per-metric evidence scans are bounded to 16 string
fields and 2,048 characters.

JSONL reading tracks byte offset and file identity, preserves an incomplete
final line, and resets after inode replacement or truncation. Malformed lines
are skipped. Startup and lagged-reader resync parse only the last 2 MiB by
default (`--initial-tail-bytes` configures the bound); the first partial row in
that suffix is discarded. Historical anomalies before the startup suffix are
not replayed. Each poll emits active conditions; metric incidents stay active
until a healthy matching metric arrives, and condition incidents clear when
their signal recovers. With no `--audit` path, stdout JSONL is the audit stream. An
explicit `--audit PATH` appends `detection`, `diagnosis`, and `resolved` events
to that path only on incident transitions; it rejects destinations inside the
watched run. Supervisors/fixers may
append `approval`, `fix`, and `test` events to the same file, with timestamp,
actor, plan, outcome, and evidence fields, to record the complete workflow.
The audit destination is optional and must be selected explicitly.

Workflow: Luna `run-triage` is read-only. Sol `watcher-planner` acts as
supervisor, checkpoints selected-file diffs, approves bounded low-risk patches,
and verifies focused tests. Luna `run-fixer` executes only an explicitly
approved bounded plan. Escalate reward/model/schema/checkpoint migration,
broad changes, more than two failed bounded attempts, and any live GPU/run
action to the user. No role may auto-repair, restart, or control CARLA. Keep a
single GPU workload under the existing shared training lock; this watcher
does not acquire it or run training.

## Requesting one bounded training run

Ask Codex for a plan first. The supervisor must inspect the selected config and
training CLI and show one exact command, a unique output path, wall-clock
budget, checkpoint choice if applicable, and whether the run connects to an
idle CARLA server or starts UE4. It must not start until you approve that exact
command. For example, the command shape is:

```sh
python -m dreamerv3_attention train --config dreamerv3_attention/configs/baseline.json --output runs/<unique-name> --budget-seconds <seconds>
```

Add `--checkpoint <path>` only when resuming is explicitly selected and
approved. Add `--ue4-root <path>` only when starting UE4 is explicitly selected
and approved. The actual command and whether it will connect to or start CARLA
must be checked against the current CLI before approval.

Copyable planning prompt:

> Prepare one bounded DreamerV3 training plan. Inspect the config and CLI, then
> show the exact command, config, unique output directory, wall-clock budget,
> checkpoint choice, and whether it will connect to an idle CARLA server or
> start UE4. Do not execute or delegate training yet. Wait for me to approve
> that exact command. A new run always needs a new explicit approval.

After approval, Sol delegates one exact invocation to Luna `run-trainer`. The
existing training CLI serializes work with `workload_lock`; if the output path
already exists or the lock is busy, the trainer stops without retrying. It
never changes settings, resumes, restarts, extends, launches a second run, or
starts UE4 unless the approved command explicitly includes it. For example,
reply: **“Approved to run this exact command once.”** Approval of a plan or
prior run does not authorize a new run or a resume. The supervisor then reads
the final metrics with the passive watcher and reports the outcome; it does
not automatically fix issues. Fixes follow the separate bounded `run-fixer`
approval workflow.

Optional `--diagnose` invokes installed `codex exec` with the selected model
(default `gpt-6-luna`), `--sandbox read-only`, and `--ephemeral`. It submits at
most 6,000 characters of evidence, asks for a short answer, times out after 90
seconds, allows one diagnostic per incident cooldown (one call per five
minutes globally), and does not retry. The CLI exposes
no hard token cap here, so payload size and call frequency are the enforceable
bounds. It relies on the user's installed Codex subscription authentication;
it does not require or inspect API keys. Diagnosis is advisory and does not
execute returned commands. The prompt treats telemetry as untrusted data and
asks Codex to analyze only the supplied JSON without reading other files.

Offline tests require no CARLA or CUDA:

```sh
python -m pytest dreamerv3_attention/tests/test_orchestration_watch.py -q
```

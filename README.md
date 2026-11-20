# CARLA DreamerV3 with attention

Autonomous-driving research in CARLA using TorchRL DreamerV3, multimodal
attention, camera perception and a safety supervisor. The main application is
`dreamerv3_attention/`; commands below run from the repository root.

## Models and evaluation results: separate OneDrive download

**The strongest driving model and its evaluation metrics are shared separately
on OneDrive. They are not included in the GitHub checkout.** Obtain the shared
OneDrive folder from the project owner; no public download URL is configured
in this repository. Download the checkpoint together with its matching
`config.json`, scenario files and evaluation reports, preserving relative paths.

Use the OneDrive reports to assess the actual checkpoint,
scenario coverage and results. This guide does not establish a new performance
score or an attention improvement.

For the examples below, put the chosen model and its matching configuration in
`final/latest.pt` and `final/config.json`. Preserve
any additional paths referenced by the downloaded configuration. Do not mix
checkpoints and configurations with different observation or model dimensions.

## Repository layout and GitHub contents

| Path | Purpose |
| --- | --- |
| `dreamerv3_attention/` | Sensors, observation tokens, attention/RSSM, training, evaluation and web viewer |
| `dreamerv3_attention/configs/` | Baseline config files. Do not use these, use the configs in OneDrive instead |
| `perception/` | YOLO object and traffic-light detection, browser demo and live depth tools |
| `joint_road_model/` | Lane-paint segmentation, crossing detection and lane-centre geometry |
| `data/` | Dataset collection, configuration and simulator preflight utilities |
| `PythonAPI/`, `LibCarla/`, `Unreal/` | CARLA bindings, native library and simulator project |
| `requirements.txt` | Consolidated Python dependencies for the project code and demos |


## Prerequisites and installation

Use Linux and Python 3.10 for the main application; its workload locking uses
`fcntl`. The reference installation used Python 3.10.22, CARLA 0.9.16,
PyTorch 2.7.1 with CUDA 12.6, TorchRL 0.14.1 and TensorDict 0.14.3.
An NVIDIA GPU is recommended for live training. CPU smoke checks do not need
CARLA running or downloaded model weights. Standalone perception demos may
support other platforms; that does not establish full application support.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m dreamerv3_attention preflight
```

The root requirements cover Dreamer, object/road perception, OCR, the Flask
road demo and pytest. They replace the project-specific requirement files;
CARLA's upstream build, documentation and optional integration requirements
remain separate. Transitive dependencies are installed by pip. Select the
appropriate PyTorch wheel source for your CUDA installation while retaining
the pinned torch/torchvision pair. Preflight reports versions and CUDA status;
it does not start the simulator. The consolidated environment still needs
validation on a fresh machine, particularly for downloaded perception weights.

Install a **matching CARLA 0.9.16 server and Python client**. Installing the
`carla` pip package supplies a client, not the Unreal simulator, maps or assets.
Use a matching packaged server, or build this checkout with its supported
Unreal Engine 4.26 toolchain and assets. For a source build, set `UE4_ROOT` to
the built engine directory, then use `make PythonAPI` and `make launch` after
completing CARLA's native build prerequisites. Python requirements do not
install the C++ compiler, Unreal Engine, Vulkan drivers or map content.

## Check the installation without driving

```bash
python -m pytest dreamerv3_attention/tests -q
python -m unittest discover -s data/tests
python -m dreamerv3_attention smoke --config dreamerv3_attention/configs/smoke.json
```

Smoke collects synthetic transitions, performs real CPU learner updates and
checks a saved checkpoint in a unique directory under `/tmp`. It verifies the
software path, not learned driving performance. Live integration tests are
skipped by default; against an idle matching server they can be enabled with:

```bash
DREAMERV3_APPROVED_LIVE_TEST=1 python -m pytest dreamerv3_attention/tests/test_live.py -q
```

## Start CARLA and run the project

Start an idle dedicated CARLA server with Town01, normally on RPC port 2000
(with ports 2001 and 2002 also available). With a source-built engine, add
`--ue4-root /path/to/UnrealEngine_4.26` to a live command to let it launch and
clean up its own server. Without that flag, it connects to an existing server.

This step is only necessary when starting a fresh training, otherwise use the model and configs from OneDrive.
The baseline `attention.json` enables attention and uses CUDA; `baseline.json`
disables attention for comparison. Copy a configuration and set `device` to
`cpu` if needed. Configure host, port, town and perception there. Every live
command needs `--budget-seconds` between 1 and 86400, including startup time.
Use a fresh output directory for each run; populated outputs are not overwritten.

Train a new attention model, or collect replay without learner updates:

```bash
python -m dreamerv3_attention train \
  --config dreamerv3_attention/configs/attention.json \
  --budget-seconds 600 --output runs/attention_train
python -m dreamerv3_attention collect \
  --config dreamerv3_attention/configs/attention.json \
  --budget-seconds 600 --output runs/attention_collect
```

Resume training with `train --checkpoint runs/NAME/latest.pt` and the matching
training configuration. Add `--world-model-only` to update only the world model.
Training requires `split: train`; keep validation and held-out replay separate.
Curriculum configurations under `runs/experiment_26/` are ignored artifacts
and must be obtained from OneDrive before using that experiment's workflow.

Run the downloaded policy for one episode:

```bash
python -m dreamerv3_attention infer \
  --config runs/example/config.json \
  --checkpoint runs/example/latest.pt \
  --budget-seconds 180 --output runs/onedrive_infer
```

For evaluation, create `runs/example/evaluation.json` from the matching
model configuration, retaining its model/perception settings and selecting the
intended validation or held-out suite. The tracked `configs/evaluation.json`
is a starting point only when its model settings match the checkpoint.
Named curriculum training scenarios are training sanity checks, not held-out
results. Compare actor and controller on identical scenarios and safety settings:

```bash
python -m dreamerv3_attention evaluate \
  --config runs/example/evaluation.json \
  --checkpoint runs/example/latest.pt \
  --budget-seconds 600 --output runs/actor_eval
python -m dreamerv3_attention evaluate \
  --config runs/example/evaluation.json --controller \
  --budget-seconds 600 --output runs/controller_eval
python -m dreamerv3_attention compare \
  --baseline runs/controller_eval/result.json \
  --candidate runs/actor_eval/result.json --output /tmp/comparison.json
```

Live runs write configuration, JSONL metrics, checkpoints and `result.json`.
Evaluation measures completion/progress, collisions, traffic and crossing
compliance, speed/lane tracking, inactivity, interventions and policy latency.
Report supervisor interventions and any restart assistance alongside success;
assisted behavior alone does not establish raw policy competence. Use multiple
seeds and held-out routes for meaningful model comparisons.

Other offline tools include `validate --checkpoint MODEL/latest.pt
--data-checkpoint HELD_OUT/latest.pt` for one held-out world-model loss batch,
and `audit-controls --trace EVAL/trace.jsonl` for control diagnostics.
Run `python -m dreamerv3_attention --help` for the complete CLI.

## Watch a run

```bash
python -m dreamerv3_attention viewer
```

Open **http://localhost:8080** while training or evaluation runs in another
terminal. The viewer follows active runs and shows RGB frames, controls,
progress and metrics. Use `--training-run runs/NAME` to select a run or
`--web-port` to change the port. Monitoring does not start a drive.

For a separate interactive checkpoint drive on an idle server:

```bash
python -m dreamerv3_attention viewer \
  --config runs/example/config.json \
  --checkpoint runs/example/latest.pt --budget-seconds 180
```

Select a destination and click **Start drive**. Use a compatible single-case
configuration when viewing a named scenario.

## Architecture: how it works

```text
RGB + aligned depth + four radars + ego state + assigned route
  -> synchronized sensor bundle and validity masks
  -> ground-truth / external / mixed perception
  -> image encoder + navigation, object and radar tokens
  -> masked single-query attention (baseline: masked mean pooling)
  -> recurrent state-space model (RSSM)
  -> imagined trajectories -> actor and critic
  -> requested action -> safety supervisor -> CARLA
                           executed action + next observation -> replay
```

TorchRL supplies the DreamerV3 image encoder/decoder, categorical RSSM,
reconstruction/reward/continuation heads and model, actor and value objectives.
The RSSM combines recurrent memory with a stochastic latent state. It learns
from replay sequences; imagined latent rollouts train the continuous-action
actor and distributional critic. The project adds structured sensor tokens
and attention so fusion can weight relevant observations. Mean pooling provides
a matched baseline. Attention weights are diagnostics, not proof of causality.

The safety supervisor checks signals, obstacles, pedestrians, crossings and
speed constraints before controls reach CARLA. Replay stores the executed
controls, so learning reflects the actions actually taken. This is a simulator
research safeguard, not a demonstrated real-world safety guarantee.

External vision combines YOLO detections, the joint road model's lane masks
and crossing boxes, lane-centre geometry, aligned depth and speed-sign OCR.
Ground-truth perception instead uses simulator annotations and must be labeled
as such in results. For external vision restore these weights before running:

- `perception/carla_yolo26_demo_results/mac_demo/best.pt`
- `joint_road_model/joint_road_model.pt`

Keep `joint_road_model/lane_center_calibration.json` with the road model.
Weights, sample frames and expected outputs must be present for standalone
perception demos. With the root environment activated, use
`python perception/demo.py --device cpu` (http://127.0.0.1:8773), or run
`../venv/bin/python run_video.py sample_frames/episode_0014 --out demo_out`
from `joint_road_model/`. Older package setup notes refer to removed dependency
files; install from the root `requirements.txt` instead.

# CARLA Self Driving Car

Can a car drive a town it has never seen, using only what our own vision models
detect? Most CARLA agents read exact positions and light states from the
simulator. This agent replaces as much of that ground truth as it can with its
own perception: seven-class object and traffic-light detection, lane and
crossing segmentation and speed-sign reading feed a DreamerV3-style world-model
policy with attention, and a rule-based safety supervisor checks every control
before it reaches the car. Trained on Town01 only, the hybrid system completed
22 of 32 evaluation runs on unseen Town02 (see [Results](#results)).

AAI3001 Computer Vision and Deep Learning term project, Singapore Institute of
Technology, AY2026/27 Trimester 1.

**Team P1-21**

| Member | Contribution |
| --- | --- |
| Kew Zhi Hui Mark | Lane detection (joint road model) and cloud deployment |
| Kevin Koh | Reinforcement learning agent and CARLA integration |
| Guo YouYan | Dataset annotation and integration |
| Lai Zhi Nuo | Seven-class detection model and its integration into the agent |
| Darius Tan | Speed-limit number recognition and the live depth tool (distance, angle and speed per detection) |

> **Markers: start here.** This repository is a fork of the CARLA simulator, so
> most folders are CARLA's own source code. The project's work is in four
> places: `perception/`, `joint_road_model/`, `dreamerv3_attention/` and `data/`
> (see [Repository map](#repository-map)).

## Choose how to try it

| Route | What you get | Needs | Time |
| --- | --- | --- | --- |
| [1. See it working](#1-see-it-working) | Recorded drives of the agent | A browser | 2 minutes |
| [2. Run the vision models locally](#2-run-the-vision-models-locally) | Detector and road model on your own machine | Python 3.10 to 3.12, any OS | 10 minutes |
| [3. Run the full system](#3-run-the-full-system) | CARLA, the driving agent and the live viewer | Linux, NVIDIA GPU with 8 GB or more | About 2 hours, mostly downloads |
| [4. Compare runs](#4-compare-runs) | Dashboard of driving score, safety and reward metrics across evaluation and training runs | Python and the run folders (evaluation traces or training metrics) | 5 minutes |

### 1. See it working

- **Recorded drives:** `videos/` in the project OneDrive folder
  ([CVDL CARLA SUBMISSION on OneDrive](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2503687_sit_singaporetech_edu_sg/IgCXTbdPMD0IT6NwPhhbny2KAbLdVIvAttJ4U6J2-PyWYW8)): `video_demo_final.webm` (final agent)
  and `video_demo_prototype.webm` (earlier prototype).
- **Perception demo:** runs on your own machine in about 10 minutes, route 2
  below. Pick a held-out CARLA frame or upload an image to see detections,
  lanes, crossings and the speed limit read.
- **Live system:** the full CARLA system and its browser viewer run on a rented
  cloud GPU VM that is switched on for demo sessions only (see
  [Deployment](#deployment)).

### 2. Run the vision models locally

Works on macOS, Linux and Windows. It does not need CARLA.

```bash
git clone --depth 1 https://github.com/YoinkOURcode/carla_selfdrivingcar.git
cd carla_selfdrivingcar
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements-perception.txt
python perception/demo.py --device cpu
```

Open http://127.0.0.1:8773. Use `--device mps` on Apple silicon or
`--device cuda` on an NVIDIA GPU, and `--port 8780` if port 8773 is taken. The
model weights come with the clone.

| Page | Address | What it shows |
| --- | --- | --- |
| Detector | `/` | Boxes for the seven classes on 50 held-out test frames or an uploaded image, with counts against the hand labels and the test results |
| Road model | `/road` | Lane paint, pedestrian crossings and lane centre on 40 reserved test frames |
| Speed signs | `/signs` | Each detected sign, the cut-out sent to OCR and the value read, on 30 held-out test frames |

Road model demo (lane paint, crossings and lane centre on a 20-frame held-out drive):

```bash
cd joint_road_model
python selftest.py
python run_video.py sample_frames/episode_0014 --out demo_out
```

`selftest.py` prints `RESULT: PASS` when the model reproduces the saved outputs.
`demo_out/overlay.mp4` shows lane paint in green, crossings in magenta and the
lane centre in red.

### 3. Run the full system

Follow [SETUP_LINUX.md](SETUP_LINUX.md). It covers a cloud VM and a local Linux
PC from a fresh Ubuntu 22.04 install: the NVIDIA driver, downloading CARLA
0.9.16, the Python environment, the trained model from OneDrive, starting the
simulator, running the agent and watching it in the browser.

The trained driving model (6.9 GB), its configuration, the evaluation reports,
the full annotated dataset and the demo videos are not in GitHub because of their
size. They are in the OneDrive folder `CVDL CARLA SUBMISSION`:
[CVDL CARLA SUBMISSION on OneDrive](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2503687_sit_singaporetech_edu_sg/IgCXTbdPMD0IT6NwPhhbny2KAbLdVIvAttJ4U6J2-PyWYW8).

| OneDrive folder | Contents |
| --- | --- |
| `models/rl/` | Trained driving model `latest.pt`, its `config.json`, training metrics and result |
| `models/perception/` | Detector weights (same file as in this repository), metadata, training curves |
| `evaluation/` | Held-out Town02 evaluation configuration `town02_final.json`, `result.json`, step trace |
| `dataset/` | Round 1 (3,580) and Round 2 (9,060) annotated frames with labels, splits and `data.yaml` |
| `videos/` | Recorded drives of the final and prototype agents |

SETUP_LINUX.md section 6 lists which file goes where on the machine.

### 4. Compare runs

```bash
python -m dreamerv3_attention.dashboard                 # every run in runs/experiment_28/evaluation
python -m dreamerv3_attention.dashboard runs/experiment_28/evaluation/eval24 runs/A/metrics.jsonl
```

This writes one small summary per run to `runs/dashboard/`. A run is an
evaluation directory (`trace.jsonl`, plus `result.json` once finished) or a
training `metrics.jsonl`. Open `dreamerv3_attention/dashboard.html` in a
browser and add up to 8 summaries. It shows CARLA Leaderboard-style driving
score, a per-scenario table, reward components, safety supervisor
interventions, TTC/TET (`--ttc-threshold`, default 1.5 s), RSS safe distance and
the evaluation report from `result.json` (speeding, unnecessary stops, comfort,
latency), with filters and a two-metric explorer. To add a metric, add an
`@metric` or `@breakdown` function in `dashboard.py`; the page picks it up
automatically.

## How it works

```text
Front camera (RGB) + aligned depth camera + ego speed + planned route
  -> perception
       YOLO26n detector: vehicles, red / yellow / green lights, adult and child
         pedestrians, speed-limit signs (nested light boxes merged)
       Joint road model: lane paint, crossings, lane-centre line
       RapidOCR: speed-limit value
  -> signal tracker: merges duplicate light boxes, holds a light's state for 2 s
  -> agent inputs: 16 driving values plus tokens for lights, objects and route
       (no raw pixels reach the policy)
  -> DreamerV3-style world model with single-query attention over the tokens
  -> actor: throttle, brake, steering
  -> safety supervisor: rules check every action and adjust throttle or brake
  -> CARLA Town02 (unseen town)

Still from CARLA, not from our models: object distance (depth camera), stop
lines, the planned route, ego speed and pose, weather.
```

| Part | What | Why this choice | Limits |
| --- | --- | --- | --- |
| Dataset | Own CARLA capture, annotated by the team in two rounds: 3,280 frames (Round 1, 4 classes) and 9,060 frames (Round 2, 7 classes), 12,340 in total, split by whole episode into train (8,258), validation (1,537) and test (2,545, with 20 whole Round 2 episodes held out) | The agent needs child and adult pedestrians told apart, alongside CARLA's own lights and signs | Two annotators boxed lights differently; one simulator town style |
| Detector | Ultralytics YOLO26n fine-tuned at 1280 px on the team's labels | Small enough for real-time use beside the simulator; small lights need the high resolution | Small, distant lights are missed in single frames (25% of red lights on the held-out test); the test sets were also used to score earlier models |
| Road model | One network with two heads on a 1280 x 720 frame: a YOLO11s crossing detector and a lane-paint decoder on its shared backbone features (9.56 M parameters, 133 k lane-specific); the lane centre is computed from the lane mask | One shared backbone gives lanes and crossings for about the cost of one model | Lane paint is absent inside junctions, so the planned route takes over there; straight boundary fit, so sharp curves are approximate |
| Driving agent | TorchRL DreamerV3 with attention over observation tokens, trained in Town01 only (about 4.05 M environment steps) and evaluated on held-out Town02 | A world model learns from replay and imagined rollouts, which DreamerV3 is designed to make data-efficient; slow simulator steps make that matter | Trained mostly on ground-truth inputs (the team's estimate: about 44 h, against at most 11 h on vision inputs), so it is less stable at speed on its own perception |
| Safety supervisor | Rule checks on every control before it reaches CARLA | Keeps the car safe while the policy learns; its interventions are reported separately | A simulator safeguard, not a real-world guarantee |

## Results

### Driving (unseen Town02, camera perception, 32 runs)

The final model was re-evaluated after patch `47c77c22d`, which removed a
ground-truth lane override that had kept the car on the road. The checkpoint is
unchanged. The suite is 16 Town02 scenarios, each run twice: scenarios 1 to 6
are the original suite (`deployment/town02_scenarios.json`) and 7 to 16 were
added for this evaluation. Town02 was never used in training.

| Measure | Value |
| --- | --- |
| Runs completed | 22 of 32 (69%) |
| Driving score (route completion x infraction penalty), run 1 / run 2 | 85.1 / 81.3 |
| Mean route completion | 88% |
| Steps adjusted by the safety supervisor | 77% |
| Collisions | 6 (4 with static objects, 1 with a vehicle, 1 with a pedestrian) |
| Red-light violations | 1 |
| Minimum time to collision (TTC), run 1 / run 2 | 1.97 s / 1.86 s |
| Time with TTC under 1.5 s (TET), run 1 / run 2 | 0.43 / 0.34 s per km |
| Steps closer than the RSS safe distance, run 1 / run 2 | 2.5% / 3.0% |
| Policy step time (p95) on the cloud VM, run 1 / run 2 | 72 / 80 ms |
| Lane position taken from the camera road model | 60% of steps |

| Scenario | Weather | Route | Run 1 | Run 2 |
| --- | --- | --- | --- | --- |
| 1 | Clear noon | 425 m | Fail at 60% (off-road) | Fail at 59% (collision) |
| 2 | Wet sunset | 503 m | Pass | Pass |
| 3 | Soft rain noon | 482 m | Pass | Pass |
| 4 | Cloudy noon | 463 m | Pass | Pass |
| 5 | Clear night | 485 m | Pass | Fail at 44% (pedestrian collision) |
| 6 | Hard rain night | 556 m | Pass | Pass |
| 7 | Clear sunset | 416 m | Fail at 66% (stopped too long) | Fail at 89% (collision) |
| 8 | Cloudy noon | 531 m | Fail at 86% (collision) | Fail at 84% (collision) |
| 9 | Wet noon | 425 m | Pass | Pass |
| 10 | Mid rain noon | 515 m | Pass | Pass |
| 11 | Cloudy sunset | 565 m | Pass | Pass |
| 12 | Soft rain sunset | 647 m | Pass | Fail at 46% (red-light violation) |
| 13 | Wet cloudy noon | 533 m | Fail at 38% (stopped too long) | Pass |
| 14 | Clear night | 476 m | Fail at 64% (collision) | Pass |
| 15 | Wet night | 424 m | Pass | Pass |
| 16 | Mid rain night | 404 m | Pass | Pass |

Each route has signalised junctions and scripted adult, child and slow-vehicle
encounters. Nine scenarios passed both runs and three failed both (1, 7 and 8),
so failures follow specific routes and encounters, not chance. The first six
scenarios pass 9 of 12 runs; the 6 of 6 reported before the patch no longer
stands. The safety supervisor adjusted 77% of steps, mostly by limiting throttle
and braking, so these results describe the learned policy and the rules
together, not the policy alone. For the other 40% of steps the lane position
comes from the planned route and fallbacks such as held estimates, mainly near
junctions, which have no lane paint.

### Detection (held-out test, scored once on 22 November 2026)

AP50 on the Round 2 test split: 1,884 frames from 20 unseen episodes, before
merging duplicate boxes. "Corrected" scores the same model against test labels
brought to one box convention for lights.

| Class | AP50 | AP50, corrected light labels |
| --- | --- | --- |
| Vehicle | 0.83 | 0.83 |
| Adult pedestrian | 0.76 | 0.76 |
| Child pedestrian | 0.56 | 0.56 |
| Speed limit | 0.52 | 0.52 |
| Red light | 0.42 | 0.53 |
| Yellow light | 0.40 | 0.50 |
| Green light | 0.45 | 0.55 |
| **mAP50** | **0.562** | **0.607** |

At the agent's settings (confidence 0.25, nested boxes merged) the detector
finds 75% of red, 84% of yellow and 79% of green lights in single frames, and
99.4% of the lights it finds get the right colour. Finding small lights is the
limit, not colour. Per-frame misses rarely became driving misses: one of the 32
runs ran a red light. Earlier models were also scored on these test episodes
in October, so this is not the first look at them. Full tables, including the
Round 1 test and the validation scores, are in
[perception/README.md](perception/README.md).

### What the team's own work improved

| Change | Measure | Before | After |
| --- | --- | --- | --- |
| Fine-tuning pretrained YOLO26n on the team's labels | Traffic-light AP50, Round 1 test | 0.45 | 0.78 |
| Retraining on Round 1 and Round 2 together | Yellow-light AP50, Round 2 test | 0.01 | 0.32 |
| Correcting light labels to one box convention | Mean AP50 of 7 classes, Round 2 validation | 0.59 | 0.64 |
| Merging nested light boxes | Nested pairs in 500 held-out frames | 558 | 0 |

### Road model (reserved test set, 200 images from 10 held-out episodes, evaluated once)

| Task | Metric | Test value [95% CI] |
| --- | --- | --- |
| Lane paint | F1 with 2 px tolerance | 0.895 [0.872, 0.926] |
| Crossings | AP50 | 0.784 [0.662, 0.899] |
| Lane centre | Offset error within 0.10 lane widths | 93% of frames (98% on high-confidence frames) |
| Lane centre | Median offset error / median heading error | 0.003 lane widths / 0.6 degrees |

Confidence intervals come from resampling whole episodes. The figures measure
agreement with the team's labels on held-out episodes of the same capture; they
are not a blind or real-world test.

### Duplicate traffic-light boxes

The detector often returned two nested boxes on one light because annotators
boxed either the whole housing or only the lit lamp. Merging nested light boxes
before the agent sees them removed all 558 nested pairs on 500 held-out test
frames (1,994 to 1,458 light boxes) with no labelled light lost (1,310 found
before and after).

## Deployment

| Where | What runs | Access |
| --- | --- | --- |
| Cloud | Full CARLA system, the agent and a live browser viewer on a rented GPU VM (RTX 4080 SUPER); policy step p95 72 to 80 ms | Switched on for demo sessions; team access through SSH (SETUP_LINUX.md section 12) |
| Local | Perception demo on any laptop (route 2); full system on a Linux GPU PC (route 3) | Routes 2 and 3 above |

Detector timing, measured on one machine (Apple M5 CPU, 1280 px, median of 30
runs over 10 CARLA frames): PyTorch 64 ms and ONNX Runtime 65 ms per frame,
with identical detections on all 10 frames. The ONNX export gives portability,
not speed, on this CPU; the repository ships the PyTorch weights. The joint
road model takes about 190 ms on CPU and 50 ms on the Apple GPU per frame (its
own README).

## Limitations

- Object distance (depth camera), stop lines, the planned route, ego speed and
  pose, and weather still come from CARLA. At traffic lights only the light
  colour comes from the camera; the stop position comes from the map, as a
  satnav knows where junctions are.
- The driving results are for the learned policy and the rule-based supervisor
  together; the supervisor adjusted 77% of steps.
- With little training on vision inputs, the policy is less stable at speed:
  6 collisions in 32 runs.
- Small, distant lights are missed in single frames. A typical miss is a red
  light 11 px tall that scores below the 0.25 confidence cut-off; lights like
  this are usually found once the car is closer.
- The agent can brake, accelerate and steer only: no reversing or overtaking.
- The detector was trained only on CARLA images. Real dashcam images are a
  possible extension, but accuracy on them has not been measured.

## Future work

- **Train on vision from the start,** so the policy learns on the inputs it
  drives with.
- **Stop lines from the vehicles ahead** and from paint or learned junction
  detection, instead of map data.
- **Real dashcam tests:** the same detector applied to real images after
  fine-tuning on a labelled real-world set.
- **Labels:** one labelling rule for lights and pedestrian age, then a retrain;
  the duplicate boxes and missed yellow lights trace back to inconsistent labels.
- **More scenes:** more towns, weathers and night footage, with a scored
  held-out test for each.
- **Weather from the image** instead of from the simulator.
- **Serving:** wrap the detector in a stateless inference service, so more users
  are served by running more copies behind a load balancer.

## Repository map

| Path | Owner | Contents |
| --- | --- | --- |
| `perception/` | This project | Seven-class detector, weights, browser demo, duplicate-box merging |
| `joint_road_model/` | This project | Lane, crossing and lane-centre model, weights, demo |
| `dreamerv3_attention/` | This project | Sensors, perception provider (`vision.py`), agent, training, evaluation, viewer, tests |
| `data/` | This project | Dataset capture and server checks |
| `SETUP_LINUX.md` | This project | Full install guide |
| `deployment/` | This project | Held-out Town02 scenario file used by the deployment (SETUP_LINUX.md section 6) |
| `VISION_INTEGRATION.md` | This project | How camera perception feeds the agent, and known gaps |
| `PythonAPI/`, `LibCarla/`, `Unreal/`, `Util/`, `Docs/`, others | CARLA upstream | Simulator source, unchanged except where noted in `VISION_INTEGRATION.md` |

## References and licences

- CARLA simulator 0.9.16, MIT licence (code) and CC-BY (assets): https://carla.org
- TorchRL DreamerV3 implementation: https://github.com/pytorch/rl
- Ultralytics YOLO, AGPL-3.0: https://github.com/ultralytics/ultralytics
- RapidOCR: https://github.com/RapidAI/RapidOCR

Papers:

- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A. and Koltun, V. (2017).
  CARLA: An Open Urban Driving Simulator. Conference on Robot Learning (CoRL).
- Hafner, D., Pasukonis, J., Ba, J. and Lillicrap, T. (2025). Mastering diverse
  control tasks through world models. Nature, 640, 647-653 (DreamerV3).
- Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016). You Only Look
  Once: Unified, Real-Time Object Detection. CVPR.
- Jocher, G., Qiu, J. et al. (2026). Ultralytics YOLO26: Unified Real-Time
  End-to-End Vision Models. arXiv:2606.03748.
- Shalev-Shwartz, S., Shammah, S. and Shashua, A. (2017). On a Formal Model of
  Safe and Scalable Self-driving Cars. arXiv:1708.06374 (RSS safe distance).

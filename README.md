# CARLA Self Driving Car

A self-driving agent in the CARLA simulator that drives from its own camera.
Seven-class object and traffic-light detection, lane and crossing segmentation,
depth and speed-sign reading feed a DreamerV3 world-model agent with
attention. A rule-based safety supervisor checks every control before it reaches
the car.

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
| [1. See it working](#1-see-it-working) | Live perception demo in the browser and recorded drives | A browser | 2 minutes |
| [2. Run the vision models locally](#2-run-the-vision-models-locally) | Detector and road model on your own machine | Python 3.10 to 3.12, any OS | 10 minutes |
| [3. Run the full system](#3-run-the-full-system) | CARLA, the driving agent and the live viewer | Linux, NVIDIA GPU with 8 GB or more | About 2 hours, mostly downloads |

### 1. See it working

- **Perception demo:** [PLACEHOLDER: public demo URL]. Upload a driving image or pick a
  sample CARLA frame to see detections, lanes, crossings and the speed limit read.
- **Recorded drives:** `videos/` in the project OneDrive folder
  ([CVDL CARLA SUBMISSION on OneDrive](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2503687_sit_singaporetech_edu_sg/IgCXTbdPMD0IT6NwPhhbny2KAbLdVIvAttJ4U6J2-PyWYW8)): `video_demo_final.webm` (final agent)
  and `video_demo_prototype.webm` (earlier prototype).
- **Poster:** [PLACEHOLDER: poster PDF link].

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
`--device cuda` on an NVIDIA GPU. The model weights come with the clone.

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

## How it works

```text
Front camera (RGB) + aligned depth camera + ego speed + planned route
  -> perception
       YOLO26n detector: vehicles, red / yellow / green lights, adult and child
         pedestrians, speed-limit signs (nested light boxes merged)
       Joint road model: lane paint, crossings, lane-centre line
       Depth camera: object and lamp distances
       RapidOCR: speed-limit value
  -> observation tokens (image, ego state, route, objects, signals)
  -> DreamerV3 world model with single-query attention over the tokens
  -> actor: throttle, brake, steering
  -> safety supervisor: signals, pedestrians, crossings, speed limits
  -> CARLA
```

| Part | What | Why this choice | Limits |
| --- | --- | --- | --- |
| Dataset | Own CARLA capture, annotated by the team in two rounds: 3,280 frames (Round 1, 4 classes) and 9,060 frames (Round 2, 7 classes), split by whole episode into train, validation and test | The agent needs child and adult pedestrians told apart, alongside CARLA's own lights and signs; [PLACEHOLDER: datasets checked and why they did not fit] | Two annotators boxed lights differently; one simulator town style |
| Detector | Ultralytics YOLO26n fine-tuned at 1280 px | Small enough for real-time use beside the simulator; small lights need the high resolution | Yellow lights often missed; no clean test score yet |
| Road model | One network with two heads on a 1280 x 720 frame: a YOLO11s crossing detector and a lane-paint decoder on its shared backbone features (9.56 M parameters, 133 k lane-specific); the lane centre is computed from the lane mask | One shared backbone gives lanes and crossings for about the cost of one model | Lane paint is absent inside junctions, so the planned route takes over there; straight boundary fit, so sharp curves are approximate |
| Driving agent | TorchRL DreamerV3 with attention over observation tokens, trained in Town01 and evaluated on held-out Town02 | A world model learns from replay and imagined rollouts, which DreamerV3 is designed to make data-efficient; slow simulator steps make that matter | Stable at low speed; unstable at high speed |
| Safety supervisor | Rule checks on every control before it reaches CARLA | Keeps the car safe while the policy learns; its interventions are reported separately | A simulator safeguard, not a real-world guarantee |

## Results

### Driving (held-out Town02, camera perception)

Evaluation suite: held-out Town02 (`evaluation/town02_final.json` in the OneDrive
folder), never used in training. The final model passed 6 of 6 scenarios. Full
reports: `evaluation/` in the OneDrive folder.

| Scenario | Route completion | Collisions | Red-light violations | Pedestrian incidents | Lane departures | Supervisor interventions |
| --- | --- | --- | --- | --- | --- | --- |
| [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

[PLACEHOLDER: one row per scenario from the final result.json files]

### Detection (Round 2 validation, AP50, before merging duplicate boxes)

| Labels | Red | Yellow | Green | Mean of 7 classes |
| --- | --- | --- | --- | --- |
| Original | 0.572 | 0.368 | 0.508 | 0.591 |
| Corrected box sizes | 0.674 | 0.456 | 0.687 | 0.643 |

Validation also selected the checkpoint, so these figures are optimistic. The
held-out test split has not been scored yet. [PLACEHOLDER: held-out test
score, if run before submission].

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
| Cloud, public | Perception API with the optimised models (see the timing note below), plus the demo page | [PLACEHOLDER: public demo URL] |
| Cloud, on demand | Full CARLA system on a GPU VM, started before demos | Team only, through SSH (SETUP_LINUX.md section 12) |
| Local | Full system on a Linux GPU PC; perception demo on any OS | Routes 2 and 3 above |

Detector export to ONNX, measured on one machine (Apple M5 CPU, 1280 px, median
of 30 runs over 10 CARLA frames): PyTorch 64 ms, ONNX Runtime 65 ms per frame,
with identical detections on all 10 frames. The ONNX file alone gives
portability, not speed, on this CPU. Optimised serving setup actually
deployed: [PLACEHOLDER: method, for example INT8 quantisation or TensorRT, and
its time per frame against PyTorch on the same machine]; the joint road model takes about 190 ms on CPU and
50 ms on the Apple GPU per frame (its own README).

## Limitations

- The stop position at traffic lights comes from the map's stop lines, as a
  satnav knows where junctions are; only the light colour comes from the camera.
- Weather is read from the simulator, not detected from the image.
- The agent can brake, accelerate and steer only: no reversing or overtaking.
- Behaviour at high speed is unstable.
- The detector was trained only on CARLA images. Real dashcam images are a
  possible extension, but accuracy on them has not been measured.

## Future work

- **Labels:** one labelling rule for lights and pedestrian age, then a retrain;
  the duplicate boxes and missed yellow lights trace back to inconsistent labels.
- **More scenes:** more towns, weathers and night footage, with a scored
  held-out test for each.
- **Perception the agent still borrows from the simulator:** weather detection
  from the image, and stop lines from paint or learned junction detection instead
  of map data.
- **Real images:** the same detector and API applied to real dashcam images
  after fine-tuning on a labelled real-world set.
- **Serving:** the perception API keeps no session state, so more users are served
  by running more copies behind a load balancer.

## Repository map

| Path | Owner | Contents |
| --- | --- | --- |
| `perception/` | This project | Seven-class detector, weights, browser demo, duplicate-box merging |
| `joint_road_model/` | This project | Lane, crossing and lane-centre model, weights, demo |
| `dreamerv3_attention/` | This project | Sensors, perception provider (`vision.py`), agent, training, evaluation, viewer, tests |
| `data/` | This project | Dataset capture and server checks |
| `SETUP_LINUX.md` | This project | Full install guide |
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

[PLACEHOLDER: any other papers cited on the poster]

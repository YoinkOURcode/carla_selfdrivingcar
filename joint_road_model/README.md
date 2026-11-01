# Joint road model — FINAL package (v2b), teammate share edition

This is the final checkpoint. It was chosen on validation by a rule fixed in advance, and was then evaluated **once** on the reserved test set
(200 images, 10 episodes that were never used for training or model selection).

One network does three jobs on a single 1280×720 front-camera frame (CARLA, camera on the car's centreline):

1. **Lane paint**: a per-pixel mask of painted lane markings.
2. **Pedestrian crossings**: bounding boxes. The class ID is 7, matching the team's label scheme.
3. **Ego-lane centre**: where the centre of the car's lane is, and how far the car is from it. This is image-space geometry computed from the lane mask.

**Start with `WORK_SUMMARY.md`.** It explains what was done in plain language and in technical detail.

## Test it (about 2 minutes)
```bash
pip install -r requirements.txt          # Python 3.10–3.12 recommended

python selftest.py                       # 1) verifies the package works on your machine
python run_video.py sample_frames/episode_0014 --out demo_out   # 2) runs on a 20-frame drive and makes a video
```
1. **`selftest.py`** checks four things:
   - Every file matches `SHA256SUMS`.
   - The model loads.
   - Outputs on the 6 frames in `sample_frames/` match the expected outputs saved in `sample_frames/expected/`. It compares the lane mask, crossing boxes and lane centre, with small tolerances for different hardware and library versions.
   - It reports how fast inference runs.

   It prints `RESULT: PASS` or `FAIL` and writes overlay images to `selftest_out/`.
   Add `--device cuda` (NVIDIA) or `--device mps` (Apple) to test on a GPU.
2. **`run_video.py`** writes two files:
   - `demo_out/overlay.mp4`: green is lane paint, magenta boxes are crossings, cyan lines are the car's lane edges and the red line is the lane centre.
   - `demo_out/frames.csv`: per-frame lane-centre offset, heading, confidence and crossing count.

   It also accepts a video file (`python run_video.py drive.mp4 --out demo_out`) or any folder of CARLA frames, for example a `Mark/episode_XXXX` folder.
   The sample episode, `episode_0014`, comes from the reserved test split, so the model never trained on it.

Measured speed (MacBook Air M4) on one 1280×720 frame, including pre/post-processing and lane centre: **~190 ms on CPU, ~50 ms on Apple GPU (MPS)**. An NVIDIA GPU should be faster.

## Quick start (your own frames)
```bash
python road_perception.py --checkpoint joint_road_model.pt --calibration lane_center_calibration.json --out out_dir frame1.png frame2.png
```
```python
import cv2
from road_perception import RoadPerception
rp = RoadPerception('joint_road_model.pt', calibration='lane_center_calibration.json', device='cpu')  # 'cuda' / 'mps' also work
frame = cv2.imread('frame.png')            # BGR, as from OpenCV or a CARLA camera converted to BGR
out = rp(frame)
out['lane_center']['offset_norm']          # lane deviation, see below
out['lane_center']['heading_deg']
out['crosswalks']                          # [{'class':'crosswalk','class_id':7,'confidence':..,'xyxy':[x1,y1,x2,y2]}]  (pixels)
out['lane_mask']                           # HxW bool
vis = rp.draw(frame, out)                  # overlay for demos
```

## Outputs

| key | meaning |
|---|---|
| `lane_mask` | Boolean H×W mask; True means lane paint. `lane_prob` holds the probabilities, and the threshold is 0.5. |
| `crosswalks` | Boxes after NMS with confidence ≥ 0.25, in input-image pixels. |
| `lane_center.mode` | One of `both`, `left_only`, `right_only` or `none`: which ego-lane boundaries were found as paint. With one side found, the centre is inferred from the calibrated lane width. |
| `lane_center.offset_norm` | The car's lateral offset from the lane centre, in lane widths, measured at the bottom image row. 0 means centred. +0.5 means the car is on the right boundary and −0.5 on the left. `None` if no lane was found. |
| `lane_center.offset_px` | The same offset in pixels at the bottom row. |
| `lane_center.heading_deg` | Angle of the lane centre line from image vertical. Positive means the lane bends or leans to the right ahead. This is image-space only, not a calibrated yaw. |
| `lane_center.center_line` | A list of `[x, y]` points along the lane centre, in image pixels. |
| `lane_center.low_confidence` | True when a boundary was extrapolated over more than half the road area, for example at intersections or when a vehicle blocks the view. Treat these frames with caution. |

**For the RL state:** `offset_norm` is a natural "lane deviation" signal. When `low_confidence` is True or `mode` is `none`, hold the last value or fall back to the CARLA waypoint.

## Reserved test results (200 images, 10 held-out episodes, evaluated once)
These figures measure agreement with the team's supplied labels. 95% CIs come from resampling whole episodes.

| task | metric | test value [95% CI] | target |
|---|---|---|---|
| Lane paint | F1 with 2 px tolerance (640 raster) | **0.895** [0.872, 0.926] | ≥ 0.80 ✅ |
| Lane paint | strict pixel IoU (640 raster) | 0.525 [0.470, 0.598] | (reported only) |
| Crossings | AP50 | **0.784** [0.662, 0.899] | ≥ 0.75 ✅ |
| Crossings | precision at conf 0.25 | 0.778 [0.611, 0.886] | ≥ 0.85 ❌ |
| Crossings | recall at conf 0.25 | 0.787 [0.694, 0.882] | ≥ 0.80 ❌ (within CI) |
| Lane centre | offset error within 0.10 lane widths | 93% of frames (98% on high-confidence frames) | – |
| Lane centre | median offset error / median heading error | 0.003 lane widths / 0.6° | – |

Notes on the test set:
- It has only 89 crossing boxes, so the crossing CIs are wide.
- Its lane labels were not geometrically re-aligned (training and validation labels were), which lowers strict IoU.

## Validation results (300 images, 15 held-out episodes; also used for model selection, so optimistic)
These figures measure agreement with the team's supplied labels. 95% CIs come from resampling whole episodes.

| task | metric | value [95% CI] |
|---|---|---|
| Lane paint | F1 with 2 px tolerance (640 raster) | 0.920 [0.906, 0.936] |
| Lane paint | strict pixel IoU (640 raster) | 0.579 [0.540, 0.624] |
| Crossings | AP50 | 0.775 [0.677, 0.863] |
| Crossings | precision / recall at conf 0.25 | 0.836 / 0.672 |
| Lane centre | offset error within 0.10 lane widths | 91% of frames (97% on high-confidence frames) |
| Lane centre | median offset error | 0.003 lane widths |

**About strict IoU:** on lines about 6 px wide, a label compared with itself shifted by 1 px already scores only 0.62. That is why the tolerance-based F1 is the primary lane metric.

**About the lane-centre reference:** the episodes contain no ground-truth vehicle pose. The lane-centre reference is therefore the same estimator run on the label masks, and this figure measures consistency with the labels, not true lane-keeping accuracy.

## Known limitations
- **Distant crossings are often missed.** Missed crossings have a median height of about 20 px in the 960×544 input. Crossing recall is the weakest metric.
- **Lane centre:**
  - It assumes the camera sits on the car's centreline.
  - The lane-width calibration is specific to this camera's mount and field of view. Re-run `lane_center.calibrate()` on label masks if the camera changes.
  - It fits straight boundary lines, so sharp curves are approximate.
- **Held-out episodes, not a blind test.** All results come from held-out episodes of the same project data, and route overlap with training is unknown. This is not a blind or deployment-safety evaluation.
- **Load only trusted checkpoints.** `torch.load(weights_only=False)` is used.

## Files
| file | what it is |
|---|---|
| `WORK_SUMMARY.md` | What was done, why, results and limitations, in plain language and in technical detail |
| `joint_road_model.pt` | The trained model: one checkpoint, 9.56 M parameters, 38 MB |
| `road_perception.py` | One-call inference: lane mask, crossings and lane centre. Also a command-line tool |
| `lane_center.py` / `lane_center_calibration.json` | Lane-centre geometry and its lane-width calibration |
| `model.py` | Network definition, needed to load the checkpoint |
| `selftest.py` | Package self-test (see above) |
| `run_video.py` | Video/folder demo with overlay video and per-frame CSV |
| `sample_frames/` | 6 frames (3 test-split, 3 validation-split) plus expected outputs for the self-test, and one 20-frame test episode (`episode_0014`) for the demo. JPEG copies of project frames |
| `examples/` | Overlays and JSON produced by this package |
| `metrics/` | Full validation and test metric files, including per-image records and bootstrap CIs |
| `requirements.txt` | Validated with Python 3.12, torch 2.14, ultralytics 8.4.171/8.4.175, numpy 1.26, opencv 4.11 |
| `SHA256SUMS` | Checksums of every file |

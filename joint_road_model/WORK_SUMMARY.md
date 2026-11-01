# Joint Road Model: work summary

*AAI3001 CVDL project, "Mark" CARLA driving data. Final model: **v2b**. Package date: 31 Oct 2026.*

This document has three parts:
- **Part A** explains everything in plain language.
- **Part B** gives the technical detail: data, architecture, training, experiments, metrics and evaluation protocol.
- **Part C** covers how to use and test the package, its limits, and what to do next.

A glossary is at the end.

---

# Part A: the plain-language version

## What it does
You give the model one picture from the car's front camera. It returns three things:

1. **Where the lane lines are.** It marks every pixel of painted road line (white and yellow lines, dashed and solid).
2. **Where the pedestrian crossings are.** It draws a box around each zebra crossing or crossing marked with paired bars, with a confidence score.
3. **Where the middle of our lane is, and how far off-centre the car is.** Using the lane lines from step 1, it finds the line on the car's left and the line on its right and takes the midpoint. It reports the offset as a fraction of the lane width:
   - **0** means perfectly centred.
   - **+0.5** means the car is sitting on the right-hand line.
   - **−0.5** means it is on the left-hand line.

   It also gives the direction the lane is heading, and says when it is unsure, for example at an intersection with no lines.

**There is one trained model.** Items 1 and 2 come out of the same neural network in a single pass, so it is not two models stuck together. Item 3 is not learned. It is ordinary geometry applied to item 1's output.

## How it was built (in brief)
- **Data:** 1,999 frames from 100 short CARLA driving recordings ("episodes", 20 frames each). Every frame has a hand-checked lane-line mask and crossing boxes.
- **Fair testing:** the episodes were split once, before any training:
  - 75 episodes (1,499 images) to **train** on,
  - 15 episodes (300 images) to **choose settings**,
  - 10 episodes (200 images) locked away as the **final exam** ("reserved test").

  Whole episodes stay together, so the model is never tested on a near-copy of a frame it trained on. The final exam was opened **exactly once**, after the model was chosen.
- **Model:** a well-known, fast object detector (YOLO11s, pre-trained on everyday photos) with an extra small "lane painter" attached. Both share the same image understanding.
- **Training:** about 3 hours on a MacBook Air M4 GPU.

## How good is it? (final exam: 200 unseen images)
| What | Score | Meaning | Target |
|---|---|---|---|
| Lane lines | **0.895** F1 (0–1, higher is better) | Almost all painted lines are found, with few false ones, counting a line as correct if it lands within 2 pixels | ≥ 0.80 ✅ |
| Crossings: overall quality (AP50) | **0.784** | Good overall detection quality | ≥ 0.75 ✅ |
| Crossings: precision | 0.778 | About 78% of the boxes it draws are real crossings | ≥ 0.85 ❌ |
| Crossings: recall | 0.787 | It finds about 79% of the real crossings; mainly distant, small ones are missed | ≥ 0.80 ❌ (just short) |
| Lane centre | **93%** of frames | The estimated centre is within 10% of a lane width of the centre worked out from the hand-drawn labels. When the model says it is confident, this rises to 98% | – |

**Bottom line:** lane lines and lane centre work well. Crossing detection is decent and reliable for nearby crossings, but still misses some far-away ones and has a few false alarms. Note that the final exam has only 89 crossings, so those two numbers could move by several points either way on a different set.

The targets above were set inside the project as development milestones; the assignment brief does not set any metric thresholds. One target was changed during the work, and this is disclosed: see B6.

## What was tried and what helped
- **Fixing the labels mattered most for measuring lanes correctly.** Many hand-drawn lane lines turned out to be shifted a few pixels off the real paint. The model was often *more* accurate than the label, and was being marked wrong for it. Automatically nudging these labels back onto the paint (train and validation only) made the scores reflect reality much better.
- **Higher input resolution, a sharper lane-painting layer and lighting/colour variation** (random brightness, contrast and noise during training) gave small, real improvements.
- **Things that did not help:**
  - Pushing the crossing detector harder (a 5× learning rate) broke it.
  - Even bigger input images did not beat the chosen model.
  - A separate, dedicated crossing detector trained the standard way scored worse than the combined model.
- **Lane metric:** a strict pixel-overlap score is misleading for thin lines. A label compared with *itself* shifted by just 1 pixel scores only 0.62 out of 1. So the main lane score now allows 2 pixels of tolerance. The strict score is still reported.

---

# Part B: the technical version

## B1. Data
- **Source:** the "Mark" CARLA dataset: 100 episodes × 20 frames at 1280×720 from a forward-facing camera on the vehicle centreline (1,999 usable frames). There is no vehicle-pose metadata.
- **Annotations:**
  - Lane-paint rasters: 0 = background, 1 = paint, 255 = ignore. Painted lane boundaries only; crossing paint and side-street crossing bars are *excluded* from the lane target.
  - Crossing boxes: public class ID 7, remapped to internal class 0.
  - Provenance: 1,519 Codex annotations, including 132 human-approved repaired versions, and 480 Claude annotations. All were used with the user's authorisation. Structural and hash checks passed; this is not a claim that every label was independently visually verified.
- **Split:** fixed and episode-level, seed 20261030: train 1,499 (75 episodes) / val 300 (15) / test 200 (10). It was never changed.
- **Label versions:**
  - `combined_v1` is the frozen original (manifest sha256 `eba24685…`).
  - `aligned_v2` is the version used to train the final model (manifest `bc5712de…`).
    - It applies automated *perpendicular snapping* of lane-label components onto the paint. This was a diagnostic finding: 15–19% of Codex lane components were offset ≥ 3 px from the paint, and 8/8 sampled cases were visually confirmed at native resolution.
    - Rules: search ±8 px, accept shifts of 1.5–6.5 px, image-evidence gain ≥ 1.10, elongated components only, and a shifted footprint may not touch any other line.
    - Result: 1,113 components moved in 602 train/val images.
    - Unchanged: the test split, all crossing boxes and the 132 human-approved repairs. Originals are preserved.

## B2. Architecture: one network, two heads
- **Backbone and detector:** YOLO11s, starting from generic COCO-pretrained weights, reconfigured for 1 class (crosswalk). It uses the standard YOLO11 detect head (anchor-free, DFL box regression).
- **Lane decoder:**
  - Forward hooks capture shared backbone features at strides 4, 8 and 16 (layers 2/4/6).
  - Each feature map gets a 1×1 projection to 48 channels, is bilinearly resized to stride 4 and concatenated.
  - These pass through a 3×3 conv + GroupNorm + SiLU, then a 3×3 conv + SiLU, then a 1×1 conv to a lane logit.
- **Stride-2 refinement (v2):**
  - A residual branch takes the stride-2 stem feature (layer 0, projected to 16 channels) and the upsampled decoder features, and predicts a correction added to the upsampled coarse logits.
  - Its last layer is zero-initialised, so training starts exactly from the coarse decoder.
  - Logits are then upsampled to input resolution.
- **Size:** 9.56 M parameters, of which the lane-specific part is 133 k. The checkpoint format is `joint-road-v1` (custom, loaded with `model.py`).
- **Inference:**
  - A single forward pass gives detections and lane logits.
  - Detections: NMS at IoU 0.6, confidence ≥ 0.25.
  - Lane mask: sigmoid ≥ 0.5, un-letterboxed back to the original resolution.

## B3. Training (final model v2b)
| setting | value |
|---|---|
| Input | RGB 0–1, aspect-preserving letterbox **960×544**, pad colour 114 (1280×720 → 960×540) |
| Augmentation | Photometric only: brightness ×0.7–1.3, contrast ×0.7–1.3, saturation ×0.6–1.4, per-channel cast ±5%, Gaussian noise (p = 0.3). **No flips or geometric changes**: flips swap traffic-side conventions, and geometric warps would blur 6-px lines |
| Schedule | 30 epochs. Epochs 1–3 are lane warm-up with the detector frozen; epochs 4–30 train jointly |
| Optimiser | AdamW, weight decay 1e-4, detector LR 1e-4, lane LR 1e-3, gradient clipping at norm 10, batch 4 |
| Loss | YOLO detection loss + lane loss, where lane loss = BCE (pos_weight 8) + soft Dice; ignore pixels are excluded |
| Hardware | MacBook Air M4 (24 GB), PyTorch MPS. 3.1 h total |
| Checkpoint selection | **Pre-registered before training**: the joint-phase epoch with the highest mean(val lane IoU, val crossing AP50). Chosen epoch: **16** |
| Reproducibility | Seed 0; run config stores code/manifest SHA-256 hashes and library versions. MPS is not bitwise deterministic |

## B4. Experiments (validation set unless stated)
"Lane IoU" is strict IoU on the 640 raster; "F1@2px" is the tolerance metric (B5). CIs are 95% episode bootstrap.

| run | what changed | outcome |
|---|---|---|
| **v1** (baseline) | 640×640, no augmentation, 40 epochs, CPU, selected by val loss | Lane IoU .577, F1@2px .911, crossing P .854 / R .642 / AP50 .741 |
| Ablation | v1 without the 380 Claude-labelled training images | Lane IoU tracked v1 within ~0.005 through epoch 27, so Claude labels were not hurting lanes. Paused at the user's request |
| v2a | 960×544 + stride-2 refinement + photometric augmentation, 30 epochs, original labels | Better crossings (AP50 .765); exposed the label-offset problem |
| **v2b (final)** | v2a settings trained on `aligned_v2` labels | Lane IoU .579 [.540, .624], **F1@2px .920** [.906, .936], P .836, R .672 [.514, .766], **AP50 .775** [.677, .863] |
| v2c | from v2b: 1280×736 input, detector LR ×5 | Detector collapsed (AP50 .04, recovered only to .55). Stopped |
| v2d | v2c with normal detector LR | Lane up to .594 but crossings worse (AP50 ≤ .72). Stopped |
| Standalone detector | YOLO11s, standard Ultralytics recipe, 960, 40 epochs | P .693 / R .693 / AP50 .745 under our protocol: worse than v2b |

- **Paired comparison:** the only statistically clear v2b-over-v1 gain is F1@2px, +0.004 to +0.014.
- **Crossing-miss analysis:** no confidence threshold gives both P ≥ .85 and R ≥ .80. Missed crossings have a median height of **20 px** (at 960×544), versus 61 px for detected ones.
- **Final-model rule:** declared in writing *before* the standalone detector's result existed: pick the candidate passing more gates on val. v2b passed 2 gates against at most 1 for the alternative, so v2b was chosen.

## B5. Metrics
- **Lane F1@2px (primary lane metric):**
  - Scored on the 640-letterbox raster.
  - Precision = share of predicted lane pixels within 2 px of a label pixel.
  - Recall = share of label pixels within 2 px of a predicted pixel.
  - F1 is their harmonic mean.
  - Ignore pixels are excluded.
- **Strict lane IoU:** TP / (TP + FP + FN) on the same raster.
  - Context: label self-agreement is 0.623 when shifted 1 px, 0.414 when shifted 2 px and 0.510 when drawn 2 px thicker. So IoU ≥ 0.60 effectively demands about 1-px agreement with the annotator on ~6-px lines.
- **Crossings:**
  - Precision and recall at confidence 0.25, with one-to-one matching at box IoU ≥ 0.5.
  - AP50 is the area under the precision–recall curve at IoU 0.5.
  - AP50:95 averages over IoU 0.50–0.95.
- **Lane centre:**
  - The reference is the *same estimator run on the hand-drawn label mask*, because there is no ground-truth vehicle pose.
  - Error = |offset_pred − offset_ref| in lane widths, plus heading error in degrees.
  - This measures consistency with the labels, not true lane-keeping accuracy.
- **Uncertainty:** 95% CIs from 1,000 bootstrap resamples of **whole episodes**, because frames within an episode are correlated.

## B6. Development gates and the one change
- **Original gates** (set before v1): lane IoU ≥ .60, AP50 ≥ .75, precision ≥ .85, recall ≥ .80.
- **The change:** after the experiments above, the lane gate was **changed to F1@2px ≥ .80**, because of the self-agreement analysis in B5. Strict IoU is still reported.
- This change was made *after seeing results*. It is documented in the experiment log and README.
- The crossing gates were not changed.

## B7. Final evaluation: reserved test, run once (200 images, 10 episodes, 89 crossings)
| metric | value [95% CI] | gate |
|---|---|---|
| Lane F1@2px | **.895** [.872, .926] | PASS |
| Lane strict IoU (640 raster) | .525 [.470, .598] (native .554) | reported |
| Crossing AP50 | **.784** [.662, .899] | PASS |
| Crossing precision @0.25 | .778 [.611, .886] | FAIL |
| Crossing recall @0.25 | .787 [.694, .882] | FAIL (CI includes .80) |
| Crossing AP50:95 | .421 | – |
| Lane centre, offset within 0.10 lane widths | **93.3%** (n = 179); 97.8% on high-confidence frames (n = 137) | – |
| Lane centre, median offset error / heading error | 0.003 lane widths / 0.6° (p90 6.6°) | – |
| Lane centre, mode agreement (both / left / right / none) | 0.925 | – |

- Test lane labels are the original, un-snapped `combined_v1` labels. This lowers strict IoU relative to validation; F1@2px is less affected.
- **How to describe this result:** an *episode-held-out* evaluation on project data with prior exposure and unknown route overlap. It is **not** route-disjoint or blind generalisation, and **not** a deployment-safety evaluation.

## B8. Lane-centre algorithm (`lane_center.py`)
1. Downscale the lane mask by 0.5 and remove everything above the calibrated horizon.
2. Take connected components and keep line-like ones (area ≥ 12 px, vertical span ≥ 6 px, |dx/dy| ≤ 4). Least-squares fit x = a·y + b to each.
3. **Merge dashes:** going from the largest pieces to the smallest, a piece joins a group if its pixels lie within 12% of the local lane width of the group's line, and the line is refitted.
4. The ego-lane boundaries are the nearest lines left and right of the image centre x₀ = W/2 (the camera is on the vehicle centreline), each extrapolated to the bottom row.
5. **Lane-width model:** w(y) = k·(y − y_h), calibrated from the *training label masks*: k = 1.741, y_h = 357.7, giving ~629 px at the bottom row, from 742 frames. A pair narrower than 0.6× or wider than 1.6× the expected width is rejected, and the side closest to half a lane from x₀ is kept.
6. The centre line is the midpoint of the two boundaries. If only one side is visible (for example a kerb with no paint), the centre is that boundary ± w(y)/2.
7. **Outputs:**
   - `offset_norm` = (x₀ − centre_x) / width at the bottom row.
   - `heading_deg` = angle of the centre line from vertical.
   - `low_confidence` is set when a boundary is extrapolated over more than 50% of the road region.
8. The design was iterated on validation failures; the test result above is the independent check.

---

# Part C: using it, its limits, and next steps

## C1. How to test it
See the README "Test it" section. In short:
```bash
pip install -r requirements.txt
python selftest.py                                              # PASS/FAIL against saved expected outputs
python run_video.py sample_frames/episode_0014 --out demo_out   # overlay video + per-frame CSV
```
Speed: ~190 ms/frame on an M4 CPU and ~50 ms on the M4 GPU, end-to-end.

## C2. How to use it in code
```python
from road_perception import RoadPerception
rp = RoadPerception('joint_road_model.pt', 'lane_center_calibration.json', device='cpu')   # or 'cuda' / 'mps'
out = rp(frame_bgr)        # 1280x720 BGR numpy array (OpenCV / CARLA camera converted to BGR), or an image path
out['lane_mask']           # HxW bool
out['crosswalks']          # [{'class':'crosswalk','class_id':7,'confidence':0.83,'xyxy':[x1,y1,x2,y2]}, ...]
out['lane_center']         # {'mode','offset_norm','offset_px','heading_deg','center_line','low_confidence',...}
vis = rp.draw(frame_bgr, out)
```
**For an RL / control state:**
- Use `offset_norm` (lane deviation) and `heading_deg`.
- When `low_confidence` is True or `mode == 'none'`, hold the last good value or fall back to the CARLA waypoint.
- Use crossing boxes, for example the nearest box's bottom edge, as a "crossing ahead" signal.

## C3. Limitations
- **Distant crossings** (less than ~20 px tall) are often missed, and there are some false alarms. Crossing precision and recall are below the project targets.
- **Lane centre:**
  - It assumes a centre-mounted forward camera with the same field of view. If the camera changes, re-run `lane_center.calibrate()` on label masks.
  - It fits straight boundaries, so sharp curves are approximate.
  - Offsets are image-space, not metres.
- **Domain:** trained and tested only on CARLA frames from this project, with unknown route overlap between splits. It is not validated for real roads, night, rain or other towns.
- **Security:** checkpoints are loaded with `torch.load(weights_only=False)`. Only load trusted files.

## C4. What this covers for the assignment, and what's next
| Brief component | Status |
|---|---|
| Dataset and annotation | Done: 1,999 annotated frames, versioned labels, documented label alignment |
| Model development and training strategy | Done: joint multi-task network, experiments, pre-registered selection, held-out test with CIs |
| Local deployment | Inference API, CLI, video demo and self-test are ready. A UI is not built |
| Cloud deployment, UI, scalability, demo | **Teammate's part.** Suggested start: wrap `RoadPerception` in a small web API (e.g. FastAPI) or a Streamlit/Gradio page using `rp.draw()`; containerise with the pinned `requirements.txt` |

**Ideas to improve crossings, if time allows:**
1. Train the detector much longer (it was still improving at 40 epochs).
2. Add tiling or higher-resolution crops for the far field.
3. Annotate more distant crossings.

---

## Glossary
- **CARLA:** an open-source driving simulator; our frames come from it.
- **Episode:** one short recording (20 frames). Splitting by episode keeps near-identical frames out of the test set.
- **Segmentation mask:** an image the same size as the frame, where each pixel is marked "lane paint" or "not".
- **Bounding box:** a rectangle around an object, given as (x1, y1, x2, y2) in pixels.
- **Confidence:** the model's score from 0 to 1 for a box. We keep boxes scoring ≥ 0.25.
- **Precision:** of the things the model flagged, the share that are real.
- **Recall:** of the real things, the share the model found.
- **F1:** a single score balancing precision and recall (their harmonic mean).
- **IoU (Intersection over Union):** overlap ÷ combined area. 1 means identical.
- **AP50:** average precision across all confidence thresholds, counting a box correct when its IoU with the label is ≥ 0.5. It summarises detection quality in one number.
- **Backbone / head:** the shared feature-extracting part of a network, and the task-specific parts on top.
- **YOLO11s:** a fast, widely used object-detection network, small ("s") variant.
- **Letterbox:** resizing an image to fit the model's input without stretching, padding the leftover area with grey.
- **Augmentation:** randomly altering training images (here brightness, contrast and noise) so the model copes with variation.
- **Learning rate (LR):** how big each training update step is.
- **Validation set:** images used to choose settings and the best checkpoint.
- **Test set:** images kept untouched until the very end, for an honest final score.
- **Bootstrap CI:** a range showing how much a score could vary with different data, from resampling episodes many times.
- **Pre-registered rule:** a decision rule written down before seeing the results, so the choice can't be biased by them.
- **Offset (lane widths):** how far the car is from the lane centre, as a fraction of the lane width.

# CARLA perception: seven-class YOLO26n

Detects vehicles, red, yellow and green traffic lights, adult and child pedestrians, and speed limit signs in CARLA camera frames. Includes a local browser demo with three pages: this detector, the joint road model, and speed-limit reading.

**Status: final.** This is the model the driving agent uses. It was scored once on the held-out test splits on 22 November 2026 (see Results).

## Set up

Follow route 2 in the main README: from the repository root, create an
environment and run `pip install -r requirements-perception.txt`, then
`python perception/demo.py --device cpu`. Open http://127.0.0.1:8773. Use
`--device mps` on Apple silicon or `--device cuda` on an NVIDIA GPU. The model,
example frames and statistics are already in this folder.

## Demo pages

| Page | Address | What it shows | Example frames |
|---|---|---|---|
| Detector | `/` | Boxes for the seven classes, counts against the hand labels, held-out test and validation results, dataset | 50 frames from the 20 held-out Round 2 test episodes |
| Road model | `/road` | Lane paint, pedestrian crossings and lane centre from `../joint_road_model` | 40 frames, 4 from each of the road model's 10 reserved test episodes |
| Speed signs | `/signs` | Each detected sign, the cleaned cut-outs sent to OCR and the value read (`../dreamerv3_attention/speed_sign.py`) | 30 held-out test frames with a labelled sign |

All example frames were picked by a fixed seeded rule, not by how well a model does on them. You can also upload your own image on every page.

The road model and speed-sign pages are optional. If the road model or `rapidocr_onnxruntime==1.2.3` is missing, those two pages say they are unavailable and the detector page runs as before. Uploads are kept for the session only, in `local_runs/`.

## Use the model in your code

```python
from ultralytics import YOLO
from dedupe import merge_duplicates

model = YOLO("carla_yolo26_demo_results/mac_demo/best.pt")
result = model.predict("frame.png", imgsz=1280, conf=0.25)[0]
rows, merged_count = merge_duplicates(result.boxes.data.cpu().numpy())
# rows: x1, y1, x2, y2, confidence, class_id, in pixels of the original frame
```

| Class ID | Name |
|---|---|
| 0 | other_vehicle |
| 1 | red_light |
| 2 | yellow_light |
| 3 | green_light |
| 4 | adult_pedestrian |
| 5 | child_pedestrian |
| 6 | speed_limit |

Run `merge_duplicates` after detection, because the model often draws two boxes on one object. It keeps the larger box with the more confident class and score. On speed limits, the larger box covers the sign and pole.

## Data

Our own CARLA captures, labelled by the team.

- **Round 1:** 3,280 frames at 800 x 600, four classes (vehicle and the three light colours), split by blocks of time: 1,971 train, 648 validation, 661 test.
- **Round 2:** 100 episodes captured at 200 frames each (20,000 frames, 1280 x 720). 9,060 frames were hand-labelled by two annotators, 25 to 119 per episode. Frames with no objects, or with only classes we already had plenty of, were skipped; labelling went to yellow lights and to adult and child pedestrians. Split by whole episode: 70 train, 10 validation, 20 test (6,287 / 889 / 1,884 frames).
- **Label fix:** the two annotators boxed traffic lights differently (whole housing with padding, or only the lit lamp). For training, annotator A's light boxes of 15 px or more were scaled to 0.78 width and 0.85 height, on copies; the source labels are unchanged.

## Training

YOLO26n, input 1280, starting from the pretrained checkpoint. A first fine-tune on Round 1, then a retrain on Round 1 and Round 2 together, then a fine-tune at 1280 input, then a final retrain on the corrected labels (1 November 2026): AdamW, learning rate 0.0005 with a cosine schedule, batch 4, mosaic off, seed 3001, a 100-minute limit (about 2.3 epochs). Training images with a yellow light appear three times per epoch. Loss is Ultralytics' standard YOLO26 detection loss. Round 2 validation chose the checkpoint; the test splits were never used for training or selection.

## Results

### Held-out test (scored once, 22 November 2026)

AP50 at IoU 0.5, all confidence levels, before merging duplicates. Round 2 test: 1,884 frames from 20 held-out episodes. Round 1 test: 661 frames. "Corrected" applies the label fix above to copies of the test labels.

| Class | Round 1 test | Round 2 test | Round 2 test, corrected |
|---|---|---|---|
| other_vehicle | 0.841 | 0.832 | 0.832 |
| red_light | 0.854 | 0.422 | 0.534 |
| yellow_light | 0.202 | 0.402 | 0.497 |
| green_light | 0.915 | 0.448 | 0.552 |
| adult_pedestrian | - | 0.762 | 0.762 |
| child_pedestrian | - | 0.556 | 0.556 |
| speed_limit | - | 0.515 | 0.515 |
| **mAP50** | **0.703** (4 classes) | **0.562** | **0.607** |
| mAP50-95 | 0.469 | 0.309 | 0.335 |

Earlier models were also scored on these test sets in October, so this is not the first look at them.

### Round 2 validation (also chose the checkpoint, so optimistic)

| Labels | Red | Yellow | Green | Mean of 7 classes |
|---|---|---|---|---|
| Original | 0.572 | 0.368 | 0.508 | 0.591 |
| Corrected box sizes | 0.674 | 0.456 | 0.687 | 0.643 |

The corrected labels used scaling factors fitted to an earlier model, so they are not an independent reference.

### Why mAP50-95 is much lower than mAP50

mAP50-95 also asks for boxes that overlap the label by up to 95%. Traffic lights are about 15 x 26 px at the model input, so a 2 px shift already drops the overlap to about 76%, and the two annotators' box edges differ by more than that. The driving agent does not use exact box edges: it needs a light's presence and colour (stop positions come from the map), and object distance comes from the median depth inside the box. So mAP50 is the measure that matters here. A camera-only car that judged distance from box size would need tighter boxes.

## Limits

- One seed and about 2.3 epochs in the final retrain; the validation gains are optimistic.
- Yellow is the weakest light (Round 2 test AP50 0.402), and red and green on Round 2 are much lower than on Round 1. Distant lights are often below the 0.25 display threshold.
- Night frames are in the test episodes but are not scored separately.
- Inconsistent training labels probably cause the double boxes. A single labelling rule and a retrain are the lasting fix.
- All frames come from CARLA. Real camera footage is untested.
- A missing light means unknown, not clear to go. This is a coursework model and has not been tested for real vehicles.

## Licence

The model and demo use Ultralytics YOLO, which is licensed under AGPL-3.0. Check that this fits how the project is shared.

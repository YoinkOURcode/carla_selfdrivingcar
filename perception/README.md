# CARLA perception: seven-class YOLO26n

Detects vehicles, red, yellow and green traffic lights, adult and child pedestrians, and speed limit signs in CARLA camera frames. Includes a local browser demo.

**Status: provisional.** This model has no clean test score. See Limits.

## Set up

Python 3.12 was used. The model, example frames and statistics are already in this folder.

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python demo.py --device mps
```

Windows:

```bash
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python demo.py --device cpu
```

Then open http://127.0.0.1:8773. Use `--device cpu` on any machine, `mps` on Apple silicon or `cuda` on an NVIDIA GPU. CUDA and Windows have not been tested, and CUDA may need the PyTorch build for your system from pytorch.org.

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

## Results

Round 2 validation, AP50, before merging duplicates:

| Labels | Red | Yellow | Green | Mean of 7 classes |
|---|---|---|---|---|
| Original | 0.572 | 0.368 | 0.508 | 0.591 |
| Corrected box sizes | 0.674 | 0.456 | 0.687 | 0.643 |

The corrected labels used scaling factors fitted to an earlier model, so they are not an independent reference.

## Limits

- No clean test score. Validation also selected the checkpoint, so these figures are optimistic.
- One seed, about 2.3 epochs of fine-tuning, and 81 yellow validation boxes.
- Night footage has no held-out evaluation.
- Yellow lights are often missed, and distant lights are often below the 0.25 display threshold.
- Inconsistent training labels probably cause the double boxes. A single labelling rule and a retrain are the lasting fix.
- A missing light means unknown, not clear to go. This is a coursework model and has not been tested for real vehicles.

## Licence

The model and demo use Ultralytics YOLO, which is licensed under AGPL-3.0. Check that this fits how the project is shared.

# Training the joint road model

The rest of this folder is the final package (weights, inference code, metrics). This page covers the code that produced the weights. `WORK_SUMMARY.md` explains the method and the experiments.

## Files

| File | Purpose |
|---|---|
| `train_v2.py` | Training script for the final checkpoint (v2b) |
| `data.py` | Dataset: image, lane mask and crossing boxes; letterbox and photometric augmentation |
| `eval_common.py` | Validation metrics and the pass/fail gate, used during training and by the evaluation scripts |
| `model.py` | The network (already part of the package) |
| `reset_v1/build_combined_dataset.py` | Builds the frozen dataset `dataset_combined_v1` with its episode-level split |
| `reset_v1/build_aligned_v2.py` | Builds `dataset_aligned_v2`: lane labels snapped onto the paint (train and val only) |
| `reset_v1/evaluate_v2.py` | Lane and crossing metrics for a checkpoint; `--final-test` is the one-time reserved test run |
| `reset_v1/evaluate_lane_center.py` | Lane-centre metrics |
| `reset_v1/bootstrap_ci.py` | Episode-level bootstrap confidence intervals |
| `reset_v1/build_yolo_crosswalk_v2.py`, `reset_v1/train_detector_v2.py` | The standalone crossing-detector baseline that the joint model was compared against |

## What is not in the repository

- **The dataset.** The scripts expect `reset_v1/dataset_aligned_v2/manifest.json` with its images, lane masks and box labels. The two build scripts also read the annotation working folders they were built from, so they document how the dataset was made rather than run from a fresh clone.
- **Starting weights.** `yolo11s.pt` (COCO-pretrained, from Ultralytics). The default path is `annotation_pilot/models/yolo11s.pt` under the repository root; pass `--weights` to use another location.

## Command used for the final checkpoint

Run from this folder, with the environment from `requirements-perception.txt`:

```bash
python train_v2.py \
  --manifest reset_v1/dataset_aligned_v2/manifest.json \
  --out reset_v1/training_v2/joint_v2b_aligned_rect960_refine_aug \
  --epochs 30 --warmup-epochs 3 --rect 960 544 --batch 4 \
  --refine-index 0 --augment \
  --detector-lr 1e-4 --lane-lr 1e-3 --lane-pos-weight 8.0 --seed 0 \
  --weights yolo11s.pt --device mps \
  --allow-authorized-supplied-labels
```

- `--device` takes `mps`, `cuda` or `cpu`. The final run used `mps` on an Apple M4.
- Epochs 1 to 3 train only the lane decoder; epochs 4 to 30 train everything.
- `best.pt` is the joint-phase epoch with the highest mean of validation lane IoU and crossing AP50. That rule was fixed before training. It picked epoch 16, which became `joint_road_model.pt`.
- The script refuses to write into an existing `--out` folder and never reads the test split.

Then evaluate on validation:

```bash
python reset_v1/evaluate_v2.py --checkpoint reset_v1/training_v2/joint_v2b_aligned_rect960_refine_aug/best.pt \
  --manifest reset_v1/dataset_aligned_v2/manifest.json --rect 960 544 --out reset_v1/training_v2/val_check
```

## Version note

`model.py` and `data.py` are the versions used for the final run. `train_v2.py` and `eval_common.py` were edited after it, for later experiments (the `--lane-only` option) and for the 2 px tolerance lane metric, so they are not byte-identical to the files that trained v2b. Because training on Apple GPU is not bitwise deterministic either, a rerun should land close to the published metrics rather than reproduce the checkpoint exactly.

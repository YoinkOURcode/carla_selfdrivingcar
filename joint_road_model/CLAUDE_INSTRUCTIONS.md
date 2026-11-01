# Instructions for Claude Code (Windows)

**How to use this file:** unzip the package, open Claude Code in that folder, and send this message:

> Read CLAUDE_INSTRUCTIONS.md and follow it. My CARLA folder is: `<path, or "I don't have CARLA">`. Camera settings used for the training frames: `<values, or "unknown">`.

---

## Your task
Set up and test a trained road-perception model on this Windows PC, using a browser demo and a live CARLA run, then report the results honestly. Read `START_HERE_WINDOWS.md` (the same steps for a person) and `README.md` first. Use `WORK_SUMMARY.md` for background.

**The model:** one network finds lane paint and pedestrian crossings. `lane_center.py` then works out the lane centre from the lane mask with geometry. `road_perception.py` is the one-call interface.

## Rules
- **Don't change the model or its results.** Leave `joint_road_model.pt`, `lane_center_calibration.json`, `metrics/` and `sample_frames/expected/` untouched. Don't retrain or recalibrate. If `lane_center.py` or `road_perception.py` seems wrong, report it rather than changing it.
- **You may fix the setup and test scripts** (`*.bat`, `carla_live_test.py`, `demo_app.py`). These have not been fully tested on Windows or real CARLA. List every change in your report and explain why. Editing them will make the checksum step of `selftest.py` report changed files; that is expected, so say so.
- **Ask the user before:** downloading CARLA or anything else over 1 GB, installing system software (Python, GPU drivers), or killing a process you didn't start. Installing Python packages into the project's `.venv` is fine.
- **Don't change Windows settings or use credentials.** Only load the checkpoint in this package, because it uses `torch.load(weights_only=False)`.
- **Run commands from this folder, keep all new files in it, and don't change the CARLA install.**

## Steps
Work through these in order. If a step fails, diagnose it and fix what the rules allow. After 2–3 failed attempts at one problem, stop and ask the user.

1. **Check the PC.** Use `py -0`, `nvidia-smi` and free disk space. Find the CARLA version from the user's folder (the `VERSION` file, or the `PythonAPI\carla\dist` file names). If CARLA is older than 0.9.15, tell the user: `carla` has no Python 3.10 package for those versions.
2. **Install.** Run `setup_windows.bat <version>`. Confirm that CUDA is available if there's an NVIDIA GPU.
3. **Self-test.** Run `run_selftest.bat` and require `RESULT: PASS`. If it fails, show the exact mismatch and stop. Don't loosen tolerances without the user's approval.
4. **Demo.** Start `run_demo.bat` as a background process and check that `http://127.0.0.1:8790/api/episodes` responds, then give the user the link. Stop your copy afterwards if the GPU memory is needed.
5. **Start CARLA.** Launch `CarlaUE4.exe -quality-level=Epic` in the background, or ask the user to. Then run `run_carla_test.bat --check` and resolve any version mismatch.
6. **Short trial.** Run `run_carla_test.bat --town Town03 --seconds 15 --no-window --out carla_runs\trial`, passing the user's camera settings if given. Open a few images in `snapshots\` and **look at them**. Check that:
   - the image is a normal forward road view with no vehicle bonnet blocking it, and colours look right (roads grey, not tinted blue);
   - green follows lane paint;
   - the signs in `frames.csv` make sense, and yellow map-crossing boxes land on painted crossings.

   If the yellow boxes are misplaced, or `true_offset_norm` looks wrong, the bug is in the test script. Fix it before going on.
7. **Full test.** Run the five runs listed in step 4 of `START_HERE_WINDOWS.md`. The two `--cam-y` runs are the key check for lane centre: moving the camera 0.6 m right must change `offset_norm` by about +0.6 divided by the lane width.
8. **Review.** Read each `summary.json` and open snapshots from each run, including at least a few around crossings and junctions. Compare `model` and `true` offsets.

## Report
Reply to the user in plain language, using this structure:
1. **Verdict for each goal:** lane detection, crossing detection and lane centre. Say whether each works live, with the numbers and what you saw.
2. **Results table**, one row per run: town, weather, frames, share with estimate, offset `within_0.10` (all, and excluding low-confidence frames and junctions), median error, crossing `share_matched` by distance, detections not matching any map crossing, ms per frame.
3. **Off-centre test:** expected versus measured offset, and whether the sign is correct.
4. **Setup used:** CARLA version, GPU, Python and torch versions, camera settings and whether they were assumed, and the `camera_check` result.
5. **What failed or looked wrong**, with snapshot paths; every script change you made; anything you skipped.

## Honesty requirements
- **Saying what's been checked:** only say you visually checked something if you opened the image. Don't call a run successful just because the script finished.
- **Crossing numbers are a rough check, not accuracy.** The map reference has no occlusion handling, and matching uses box IoU ≥ 0.3. Disagreement may be the reference's fault, so inspect the frames before blaming the model. Never present them as AP, precision or recall, or compare them directly with the offline results.
- **Known offline results** (`README.md`, reserved test, against labels): lane F1@2px 0.895; crossing AP50 0.784, precision 0.778, recall 0.787, with the last two below project targets; lane centre 93% of frames within 0.10 lane widths, 98% on high-confidence frames. Those lane-centre numbers were measured against the labels, while the live test uses true position, so they aren't the same measurement.
- **Known weakness:** near junctions the lane-centre estimate can be badly wrong and sensitive to tiny image changes. It's usually flagged `low_confidence`. Report how often it was wrong *without* being flagged, because that's the dangerous case.
- **Camera:** the dataset doesn't record the camera mount or field of view, so unless the user supplied them, results depend on assumptions. A poor result might come from a camera mismatch or from a town or weather the model didn't see, so say which you could and couldn't rule out.
- **Scope:** this is a simulator test with autopilot, no other traffic, and traffic lights ignored. It's not evidence of real-world or deployment safety.

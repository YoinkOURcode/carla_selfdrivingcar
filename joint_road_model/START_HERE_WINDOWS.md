# Run the joint road model on Windows: manual steps

This covers the browser demo and a live CARLA test. To have Claude do it instead, see `CLAUDE_INSTRUCTIONS.md`.

**Status of this export:** the model, demo and self-test were tested on a Mac, including with the Python and PyTorch versions used here. The `.bat` files have not been run on Windows, and `carla_live_test.py` has only been run against a simulated CARLA. Your run is its first real test, so expect that something may need fixing.

## What you need
| Item | Notes |
|---|---|
| Windows 10/11, 64-bit | |
| NVIDIA GPU, ideally 8 GB+ | CARLA and the model share it. Without an NVIDIA GPU the demo still works on CPU, but CARLA will struggle |
| Python 3.10, 64-bit | From python.org. Keep the "py launcher" option on. Needed for the demo too |
| CARLA 0.9.15 or 0.9.16 | Only for step 4. If a teammate already has it, use theirs |
| About 25 GB free disk | Mostly CARLA; check the size on its download page |

Unzip to a short path with plain English letters, such as `C:\road_model`. OpenCV cannot read paths with non-English characters.

## Step 1: Install
Open Command Prompt in the folder and run this with **your CARLA version**. Use the default if you don't have CARLA yet:
```bat
cd C:\road_model
setup_windows.bat 0.9.16
```
At the end, look for `CUDA available: True`. If it says `False` on a PC with an NVIDIA GPU, update the graphics driver and run setup again.

## Step 2: Check the package
```bat
run_selftest.bat
```
You want `RESULT: PASS`. It also writes example pictures to `selftest_out\`.

## Step 3: Browser demo (no CARLA needed)
```bat
run_demo.bat
```
Your browser opens at http://127.0.0.1:8790. Use Play, the slider and **Upload image…**. To browse your full frames, run `run_demo.bat C:\path\to\Mark`. Close the window or press Ctrl+C to stop it.

## Step 4: Live CARLA test
**Get CARLA** if you don't have it: https://github.com/carla-simulator/carla/releases (choose the 0.9.15 or 0.9.16 release, then the Windows package). Unzip it to a short path. The version must match step 1.

**In a first Command Prompt**, start CARLA and leave it open:
```bat
cd C:\path\to\CARLA
CarlaUE4.exe -quality-level=Epic
```
Wait for the 3D city window. If it runs out of graphics memory, add `-ResX=800 -ResY=600 -windowed`. `-quality-level=Low` also helps, but changes the look of the scene, so results may get worse for that reason alone.

**In a second Command Prompt**, check the connection, then run the tests one at a time:
```bat
cd C:\road_model
run_carla_test.bat --check

run_carla_test.bat --town Town03 --seconds 120 --out carla_runs\town03
run_carla_test.bat --town Town03 --seconds 60 --cam-y 0.6 --out carla_runs\town03_right
run_carla_test.bat --town Town03 --seconds 60 --cam-y -0.6 --out carla_runs\town03_left
run_carla_test.bat --town Town05 --seconds 120 --out carla_runs\town05
run_carla_test.bat --town Town03 --weather HardRainNoon --seconds 120 --out carla_runs\town03_rain
```
A window shows the live view; press `q` to stop early. Each run saves `overlay.mp4`, `frames.csv`, `summary.json` and `snapshots\` in its folder.

**Camera settings:** the dataset doesn't record how the camera was set up. If you know the settings used to collect the frames, add them to each run, for example `--fov 90 --cam-x 1.5 --cam-z 2.0 --cam-pitch 0`. Otherwise those defaults are used; height and pitch were estimated from the training labels, and the field of view is an assumption.

## Step 5: Read the results
On screen: green is lane paint, magenta is a model crossing, cyan is the lane edges, red is the lane centre, white is the car, and yellow is a crossing from CARLA's map.

| Question | Where to look in `summary.json` | Good sign |
|---|---|---|
| Are lanes detected? | `overlay.mp4`, `lane_centre.share_with_estimate` | Green follows the paint; most frames have an estimate |
| Is the lane centre right? | `lane_centre.abs_error_excl_low_confidence_and_junctions` | `within_0.10` is high. On saved test frames it was 98%, but that was against labels rather than true position, so expect lower |
| Does the offset respond? | `mean_model_offset_norm` in the `--cam-y` runs | About +0.17 for `0.6` and −0.17 for `-0.6` with a 3.5 m lane; compare with `mean_true_offset_norm`. **A flipped sign is a bug** |
| Are crossings detected? | `overlay.mp4`, `crossings.by_distance` | Magenta on nearby crossings; `share_matched` is highest at 0–15 m |
| Is the camera similar to training? | `camera_check` | The two `k` values are within about 25% |

These are guides, not pass marks. The crossing numbers are a rough check: the map doesn't know when a crossing is hidden behind a building, so don't quote them as the model's accuracy. Trust the video first.

## Common problems
| Problem | Fix |
|---|---|
| `Python 3.10 was not found` | Install it from python.org, open a new Command Prompt, try again |
| `time-out while waiting for the simulator` | CARLA isn't fully loaded yet, or it's on another port. Wait and retry |
| Version warning, or a crash right after connecting | Client and server differ. Run `setup_windows.bat <your CARLA version>` again |
| Out of memory, or CARLA crashes | Close other programs and lower the CARLA window size or quality. To free GPU memory for CARLA, run the model on CPU with `--device cpu` |
| The car is stuck | Try another start point: `--spawn 10` |
| CARLA stays frozen after a crash | Close and restart CARLA; a crashed script can't switch off synchronous mode |
| Self-test `FAIL` on only tiny differences | Send the output to whoever is helping. Small GPU differences are possible, and the tolerance may need adjusting |

## Sending results back
Zip the `carla_runs` folder and note your CARLA version, GPU and any camera settings used.

# Live YOLO + depth: distance, angle, object speed and speed limits

`yolo_depth_live.py` drives a car on CARLA's autopilot with an RGB camera and a depth camera at the same mount. Each frame, it runs the team's seven-class YOLO26 model (`../carla_yolo26_demo_results/mac_demo/best.pt`) and shows, for every detection:

- **distance** from the camera, in metres
- **angle** from straight ahead, for example `15 deg L` or `8 deg R`
- **speed** of vehicles and pedestrians over the ground, in km/h
- the **number** on speed limit signs

The car also obeys the speed limits it reads: when it passes a sign whose number was confirmed, its top speed is set to that number.

## Relation to `dreamerv3_attention/vision.py`

This is a separate tool, and it does not change the camera-only provider. It uses **simulator data that `vision.py` deliberately excludes**:

- the CARLA depth camera, for distance
- the camera's pose from CARLA, to cancel our own motion out of object speeds

The speed limit reading uses only the RGB image, so it could be moved into `vision.py` later to fill `speed_limit_mps`. That has not been done.

## Setup

These steps were tested on Windows 11 with Python 3.12, CARLA 0.9.16 (packaged release) and an RTX 5070.

```
py -3.12 -m pip install carla==0.9.16 ultralytics opencv-python rapidocr_onnxruntime==1.2.3
```

- **GPU:** install the CUDA build of PyTorch from pytorch.org. The RTX 50 series needs CUDA 12.8 or newer. The `cu132` build was used.
- **OCR version:** use `rapidocr_onnxruntime==1.2.3`. Version 1.4.4 also runs, but it missed both real CARLA test signs and read fewer rendered signs.
- **Windows Smart App Control:** it blocks PyTorch's unsigned DLLs with `WinError 4551` ("An Application Control policy has blocked this file"). Microsoft offers no per-file exception, so it has to be turned off in Windows Security to use PyTorch.

## Run

1. Start CARLA and load one of Town01 to Town05, the towns the detector was trained on:
   ```
   CarlaUE4.exe
   py -3.12 set_map.py Town01
   ```
2. Run the script from this folder:
   ```
   py -3.12 yolo_depth_live.py
   ```

`classes.txt` (the team's class order) must stay next to the script. When the script is in this folder, it finds the team model and its `model_metadata.json` automatically, and uses image size 1280, confidence 0.25 and the duplicate merging rules from `../dedupe.py`.

| Option | Effect |
|---|---|
| `--model PATH` | Use another YOLO model, for example `yolo26n.pt` (pretrained COCO, mapped to team classes) |
| `--log detections.csv` | Save every detection per frame (see below) |
| `--no-obey` | Only display speed limits; the car keeps following CARLA's map limit |
| `--vehicles N`, `--walkers N` | Traffic and pedestrians to spawn (default 30 and 50) |
| `--imgsz`, `--conf` | Override the model's image size and confidence |

Window keys: `d` switches between the camera and depth views, and `q` quits. Quitting removes the spawned actors and restores CARLA's settings.

`set_map.py --list` shows the available maps. `set_map.py TownXX` switches maps and moves the spectator view to the busiest streets.

## How the values are computed

- **Distance:** the median CARLA depth of the central half of the box, converted from depth along the camera axis to straight-line distance.
- **Angle:** `atan2(right, forward)` of the box centre, using the camera intrinsics (1280x720, FOV 90, mount x=1.5 m and z=2.0 m, the same as `data/config.json`).
- **Speed:** each detection is placed in world coordinates using the camera pose of that exact frame, and followed across frames by a nearest-position tracker within the same kind of object. The speed is a least-squares fit over the last 0.6 s of simulation time, needing at least 5 samples spanning 0.2 s or more. Lights and signs get no speed.
- **Speed limit number:**
  1. The round red-ringed plate is found inside the `speed_limit` box, which may include the pole.
  2. The ring is whitened, the contrast is stretched, and the digits are read with RapidOCR.
  3. Only values used by CARLA's sign assets are accepted: 10, and 30 to 130 in steps of 10.
  4. A sign's value is confirmed once it has been read 3 times and at least twice as often as any other value.
- **Obeying limits:** a confirmed sign last seen within 15 m that then leaves the view counts as passed. The script then calls Traffic Manager `set_desired_speed(ego, value)`, in km/h. The Traffic Manager still slows for turns. Before any sign is passed, the car follows the map limit. In 0.9.16 the Traffic Manager targets exactly the map limit by default.

## CSV log columns

| Column | Meaning |
|---|---|
| `frame`, `time_s` | Frame number and simulation time |
| `track_id` | ID of the tracked object |
| `class`, `confidence` | Detected class and model confidence |
| `distance_m`, `angle_deg` | Distance from the camera; angle, positive to the right |
| `forward_m`, `right_m`, `up_m` | Position relative to the camera |
| `speed_kmh` | The object's speed over the ground |
| `closing_speed_mps` | Change in distance per second; negative means getting closer |
| `motion_dir_deg` | Direction of the object's motion relative to our heading |
| `speed_limit_kmh` | The number read from this sign |
| `applied_limit_kmh` | The limit the car is following |

## Testing

These figures come from tests run outside the simulator, plus live runs in CARLA. The live runs were checked by eye only, not against ground truth.

- **Fake world with known answers:**
  - The real script logic ran with stand-ins for CARLA and the detector, using depth noise and a camera that drives straight and then turns.
  - Speeds: a 54 km/h car was estimated at a median of 53.9 km/h (range 51.7 to 56.9), a parked car at 0.7 km/h (at most 3.1, while turning), and a 5.4 km/h pedestrian at 5.4 km/h.
  - Distance and angle: median errors were 0.09 to 0.14 m and under 0.1 degrees.
  - Tracks: each object kept one track ID.
  - Signs: a rendered 60 sign and then a 30 sign were each confirmed about 20 m ahead and applied exactly at the sign. No wrong value was accepted.
- **Sign reading per frame:**
  - Both legible real CARLA signs among the bundled examples (`ep070_000002`: 90, `ep076_000142`: 30) were read correctly with tight, loose and sign-plus-pole boxes.
  - On rendered signs, the reader was right about 78% of the time at 24 px tall, 90% at 32 px and 100% at 48 px. Wrong reads ran at 0 to 4% per frame before voting.
  - A sign cut off at the image edge was rejected, not guessed.

## Limitations

- **Simulator data:** distance and speed use the depth camera and the simulator camera pose. A real car would need LiDAR, stereo, a depth model and its own odometry.
- **Speed jitter:** partly hidden objects, or boxes whose centre falls on background, can make the speed jump for a few frames.
- **Small signs:** signs under about 18 px tall are rarely read, so a number appears only as the car gets closer.
- **Sign ownership:** the reader does not know which road a sign belongs to. A sign for another road passed within 15 m can be applied.
- **Detector limits:** the detector's own limits apply. See `../README.md`: it is provisional, with weak yellow lights and possible red/green confusion.

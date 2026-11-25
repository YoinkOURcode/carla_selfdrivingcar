# Full setup on Linux (cloud VM or local PC)

This guide goes from an empty Ubuntu 22.04 LTS GPU machine to a running CARLA
0.9.16 server, the DreamerV3 driving agent with the camera vision models, and
the web viewer. Every step runs in a terminal on the VM unless it says
"on your laptop". The main `README.md` explains the project and the full CLI;
this file is only the install and run path.

> **Local Linux PC instead of a cloud VM?** Use Ubuntu 22.04 with an NVIDIA GPU
> and follow the same steps, with three differences: skip step 1 (choosing a
> VM) and step 2.1 if you already log in as a normal user; in step 10 open
> http://localhost:8080 directly, without the SSH tunnel; section 12
> (start on boot) is optional. Windows and macOS cannot run the full system: the
> agent uses Linux-only file locking, and CARLA has no macOS build. They can run
> the vision models alone (route 2 in the main README).

Steps:

1. [Choose the VM](#1-choose-the-vm)
2. [Prepare the system](#2-prepare-the-system)
3. [Download and unpack CARLA 0.9.16](#3-download-and-unpack-carla-0916)
4. [Get the project code](#4-get-the-project-code)
5. [Create the Python environment](#5-create-the-python-environment)
6. [Add the OneDrive files](#6-add-the-onedrive-files)
7. [Check the install without the simulator](#7-check-the-install-without-the-simulator)
8. [Start the CARLA server](#8-start-the-carla-server)
9. [Run the agent](#9-run-the-agent)
10. [Watch it in the browser](#10-watch-it-in-the-browser)
11. [Troubleshooting](#11-troubleshooting)
12. [Demo operation: start on boot, test once, stop, restart before the demo](#12-demo-operation-start-on-boot-test-once-stop-restart-before-the-demo)
13. [Train a new model and other tools](#13-train-a-new-model-and-other-tools)

## 1. Choose the VM

| Item | Minimum | Why |
| --- | --- | --- |
| OS image | Ubuntu 22.04 LTS, x86-64 | CARLA 0.9.16 supports Ubuntu 20.04 and 22.04; there is no ARM build |
| GPU | NVIDIA, 8 GB VRAM or more (RTX 2070 class or better) | CARLA's stated recommendation; the vision models share the same GPU |
| RAM | 32 GB | CARLA, PyTorch and the replay buffer run together |
| Disk | 100 GB | CARLA about 20 GB unpacked plus the download, repository about 4 GB, Python environment about 10 GB, run outputs |
| Inbound ports | SSH (22) only | The viewer and CARLA ports stay private; reach the viewer through an SSH tunnel (step 10) |

A cloud image with the NVIDIA driver already installed saves step 2.2.

## 2. Prepare the system

### 2.1 Work as a normal user, not root

Unreal Engine refuses to start as root. If the VM only gives you root, create
a user and switch to it:

```bash
sudo adduser carla
sudo usermod -aG sudo carla
su - carla
```

### 2.2 NVIDIA driver

Check for a driver first:

```bash
nvidia-smi
```

If the command is missing, install the recommended driver and reboot:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

After the reboot, `nvidia-smi` must list the GPU before you continue.

### 2.3 System packages

```bash
sudo apt update
sudo apt install -y git git-lfs wget tmux build-essential \
    python3.10 python3.10-venv python3.10-dev \
    libomp5 xdg-user-dirs libvulkan1 vulkan-tools \
    libgl1 libglib2.0-0
```

- `libomp5`, `xdg-user-dirs` and `libvulkan1` are needed by the CARLA server binary.
- `libgl1` and `libglib2.0-0` are needed by OpenCV in the Python environment.
- Python 3.10 is the Ubuntu 22.04 default and matches the project's reference install.

Check that Vulkan sees the NVIDIA GPU (CARLA renders through Vulkan):

```bash
vulkaninfo --summary | grep -i -E "deviceName|driverName"
```

The output must name the NVIDIA GPU. If it only shows `llvmpipe`, the NVIDIA
driver is missing or did not load; go back to step 2.2.

## 3. Download and unpack CARLA 0.9.16

The project uses the CARLA 0.9.16 Python client, so the server must be the
same version. Use the packaged release; you do not need Unreal Engine or a
source build.

```bash
mkdir -p ~/carla-0.9.16
cd ~/carla-0.9.16
wget https://downloads.carlasim.com/Linux/CARLA_0.9.16.tar.gz
tar -xzf CARLA_0.9.16.tar.gz
rm CARLA_0.9.16.tar.gz
ls CarlaUE4.sh
```

The archive unpacks into the current folder, which is why the folder is created
first. Town01 (training) and Town02 (held-out evaluation) are in this base
package. `AdditionalMaps_0.9.16.tar.gz` from the same release page is not
needed. Release page:
https://github.com/carla-simulator/carla/releases/tag/0.9.16

## 4. Get the project code

```bash
cd ~
GIT_LFS_SKIP_SMUDGE=1 git clone --branch main --single-branch --depth 1 \
    https://github.com/YoinkOURcode/carla_selfdrivingcar.git
cd ~/carla_selfdrivingcar
```

- `main` holds the final code, with the vision models wired into the agent.
- `--depth 1` skips the commit history (the full history is about 2 GB).
- `GIT_LFS_SKIP_SMUDGE=1` skips CARLA's sample videos, some of which are missing
  from LFS storage and would otherwise stop the clone.

The perception weights come with the clone. Check they arrived as real files, not
LFS pointers (each should be several MB):

```bash
ls -lh perception/carla_yolo26_demo_results/mac_demo/best.pt \
       joint_road_model/joint_road_model.pt \
       joint_road_model/lane_center_calibration.json
```

## 5. Create the Python environment

Create the environment as `venv` in the repository root; the start-on-boot
services in section 12 use that path.

```bash
cd ~/carla_selfdrivingcar
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m dreamerv3_attention preflight
```

On Linux, the pinned `torch==2.7.1` from PyPI already includes CUDA 12.6
support, and `carla==0.9.16` installs the matching Python client. `preflight`
prints a JSON report; it must show `"carla": "0.9.16"` under `versions` and
`"cuda_available": true`. If CUDA is not available, recheck `nvidia-smi`.

Run `source ~/carla_selfdrivingcar/venv/bin/activate` in every new terminal
before using `python`.

## 6. Add the OneDrive files

The trained driving model and its configuration are not in GitHub (`*.pt` files
and `runs/` are git-ignored) because of their size. Download them from the
view-only OneDrive folder linked in the main `README.md`
([CVDL CARLA SUBMISSION on OneDrive](https://sitsingaporetechedu-my.sharepoint.com/:f:/g/personal/2503687_sit_singaporetech_edu_sg/IgCXTbdPMD0IT6NwPhhbny2KAbLdVIvAttJ4U6J2-PyWYW8)) on your laptop, then copy the files to
the VM.

Create the target folder on the VM:

```bash
mkdir -p ~/carla_selfdrivingcar/final/reports
```

The OneDrive folder is called `CVDL CARLA SUBMISSION`. The deployment runs the
trained model on the held-out Town02 scenarios with the camera vision models.
Put each file in this slot (paths on the VM are relative to
`~/carla_selfdrivingcar`):

| OneDrive file | Slot on the VM | Used by |
| --- | --- | --- |
| `models/rl/latest.pt` (6.9 GB, the trained driving model) | `final/latest.pt` | `infer`, `evaluate`, `viewer` |
| `evaluation/town02_final.json` (held-out Town02 configuration) | `final/evaluation.json` | `infer`, `evaluate`, and the viewer configuration below |
| `evaluation/result.json`, `evaluation/trace.jsonl`, `models/rl/config.json`, `models/rl/result.json`, `models/rl/metrics.jsonl` | `final/reports/` | Reference only; not read by the code |
| `models/perception/yolo26n_carla_best.pt` | Not needed | Same file as `perception/carla_yolo26_demo_results/mac_demo/best.pt` in the clone |
| `dataset/`, `videos/` | Not needed | Training data and demo videos, for reference |

`models/rl/config.json` is the Town01 training configuration. It is kept for
reference and is only needed to train again (section 13).

Copy from your laptop with `scp` (run on your laptop, inside the downloaded
`CVDL CARLA SUBMISSION` folder):

```bash
scp models/rl/latest.pt carla@VM_IP:~/carla_selfdrivingcar/final/
scp evaluation/town02_final.json carla@VM_IP:~/carla_selfdrivingcar/final/evaluation.json
scp evaluation/result.json evaluation/trace.jsonl models/rl/config.json carla@VM_IP:~/carla_selfdrivingcar/final/reports/
```

### Put the scenario file where the configuration expects it

`final/evaluation.json` reads its six scenarios (routes, weather, pedestrian
and vehicle encounters) from `runs/experiment_28/evaluation/scenarios.json`.
That file is in this repository as `deployment/town02_scenarios.json`, rebuilt
exactly from the scenario records in the evaluation's `result.json`. Copy it
into place:

```bash
cd ~/carla_selfdrivingcar
mkdir -p runs/experiment_28/evaluation
cp deployment/town02_scenarios.json runs/experiment_28/evaluation/scenarios.json
```

### Make the viewer configuration

The viewer drives one scenario at a time, so it needs a copy of the
configuration with a single scenario selected:

```bash
cd ~/carla_selfdrivingcar
source venv/bin/activate
python - <<'PY'
import json
from pathlib import Path
SCENARIO = "held_out_long_mixed_1"
cfg = json.loads(Path("final/evaluation.json").read_text())
assert cfg["perception"] == "external" and cfg["device"] == "cuda", "not the vision configuration"
cfg["scenarios"] = [SCENARIO]
Path("final/viewer.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("final/viewer.json uses", SCENARIO, "on", cfg["town"])
PY
```

To show another scenario, change `SCENARIO` and run it again:

| Scenario | Weather | Encounters |
| --- | --- | --- |
| `held_out_long_mixed_1` | Clear noon | 2 adults, 2 children |
| `held_out_long_mixed_2` | Wet sunset | 2 adults, 2 children, 2 vehicles |
| `held_out_long_mixed_3` | Soft rain noon | 2 adults, 1 child |
| `held_out_long_mixed_4` | Cloudy noon with fog | 2 adults, 2 children, 2 vehicles |
| `held_out_long_mixed_5` | Clear night | 1 adult, 2 children |
| `held_out_long_mixed_6` | Hard rain night with fog | 2 adults, 2 children, 2 vehicles |

The configuration already uses the camera vision models
(`"perception": "external"` with `dreamerv3_attention.vision:VisionPerceptionProvider`)
on `cuda`; the script stops if it does not.

## 7. Check the install without the simulator

These run on the CPU and do not need CARLA running or the OneDrive files:

```bash
cd ~/carla_selfdrivingcar
source venv/bin/activate
python -m pytest dreamerv3_attention/tests -q
python -m unittest discover -s data/tests
python -m dreamerv3_attention smoke --config dreamerv3_attention/configs/smoke.json
```

`smoke` collects synthetic data, does real learner updates and checks a saved
checkpoint under `/tmp`. It confirms the software works, not driving quality.

## 8. Start the CARLA server

Run the server in `tmux` so it keeps running after you disconnect:

```bash
tmux new -s carla
cd ~/carla-0.9.16
./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low
```

- `-RenderOffScreen` renders on the GPU without a display (no X server needed).
- The server uses ports 2000 and 2001 (and 2002). Start only one server.
- Wait about 30 seconds for it to finish loading, then detach with `Ctrl+B`, then `D`.
  Reattach later with `tmux attach -t carla`.
- These are the same flags the project uses when it launches its own server.

Load the map the run needs. The deployment configurations use Town02; only
training a new model (section 13) uses Town01. The agent does not switch maps itself; it stops with
`Expected Town01; launch that map on a dedicated server first` if the wrong map
is loaded.

```bash
cd ~/carla_selfdrivingcar
source venv/bin/activate
python PythonAPI/util/config.py --host 127.0.0.1 -p 2000 -m Town02
```

Check the town a configuration expects with `grep '"town"' final/evaluation.json`.

## 9. Run the agent

Every live command needs `--budget-seconds` (1 to 86400, including startup) and
a new output folder each time; existing outputs are never overwritten. Only one
live run can use the machine at a time.

### One episode with the downloaded model

Load Town02 first (step 8), then:

```bash
python -m dreamerv3_attention infer \
  --config final/evaluation.json \
  --checkpoint final/latest.pt \
  --budget-seconds 300 --output runs/cloud_infer_01
```

### Held-out evaluation on Town02 with the vision models

The OneDrive evaluation suite:

```bash
python -m dreamerv3_attention evaluate \
  --config final/evaluation.json \
  --checkpoint final/latest.pt \
  --budget-seconds 3600 --output runs/cloud_eval_01
```

Each run folder contains `result.json` (completion, collisions, traffic and
crossing violations, lane tracking, safety-supervisor interventions), the
configuration used, and JSONL metrics.

## 10. Watch it in the browser

Start the viewer on the VM in a second `tmux` window while a run is going:

```bash
tmux new -s viewer
cd ~/carla_selfdrivingcar && source venv/bin/activate
python -m dreamerv3_attention viewer
```

The viewer only accepts connections from the VM itself (`127.0.0.1:8080`), so do
not open port 8080 in the firewall. Forward it over SSH instead. On your laptop:

```bash
ssh -L 8080:127.0.0.1:8080 carla@VM_IP
```

Then open http://localhost:8080 on your laptop. Use `--web-port` on the VM and
the same number in the tunnel if 8080 is taken.

To drive the downloaded model interactively from the viewer instead of watching
another run:

```bash
python -m dreamerv3_attention viewer \
  --config final/viewer.json \
  --checkpoint final/latest.pt --budget-seconds 300
```

Pick a destination and click **Start drive**.

## 11. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Refusing to run with the root privileges` | Run CARLA as a normal user (step 2.1) |
| CARLA exits at once, or a Vulkan error mentions `llvmpipe` or no device | NVIDIA driver missing or not used by Vulkan; check `nvidia-smi` and `vulkaninfo --summary` (step 2) |
| `error while loading shared libraries: libomp.so.5` | `sudo apt install -y libomp5` |
| `Expected Town01; launch that map on a dedicated server first` | Wrong map loaded; run `config.py -m` with the town in the configuration (step 8) |
| Client and server version warning, or a crash right after connecting | Server is not 0.9.16; reinstall it from step 3 |
| `time-out of 10000ms while waiting for the simulator` | Server still loading or not running; check `tmux attach -t carla` |
| `Another CARLA training workload holds the machine lock` | A second live run was started; wait for the first one or stop it |
| `Output already exists` | Choose a new `--output` folder or run name |
| Checkpoint fails to load with a size mismatch | `latest.pt` and the configuration are from different runs (step 6) |
| `FileNotFoundError` for `runs/experiment_28/evaluation/scenarios.json` | The scenario file was not copied into place (step 6) |
| `torch.cuda.is_available()` is `False` | Driver problem; reinstall the driver (step 2.2) and reboot |
| `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |
| Viewer page does not load on the laptop | SSH tunnel not open, or opened to the public IP instead of `127.0.0.1:8080` |

`--ue4-root` in the main README is only for a source-built Unreal Engine. With
the packaged server from step 3, start CARLA yourself as in step 8.

## 12. Demo operation: start on boot, test once, stop, restart before the demo

The VM is set up once, tested once, then stopped so it costs only disk storage.
Before the demo it is started again, and three services bring everything up
without anyone typing commands: the CARLA server, the map, and the viewer with
the trained model loaded. The presenter then opens the viewer in a browser,
picks a destination and clicks **Start drive**.

Complete steps 1 to 9 first, and use the user name `carla` from step 2.1 (change
the paths below if you used another name).

### 12.1 Install the services

CARLA server:

```bash
sudo tee /etc/systemd/system/carla.service >/dev/null <<'UNIT'
[Unit]
Description=CARLA 0.9.16 server
After=network-online.target
Wants=network-online.target

[Service]
User=carla
WorkingDirectory=/home/carla/carla-0.9.16
ExecStart=/home/carla/carla-0.9.16/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
```

Map loader. It waits for the server, then loads the town named in
`final/viewer.json`, so the map always matches the viewer's configuration:

```bash
sudo tee /usr/local/bin/carla-load-map >/dev/null <<'SCRIPT'
#!/bin/bash
# Wait for the CARLA server and load the town named in the model configuration
cd /home/carla/carla_selfdrivingcar || exit 1
TOWN=$(venv/bin/python -c 'import json; print(json.load(open("final/viewer.json"))["town"])') || exit 1
for attempt in $(seq 1 60); do
    venv/bin/python PythonAPI/util/config.py --host 127.0.0.1 -p 2000 -m "$TOWN" && exit 0
    sleep 5
done
exit 1
SCRIPT
sudo chmod +x /usr/local/bin/carla-load-map

sudo tee /etc/systemd/system/carla-map.service >/dev/null <<'UNIT'
[Unit]
Description=Load the configured CARLA map
After=carla.service
Requires=carla.service

[Service]
Type=oneshot
User=carla
WorkingDirectory=/home/carla/carla_selfdrivingcar
ExecStart=/usr/local/bin/carla-load-map
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
```

Viewer, with the downloaded model ready to drive. `--budget-seconds` is the time
limit for each drive started from the page:

```bash
sudo tee /etc/systemd/system/carla-viewer.service >/dev/null <<'UNIT'
[Unit]
Description=DreamerV3 driving viewer
After=carla-map.service
Requires=carla-map.service

[Service]
User=carla
WorkingDirectory=/home/carla/carla_selfdrivingcar
ExecStart=/home/carla/carla_selfdrivingcar/venv/bin/python -m dreamerv3_attention viewer --config final/viewer.json --checkpoint final/latest.pt --budget-seconds 600
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
```

Enable them so they start on every boot:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now carla.service carla-map.service carla-viewer.service
```

If CARLA is still running in `tmux` from step 8, stop it first
(`tmux kill-session -t carla`); two servers cannot share port 2000.

### 12.2 The one-time test

Do this once, after the services are enabled, and tick every line.

1. Reboot the VM (`sudo reboot`) so the test covers the start-on-boot path, not
   a hand-started server.
2. After about 3 minutes, SSH in and check all three services are running:
   `systemctl status carla carla-map carla-viewer --no-pager`. `carla` and
   `carla-viewer` show `active (running)`; `carla-map` shows `active (exited)`.
3. Open the SSH tunnel from step 10 on a laptop and load http://localhost:8080.
4. Pick a destination, click **Start drive**, and watch a full drive: camera
   frames update, the car moves, the traffic-light and object labels appear, and
   the drive ends on its own.
5. Run the Town02 vision evaluation once and keep its `result.json` as the
   reference result for the demo (step 9). Stop `carla-viewer` while it runs
   (`sudo systemctl stop carla-viewer`), because only one live run can use the
   machine, and start it again afterwards.
6. Write down how long the VM took from start to a working viewer page. That
   decides how early to start it on demo day.
7. Test the SSH tunnel from the campus network or the demo room's network if
   possible. Some networks block outgoing SSH; if so, see 12.5.
8. Record a screen video of a complete drive. It is the fallback if the VM or the
   network fails during the demo.

Then stop the VM (12.3).

### 12.3 Stop the VM

```bash
sudo shutdown -h now
```

Or press **Stop** in the cloud console. A stopped VM is not charged for the GPU
or CPU, but its disk is still billed, so do not delete the disk. Do not use
spot or preemptible instances for the demo VM; they can be reclaimed mid-demo.

### 12.4 Before the demo

1. **Start the VM at least an hour before the demo**, or the day before, and
   leave it running. Cloud GPUs are sometimes unavailable in a zone. If the
   console reports no capacity (for example `ZONE_RESOURCE_POOL_EXHAUSTED` on
   Google Cloud or `InsufficientInstanceCapacity` on AWS), retry, or move the disk
   to another zone, which takes time.
2. Note the VM's IP address in the console. Without a reserved static address it
   changes on each start, so update the `ssh -L` command.
3. Repeat test steps 2 to 4 above (services running, tunnel, one full drive).
4. Keep the tunnel and the viewer page open on the presenting laptop, with the
   fallback video ready in another window.
5. Stop the VM after the demo (12.3).

### 12.5 If SSH is blocked on the demo network

Use a phone hotspot for the presenting laptop, or set up Cloudflare Tunnel in
advance: `cloudflared` on the VM connects out to Cloudflare and serves the
viewer on an HTTPS address, with Cloudflare Access adding an email login. The
viewer only accepts requests addressed to `localhost:8080`, so the tunnel's
origin settings must set `httpHostHeader: localhost:8080`. Never open port 8080
or CARLA's ports 2000 to 2002 to the internet: neither has a login, and the
viewer can start drives.

### 12.6 Service troubleshooting

| Symptom | Check |
| --- | --- |
| A service is `failed` | `journalctl -u carla -u carla-map -u carla-viewer -b --no-pager \| tail -50` |
| `carla-map` failed after 5 minutes | CARLA did not finish loading; check `journalctl -u carla -b`, then `sudo systemctl restart carla-map carla-viewer` |
| Viewer page loads but **Start drive** reports another workload | An evaluation or `infer` run is still going; wait for it or stop it |
| Viewer reports the wrong town | `final/viewer.json` was changed after boot; `sudo systemctl restart carla-map carla-viewer` |

## 13. Train a new model and other tools

Steps 1 to 9 run the trained model. Training a new one is only needed to
reproduce the training itself. It needs Town01 loaded (step 8). The tracked
`dreamerv3_attention/configs/attention.json` enables attention and uses CUDA;
`baseline.json` disables attention for comparison. Copy one, then set host,
port, town and perception in the copy (step 6 shows the vision fields).

```bash
python -m dreamerv3_attention train \
  --config dreamerv3_attention/configs/attention.json \
  --budget-seconds 600 --output runs/attention_train
python -m dreamerv3_attention collect \
  --config dreamerv3_attention/configs/attention.json \
  --budget-seconds 600 --output runs/attention_collect
```

`collect` gathers replay without learner updates. Resume training with
`train --checkpoint runs/NAME/latest.pt` and the matching configuration; add
`--world-model-only` to update only the world model. Training requires
`"split": "train"`; keep validation and held-out replay separate.

Compare the agent against the rule-based controller on identical scenarios:

```bash
python -m dreamerv3_attention evaluate \
  --config final/evaluation.json --controller \
  --budget-seconds 3600 --output runs/controller_eval
python -m dreamerv3_attention compare \
  --baseline runs/controller_eval/result.json \
  --candidate runs/cloud_eval_01/result.json --output /tmp/comparison.json
```

Other offline tools: `validate --checkpoint MODEL/latest.pt --data-checkpoint
HELD_OUT/latest.pt` scores one held-out world-model loss batch, and
`audit-controls --trace EVAL/trace.jsonl` summarises control behaviour. Live
integration tests run against an idle matching server with
`DREAMERV3_APPROVED_LIVE_TEST=1 python -m pytest dreamerv3_attention/tests/test_live.py -q`.
`python -m dreamerv3_attention --help` lists every command.

# Demo server files (Vast.ai GPU instance)

These are the scripts the demo server ran. `SETUP_LINUX.md` in the repository root describes a full Linux VM that starts everything with systemd services. The rented Vast.ai instance is a Docker container without systemd, so the demo used `tmux` sessions started by `start.sh` instead, with a Caddy proxy adding a login in front of the two web pages.

Install steps 1 to 9 of `SETUP_LINUX.md` still apply (user `carla`, CARLA 0.9.16 in `~/carla-0.9.16`, this repository in `~/carla_selfdrivingcar` with its `venv`, model files in `final/`). The files here go in `~/demo/` on the server.

## Instance settings

The instance was created with these Docker options, which publish the two proxy ports and expose all NVIDIA driver capabilities to the container, for CARLA's rendering:

```
-p 8081:8081 -p 8082:8082 -e NVIDIA_DRIVER_CAPABILITIES=all
```

Vast maps each of those to a public port, shown on the instance page.

## Files

| File | Purpose |
|---|---|
| `start.sh` | Starts the CARLA server, the proxy, the perception demo and the driving viewer, each in its own `tmux` session. Safe to rerun; run it again after a reboot. |
| `load_map.py` | Waits for CARLA, clears a leftover synchronous session and loads the town named in `final/viewer.json`. Called by `start.sh`. |
| `Caddyfile.example` | Proxy configuration: port 8081 is the driving viewer (and the dashboard under `/dashboard/`), port 8082 is the perception demo, both behind a login. |
| `live_run.sh` | Starts, checks or stops a run that shows on the viewer page under "Live run". |
| `run_one.sh` | Runs one Town02 scenario by number as a live run. |
| `batch_eval.sh` | Evaluates scenarios one process at a time, so a crash in one does not lose the others. Used for the 32-run re-evaluation. |
| `gen_scenarios.py` | Picked the ten added Town02 scenarios (7 to 16): tries seeds and keeps those with a new route where every encounter can be placed. |
| `report.py` | Builds the evaluation summary (`../town02_evaluation_summary.md`) from the per-scenario run folders. |
| `viewer_server_edits.patch` | Changes made to the viewer on the server only (see below). |

## Proxy login

Copy `Caddyfile.example` to `~/demo/Caddyfile`, then replace:

- `PASSWORD_HASH` (two places) with the output of `caddy hash-password --plaintext 'your-password'`. The user name is `team`.
- `PUBLIC_IP:VIEWER_PORT` with the public address Vast gives for port 8081. The proxy rejects requests whose origin is any other site.

## Server-only viewer changes

`viewer_server_edits.patch` is the diff of edits made on the server. They are not part of the code in `dreamerv3_attention/`; apply them with `git apply deployment/vast/viewer_server_edits.patch`.

- The camera frame is sent at 640x360, JPEG quality 60, and the page polls every 400 ms and skips frames while one is still loading. This keeps the feed usable over a slow link.
- A destination drive uses the configured device instead of being forced onto the CPU.
- The start and destination boxes are filled only when the server reports valid values.
- The evaluation dashboard is shown at the bottom of the viewer page, served from `~/demo/dash`.

## Notes from this instance

- `start.sh` leaves out `-quality-level=Low`. On this instance that flag crashed the packaged CARLA 0.9.16 server when RGB and depth cameras ran in synchronous mode.
- The perception demo runs on the CPU because the driving agent refuses to start while another process is using the GPU.
- The `taskset` core ranges in `start.sh` assume at least 24 CPU cores; change them for a different machine.
- A reboot stops every `tmux` session. Run `~/demo/start.sh` again afterwards.

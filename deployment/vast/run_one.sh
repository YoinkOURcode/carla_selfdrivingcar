#!/bin/bash
# Run ONE Town02 scenario as a live run on the viewer page (run as user carla).
#   ~/demo/run_one.sh 2        runs held_out_long_mixed_2
N="$1"
cd ~/carla_selfdrivingcar || exit 1
CFG="final/batch/held_out_long_mixed_$N.json"
if [ ! -f "$CFG" ]; then echo "No Town02 scenario $N. Use a number from 1 to 16."; exit 1; fi
if tmux has-session -t job 2>/dev/null; then echo "A job is already running. Use: ~/demo/live_run.sh stop"; exit 1; fi
STATE=$(curl -s -m 5 http://127.0.0.1:8080/api/state | venv/bin/python -c "import json,sys; print(json.load(sys.stdin).get(\"status\"))" 2>/dev/null)
if [ "$STATE" = driving ] || [ "$STATE" = starting ]; then echo "A Destination drive is running. Press Stop on the page first."; exit 1; fi
tmux kill-session -t viewer 2>/dev/null; sleep 2
~/demo/start.sh >/dev/null 2>&1
OUT="runs/live_town02_s${N}_$(date +%Y%m%d_%H%M%S)"
tmux new -d -s job "cd ~/carla_selfdrivingcar && venv/bin/python -m dreamerv3_attention infer --config $CFG --checkpoint final/latest.pt --budget-seconds 1500 --output $OUT > ~/demo/job.log 2>&1; echo EXIT=\$? >> ~/demo/job.log"
echo "Started Town02 scenario $N. Watch it on the viewer page under Live run (page is down for about a minute first)."

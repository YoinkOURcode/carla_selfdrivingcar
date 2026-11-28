#!/bin/bash
# Start a job that shows up on the viewer page under "Live run" (run as user carla).
#   ~/demo/live_run.sh infer      one scenario (about 7 minutes)
#   ~/demo/live_run.sh evaluate   all six Town02 scenarios (about 45 minutes)
#   ~/demo/live_run.sh status     show whether a job is running and its last output
#   ~/demo/live_run.sh stop       stop the job
MODE="${1:-infer}"
cd ~/carla_selfdrivingcar || exit 1
case "$MODE" in
  status) tmux has-session -t job 2>/dev/null && echo "A job is running." || echo "No job is running."; tail -5 ~/demo/job.log 2>/dev/null; exit 0;;
  stop) tmux kill-session -t job 2>/dev/null && echo "Job stopped." || echo "No job was running."; tmux kill-session -t viewer 2>/dev/null; ~/demo/start.sh >/dev/null 2>&1; exit 0;;
  infer) BUDGET=1500;;
  evaluate) BUDGET=7200;;
  *) echo "Use: infer, evaluate, status or stop"; exit 1;;
esac
if tmux has-session -t job 2>/dev/null; then echo "A job is already running. Use: ~/demo/live_run.sh stop"; exit 1; fi
STATE=$(curl -s -m 5 http://127.0.0.1:8080/api/state | venv/bin/python -c "import json,sys; print(json.load(sys.stdin).get(\"status\"))" 2>/dev/null)
if [ "$STATE" = driving ] || [ "$STATE" = starting ]; then echo "A Destination drive is running. Wait for it to finish or press Stop on the page first."; exit 1; fi
# The viewer keeps GPU memory after a drive and the project then refuses to start a job, so restart it.
tmux kill-session -t viewer 2>/dev/null; sleep 2
~/demo/start.sh >/dev/null 2>&1
OUT="runs/live_${MODE}_$(date +%Y%m%d_%H%M%S)"
tmux new -d -s job "cd ~/carla_selfdrivingcar && venv/bin/python -m dreamerv3_attention $MODE --config final/evaluation.json --checkpoint final/latest.pt --budget-seconds $BUDGET --output $OUT > ~/demo/job.log 2>&1; echo EXIT=\$? >> ~/demo/job.log"
echo "Started $MODE. Watch it on the viewer page under Live run. Results will be in ~/carla_selfdrivingcar/$OUT"

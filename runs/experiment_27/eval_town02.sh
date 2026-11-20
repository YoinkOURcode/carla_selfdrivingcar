#!/usr/bin/env bash
# Usage: eval_town02.sh <name> <checkpoint|controller>. Requires idle Town02.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
N=runs/experiment_27/eval_town02
O="${EVAL_OUTPUT_ROOT:-runs/vision_integration/town02}/results_${1:?Missing run name}"
[[ ! -e "$O" ]] || { echo "Output already exists: $O" >&2; exit 1; }
mkdir -p "$O/configs"
if [[ "${2:?Missing checkpoint or controller}" == controller ]]; then
  who=(--controller)
else
  who=(--checkpoint "$2")
fi
status=0
for v in clear wet night lowvis; do
  venv/bin/python - "$N/eval_$v.json" "$O/configs/$v.json" <<'PY'
import json
import os
from pathlib import Path
import sys

cfg = json.loads(Path(sys.argv[1]).read_text())
cfg.update(perception="external", provider="dreamerv3_attention.vision:VisionPerceptionProvider",
           port=int(os.environ.get("CARLA_PORT", cfg["port"])))
cfg["driving"]["default_speed_mps"] = 30.0 / 3.6
# No inferred green phase/traffic-light identity is available from these models.
cfg["driving"]["restart_assist_enabled"] = False
Path(sys.argv[2]).write_text(json.dumps(cfg, indent=2) + "\n")
PY
  if ! venv/bin/python -m dreamerv3_attention evaluate --config "$O/configs/$v.json" \
      "${who[@]}" --output "$O/$v" --budget-seconds "${EVAL_BUDGET_SECONDS:-1500}" > "$O/$v.log" 2>&1; then
    echo "$v" >> "$O/FAILED"
    status=1
  fi
done
if (( status == 0 )); then echo done > "$O/ALL_DONE"; fi
exit "$status"

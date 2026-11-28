#!/bin/bash
# Evaluate scenarios one at a time (a crash in one does not lose the others).
#   ~/demo/batch_eval.sh OUTDIR name1 name2 ...
cd ~/carla_selfdrivingcar || exit 1
OUT="$1"; shift
mkdir -p "$OUT" final/batch
for NAME in "$@"; do
  venv/bin/python - "$NAME" <<PY
import json, sys
from pathlib import Path
cfg = json.loads(Path("final/evaluation.json").read_text())
cfg["scenarios"] = [sys.argv[1]]
Path("final/batch/%s.json" % sys.argv[1]).write_text(json.dumps(cfg, indent=2) + "\n")
PY
  venv/bin/python ~/demo/load_map.py > "$OUT/$NAME.log" 2>&1
  echo "$(date +%H:%M:%S) START $NAME" >> "$OUT/progress.log"
  venv/bin/python -m dreamerv3_attention evaluate --config "final/batch/$NAME.json" --checkpoint final/latest.pt \
    --budget-seconds 1500 --output "$OUT/$NAME" >> "$OUT/$NAME.log" 2>&1
  echo "$(date +%H:%M:%S) END $NAME exit=$?" >> "$OUT/progress.log"
done
echo "$(date +%H:%M:%S) ALL DONE" >> "$OUT/progress.log"

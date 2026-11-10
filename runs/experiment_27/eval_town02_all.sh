#!/usr/bin/env bash
# Frozen RGB perception: RouteController and run_10, four Town02 conditions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-runs/vision_integration/town02_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$EVAL_OUTPUT_ROOT" ]]; then
  echo "Output already exists: $EVAL_OUTPUT_ROOT; select a fresh EVAL_OUTPUT_ROOT" >&2
  exit 1
fi
mkdir -p "$EVAL_OUTPUT_ROOT"
status=0
bash runs/experiment_27/eval_town02.sh controller controller || status=1
bash runs/experiment_27/eval_town02.sh run_10 runs/experiment_26/curriculum/stage7_run_10/latest.pt || status=1
if (( status == 0 )); then echo done > "$EVAL_OUTPUT_ROOT/ALL_DONE"; fi
echo "Vision evaluation results: $EVAL_OUTPUT_ROOT"
exit "$status"

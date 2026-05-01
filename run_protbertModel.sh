#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

LOG=logs/protbert_model.log
: > "$LOG"   # truncate so each pipeline run starts fresh

# -u: unbuffered Python stdout/stderr (real-time terminal + log)
# stdbuf -oL -eL: line-buffered tee (avoid pipe-buffer batching)
# tee -a: append, so all four phases land in the same single log
export PYTHONUNBUFFERED=1

run() {
  local phase=$1; shift
  printf '\n========== [%s] %s ==========\n' "$phase" "$(date -Iseconds)" | tee -a "$LOG"
  stdbuf -oL -eL python -u "$@" 2>&1 | stdbuf -oL -eL tee -a "$LOG"
}

run TUNE     protbert_model/tune.py
run TRAIN    protbert_model/train.py
run TEST     protbert_model/test.py
run ANALYZE  protbert_model/analyze.py

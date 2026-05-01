#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

LOG=logs/esm_model.log
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

run TUNE     esm_model/tune.py
run TRAIN    esm_model/train.py
run TEST     esm_model/test.py
run ANALYZE  esm_model/analyze.py

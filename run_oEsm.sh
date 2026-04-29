#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p logs
python o_esm/tune.py  2>&1 | tee logs/o_esm_tune.log
python o_esm/train.py 2>&1 | tee logs/o_esm_train.log
python o_esm/test.py  2>&1 | tee logs/o_esm_test.log

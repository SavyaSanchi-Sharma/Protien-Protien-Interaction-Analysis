#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p logs
python esm_model/tune.py  2>&1 | tee logs/esm_model_tune.log
python esm_model/train.py 2>&1 | tee logs/esm_model_train.log
python esm_model/test.py  2>&1 | tee logs/esm_model_test.log

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p logs
python protbert_model/tune.py  2>&1 | tee logs/protbert_model_tune.log
python protbert_model/train.py 2>&1 | tee logs/protbert_model_train.log
python protbert_model/test.py  2>&1 | tee logs/protbert_model_test.log

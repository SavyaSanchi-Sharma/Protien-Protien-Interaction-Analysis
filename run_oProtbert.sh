#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p logs
python o_protbert/tune.py  2>&1 | tee logs/o_protbert_tune.log
python o_protbert/train.py 2>&1 | tee logs/o_protbert_train.log
python o_protbert/test.py  2>&1 | tee logs/o_protbert_test.log

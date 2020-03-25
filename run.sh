#!/bin/bash

EXPORT_DIR="../export_dir"
DATA_ROOT="../data_debug"
BATCH_SIZE=8

python code/main.py --export-dir $EXPORT_DIR --sigmorphon2020-root $DATA_ROOT --debug --no-gpu --batch-size $BATCH_SIZE --language-families niger-congo

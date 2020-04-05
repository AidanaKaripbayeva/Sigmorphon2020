#!/bin/bash

module load gcc/5.3.0 cmake/3.9.4 cudatoolkit/9.1.85_3.10-1.0502.df1cc54.3.1 cray-libsci/18.12.1 
source /projects/eot/bbcj/bjlunt2/opt/miniconda3/bin/activate
source activate torch

EXPORT_DIR="../export_dir"
DATA_ROOT="../task0-data"
BATCH_SIZE=8

which python

aprun -d 8 -n1 -N1 python code/main.py --export-dir $EXPORT_DIR --sigmorphon2020-root $DATA_ROOT --batch-size $BATCH_SIZE --languages eng \
--language-info-file ./code/data/languages/individual_alphabets.tsv \
--model dummy \
--adadelta-lr 5.0 

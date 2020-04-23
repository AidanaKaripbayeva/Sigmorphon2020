#!/bin/bash

#THis is for GPU nodes.
#PBS -l nodes=1:ppn=16:xk,walltime=5:00:00
#PBS -q normal

module load gcc/5.3.0 cmake/3.9.4 cudatoolkit/9.1.85_3.10-1.0502.df1cc54.3.1 cray-libsci/18.12.1
source /projects/eot/bbcj/bjlunt2/opt/miniconda3/bin/activate
#source activate torch


EXPORT_DIR="${HOME}/scratch/TURKS/$PBS_JOBID"
DATA_ROOT="../task0-data"
BATCH_SIZE=20

mkdir -p ${EXPORT_DIR}
export OMP_NUM_THREADS=${PBS_NUM_PPN}

which python

cd ${PBS_O_WORKDIR}

module load ccm
ccmrun python code/main.py --export-dir $EXPORT_DIR --sigmorphon2020-root $DATA_ROOT --batch-size $BATCH_SIZE --languages eng \
--language-info-file ./code/data/languages/individual_alphabets.tsv \
--model baby-transducer \
--adadelta-lr 5.0 \
--checkpoint-step 100

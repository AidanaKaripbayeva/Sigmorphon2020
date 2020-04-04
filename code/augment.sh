#!/bin/bash
FILE=~/Documents/UIUC/2020_Spring/CS546/sigmorphon2020/TurkicSigmorphon2020/code
LANG_PATH=~/Documents/UIUC/2020_Spring/CS546/sigmorphon2020/neural-transducer/task0-data-master

for lng in mlg ceb hil tgl mao; do
python3 ${FILE}/main_augment.py ${LANG_PATH} austronesian $lng --examples 10000
done

for lng in dan isl nob swe nld eng deu gmh frr ang; do
python3 ${FILE}/main_augment.py ${LANG_PATH} germanic $lng --examples 10000
done

for lng in nya kon lin lug sot swa zul aka gaa; do
python3 ${FILE}/main_augment.py ${LANG_PATH} niger-congo $lng --examples 10000
done

for lng in cpa azg xty zpv ctp czn cly otm ote pei; do
python3 ${FILE}/main_augment.py ${LANG_PATH} oto-manguean $lng --examples 10000
done

for lng in est fin izh krl liv vep vot mhr myv mdf sme; do
python3 ${FILE}/main_augment.py ${LANG_PATH} uralic $lng --examples 10000
done

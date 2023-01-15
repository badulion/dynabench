#!/bin/sh

EQUATION=$1
MODEL=$2
TASK=$3
SUPPORT=$4

cp -r data_ceph/$EQUATION/ data/
python main.py --equation $EQUATION --model $MODEL --task $TASK --support $SUPPORT
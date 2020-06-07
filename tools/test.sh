#!/bin/bash
CONFIG_FILE=$1
TASK_DESC=$2
MODEL=$3

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

CUDA_VISIBLE_DEVICES=2 python ./tools/test.py $CONFIG_FILE $TASK_DESC/epoch\_$MODEL.pth --out=$TASK_DESC/det.pkl \
--eval=bbox --options classwise=True include=True
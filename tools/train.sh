#!/bin/bash
CONFIG_FILE=$1
TASK_DESC=$2
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR='/Extra/zhangmeng/face_mask_detection/AIZOO/output'

PP_WORK_DIR=$OUT_DIR/$TASK_DESC\_$DATE_WITH_TIME

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

CUDA_VISIBLE_DEVICES=0 python ./tools/train.py $CONFIG_FILE --work-dir=$PP_WORK_DIR 
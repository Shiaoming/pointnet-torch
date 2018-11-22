#!/bin/bash

#ROOT=.
#export PYTHONPATH=$ROOT:$PYTHONPATH
#export | grep PYTHONPATH

#tensorboard --logdir=./runs --port=8008 & # $: for background running

python train.py  \
            --batch_size=12 \
            --base_dir=../data/ \
            --datadir=../data/indoor3d_sem_seg_hdf5_data \
            --save_dir=./checkpoints \
            --summary_dir=./runs \
            --resume=None  \
            --gpu_id=-1  \
            --epochs=50 \
            --start_epoch=0   \
            --save_every_epoch=1\
            --learning-rate=0.001   \
            --lr_decay=0.5 \
            --lr_decay_step=30 \
            --weight_decay=1e-4    \
            --test_area=Area_1_conferenceRoom_1    \
            --part_num=13

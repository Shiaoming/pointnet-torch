#!/bin/bash

#ROOT=.
#export PYTHONPATH=$ROOT:$PYTHONPATH
#export | grep PYTHONPATH

#tensorboard --logdir=./runs --port=8008 & # $: for background running

python train.py  \
            --batch_size=16 \
            --base_dir=../data/hdf5_data/ \
            --datadir=../data/hdf5_data \
            --save_dir=./checkpoints \
            --summary_dir=./runs \
            --resume=./checkpoints/checkpoint_e332.pth  \
            --gpu_id=1  \
            --epochs=500 \
            --start_epoch=0   \
            --learning-rate=0.001   \
            --lr_decay=0.5 \
            --lr_decay_step=30 \
            --weight_decay=1e-4    \
            --cat_num=16    \
            --part_num=50

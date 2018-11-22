#!/bin/bash

#ROOT=.
#export PYTHONPATH=$ROOT:$PYTHONPATH
#export | grep PYTHONPATH

python train.py  \
            --batch_size=32 \
            --base_dir=../ \
            --datadir=../data/modelnet40_ply_hdf5_2048 \
            --save_dir=./checkpoints_test \
            --summary_dir=../runs_test \
            --resume=None  \
            --gpu_id=0  \
            --epochs=300 \
            --num_points=1024 \
            --start_epoch=0   \
            --learning-rate=0.001   \
            --lr_decay=0.7 \
            --lr_decay_step=30 \
            --weight_decay=1e-4    \
            --rotation_option=Rotate \
            --input_trans=1    \
            --feature_trans=1

tensorboard --logdir=../runs_test --port=8008
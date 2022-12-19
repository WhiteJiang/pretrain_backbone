#!/bin/bash

export PYTHONPATH=/home/jx/code/pretain_backbone:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 python tools/cub_train.py

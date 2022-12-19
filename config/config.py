# -*- coding: utf-8 -*-
# @Time    : 2022/12/19
# @Author  : White Jiang
from yacs.config import CfgNode as CN
import os

_C = CN()

_C.MODEL = CN()
_C.MODEL.USE_CHECKPOINT = True
# Model's Mode
_C.MODEL.MODE = 'train'
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0, 1'
# Name of backbone
_C.MODEL.NAME = 'convnext_small'
_C.MODEL.INFO = 'pretrain_cub'

# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = '/home/jx/code/pretrain_backbone/resnet18.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 224
# Size of the image during test

# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('CUB')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/home/jx/dataset/CUB_200_2011/')
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 128

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = 'Adam'
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 300
# Base learning rate
_C.SOLVER.BASE_LR = 0.001
# Momentum
_C.SOLVER.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# lr_scheduler
# lr_scheduler method, option WarmupMultiStepLR, WarmupCosineAnnealingLR
_C.SOLVER.LR_NAME = 'WarmupCosineAnnealingLR'
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = [40, 70]

# Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 30
_C.SOLVER.ETA_MIN_LR = 1e-7

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.1
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = _C.SOLVER.MAX_EPOCHS
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 50
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = _C.SOLVER.MAX_EPOCHS

_C.SOLVER.IMS_PER_BATCH = 128
_C.SOLVER.SEED = 42

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128

# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./log"
if not os.path.isdir(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)

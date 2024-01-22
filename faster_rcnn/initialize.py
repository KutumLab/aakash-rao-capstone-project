from turtle import colormode
import pandas as pd
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

from utils.MyTrainer import MyTrainer

import sys



# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader
fold = 1

# data_path = f'/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_project/object-detection/benchmarking/datasets/NuCLS/folds/fold_1/'
# config_info = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# max_iters = 1500
# name  = 'exp_1_iters_1'
# project = 'capstone-project' 
# version = '1'



def set_config(config_info, fold, max_iters, batch_size, name,save_path, version):
    max_iters = int(max_iters)
    batch_size = int(batch_size)
    cfg = get_cfg()
    if 'COCO' in config_info:
        cfg.merge_from_file(model_zoo.get_config_file(config_info))
    else:
        cfg.merge_from_file(config_info)
    cfg.DATASETS.TRAIN = (f'fold_{fold}_train',)
    cfg.DATASETS.TEST = (f'fold_{fold}_val',)
    cfg.TEST.EVAL_PERIOD = max_iters//150
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_info)
    cfg.MODEL.LOAD_PROPOSALS = False
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = max_iters
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 if version == 'single' else (3 if version == 'three_class' else 4)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.OUTPUT_DIR = os.path.join(save_path, f'detectron/{name}_fold_{fold}')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # trainer.train()
    return cfg

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data_path', type=str, default='/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/detectron', help='path to data')
    argparse.add_argument('--config_info', type=str, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help='config info')
    argparse.add_argument('--max_iters', type=int, default=1500, help='max iters')
    argparse.add_argument('--batch_size', type=str, default='capstone-project', help='project')
    argparse.add_argument('--name', type=str, default='faster_rcnn_R_50_FPN_3x', help='name')
    argparse.add_argument('--fold', type=str, default=1, help='version')
    argparse.add_argument('--version', type=str, default='', help='version')
    argparse.add_argument('--save_path', type=str, default='/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs', help='save path')
    args = argparse.parse_args()
    cfg = set_config(args.config_info, args.fold, args.max_iters, args.batch_size, args.name, args.save_path, args.version)
    # cache the model 
    
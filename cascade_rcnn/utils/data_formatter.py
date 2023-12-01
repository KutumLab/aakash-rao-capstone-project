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
import sys


from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
fold = 1



def generate_coco_dataset(data_path, fold):
    def data_train():
        data = np.load(os.path.join(data_path, f'fold_{fold}', 'train.npy'), allow_pickle=True)
        data = list(data)
        print(f'Number of training images: {len(data)}')
        return data

    def data_val():
        data = np.load(os.path.join(data_path, f'fold_{fold}', f'val.npy'), allow_pickle=True)
        data = list(data)
        print(f'Number of validation images: {len(data)}')
        return data

    def data_test():
        data = np.load(os.path.join(data_path,f'test.npy'), allow_pickle=True)
        data = list(data)
        print(f'Number of test images: {len(data)}')
        return data
    
    try:
        DatasetCatalog.register(f'fold_{fold}_train', data_train)
        MetadataCatalog.get(f'fold_{fold}_train').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other']
        MetadataCatalog.get(f'fold_{fold}_train').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]
    except:
        pass
    
    try:
        DatasetCatalog.register(f'fold_{fold}_val', data_val)
        MetadataCatalog.get(f'fold_{fold}_val').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other']
        MetadataCatalog.get(f'fold_{fold}_val').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]
    except:
        pass

    try:
        DatasetCatalog.register(f'test', data_test)
        MetadataCatalog.get(f'test').thing_classes = ['nonTIL_stromal','sTIL','tumor_any','other']
        MetadataCatalog.get(f'test').thing_colors = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]
    except:
        pass

    output_dir = os.path.join(args.output_dir, f'fold_{fold}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # register_coco_instances(f'fold_{fold}_train', {}, os.path.join(output_dir, f'train.json'), output_dir)
    # register_coco_instances(f'fold_{fold}_val', {}, os.path.join(output_dir, f'val.json'), output_dir)
    # register_coco_instances(f'test', {}, os.path.join(output_dir, f'test.json'), output_dir)
    convert_to_coco_json(f'fold_{fold}_train', os.path.join(output_dir, f'train.json'), allow_cached=True)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Coco-format dataset using Detectron2')
    parser.add_argument('--data_path', type=str, help='path to data', required=True)
    parser.add_argument('--output_dir', type=str, help='path to output directory', required=True)
    parser.add_argument('--fold', type=int, default=1, help='fold to train on', required=False)
    args = parser.parse_args()
    generate_coco_dataset(args.data_path, args.fold)
    print('Done')
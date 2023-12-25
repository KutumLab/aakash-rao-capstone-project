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






def train_detectron2(cfg,fold,data_path, version=""):
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

    DatasetCatalog.register(f'fold_{fold}_train', data_train)
    if version =="four_class":
        classes = ['nonTIL_stromal','sTIL','tumor_any','other']
        colour_arr = [(161,9,9),(239,222,0),(22,181,0),(0,32,193),(115,0,167)]
    elif version == "three_class":
        classes = ['nonTIL_stromal','sTIL','tumor_any']
        colour_arr = [(161,9,9),(239,222,0),(22,181,0),(0,32,193)]
    else:
        classes = ['cell']
        colour_arr = [(161,9,9)]
    MetadataCatalog.get(f'fold_{fold}_train').thing_classes = classes
    MetadataCatalog.get(f'fold_{fold}_train').thing_colors = colour_arr


    DatasetCatalog.register(f'fold_{fold}_val', data_val)
    MetadataCatalog.get(f'fold_{fold}_val').thing_classes = classes
    MetadataCatalog.get(f'fold_{fold}_val').thing_colors = colour_arr

    DatasetCatalog.register(f'test', data_test)
    data = DatasetCatalog.get(f'test')
    MetadataCatalog.get(f'test').thing_classes = classes
    MetadataCatalog.get(f'test').thing_colors = colour_arr

    dataset_dicts = DatasetCatalog.get(f'fold_{fold}_train')
    metadata = MetadataCatalog.get(f'fold_{fold}_train')

    i = 0
    for d in tqdm (range(10), desc="Sample Training Images...", ascii=False, ncols=75):
        num = random.randint(i*len(dataset_dicts)//10, (i+1)*len(dataset_dicts)//10) -1
        d = dataset_dicts[num]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        im = vis.get_image()[:, :, ::-1]
        plt.imshow(im)
        train_img_path = os.path.join(cfg.OUTPUT_DIR, f'train_images')
        if not os.path.exists(train_img_path):
            os.makedirs(train_img_path)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(train_img_path, f'{i}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        # cv2.waitKey(0)
        # break
        i+=1

    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    history = trainer.train()
    # evaluator = DefaultTrainer.build_evaluator(trainer)

    print(history)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.DATASETS.TEST = (f'test',)

    predictor = DefaultPredictor(cfg)
    predictions = []
    pred_save_path = os.path.join(cfg.OUTPUT_DIR, 'predictions')
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    # for d in data_test():
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     predictions.append(outputs)
    #     v = Visualizer(im[:, :, ::-1],
    #                     metadata=MetadataCatalog.get(f'test'), 
    #                     scale=0.8,
    #     )
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.imshow(out.get_image()[:, :, ::-1])
    #     plt.axis('off')
    #     plt.savefig(os.path.join(pred_save_path, d['file_name'].split('/')[-1]), bbox_inches='tight', pad_inches=0, dpi=300)
    # print('Predictions: ', predictions)
    # predictions = np.array(predictions)
    # np.save(os.path.join(cfg.OUTPUT_DIR, 'predictions.npy'), predictions)

    evaluator = COCOEvaluator("test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "test")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)

    # OrderedDict to dict
    results = dict(results)
    # save results
    results_save_path = os.path.join(cfg.OUTPUT_DIR, 'results')
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    np.save(os.path.join(results_save_path, 'results.npy'), results)
    print(results)
    return results



    # experiment.end()


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
    results = train_detectron2(cfg, args.fold, args.data_path, args.version)
    print(results)
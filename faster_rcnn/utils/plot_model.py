import json
import argparse
import os
import numpy as np
import pandas as pd

cols = ['bbox/AP', 'bbox/AP-nonTIL_stromal', 'bbox/AP-other', 'bbox/AP-sTIL',
       'bbox/AP-tumor_any', 'bbox/AP50', 'bbox/AP75', 'bbox/APl', 'bbox/APm',
       'bbox/APs', 'data_time', 'eta_seconds', 'fast_rcnn/cls_accuracy',
       'fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy', 'iteration',
       'loss_box_reg', 'loss_cls', 'loss_rpn_cls', 'loss_rpn_loc', 'lr',
       'rank_data_time', 'roi_head/num_bg_samples', 'roi_head/num_fg_samples',
       'rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'time', 'timetest',
       'total_loss', 'validation_loss']

relevant_cols = ['bbox/AP', 'bbox/AP-nonTIL_stromal', 'bbox/AP-other', 'bbox/AP-sTIL','bbox/AP-tumor_any', 'bbox/AP50', 'bbox/AP75', 'iteration', 'loss_box_reg', 'loss_cls', 'total_loss', 'validation_loss']

translations_arr = ['mAP', 'map0', 'map1', 'map2','map3', 'mAP50', 'mAP75', 'iteration', 'loss_box_reg', 'loss_cls', 'total_loss', 'validation_loss']

titles_arr = ['mean AP', 'mean AP for nonTIL_stromal', 'mean AP for other', 'mean AP for sTIL','mean AP for tumor_any', 'mean AP at IoU 50', 'mean AP at IoU 75', 'iteration', 'Box Loss', 'Class Loss', 'Total Loss', 'Validation Loss']

def plot_model(path):
    # read json file
    info = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=False, header=0)
    info = info[relevant_cols]
    info = info.rename(columns=dict(zip(translations_arr, translations_arr)))
    info_cols = info.columns
    print(info_cols)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    plot_model(args.path)
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
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

axis_arr = ['mAP', 'mAP', 'mAP', 'mAP','mAP', 'mAP50', 'mAP75', 'Iterations', 'Loss', 'Loss', 'Loss', 'Loss']

titles_arr = ['mean AP', 'mean AP for Stromal', 'mean AP for Other', 'mean AP for sTIL','mean AP for Tumor', 'mean AP at IoU 50', 'mean AP at IoU 75', 'iteration', 'Box Loss', 'Class Loss', 'Total Loss', 'Validation Loss']
title_dict = dict(zip(translations_arr, titles_arr))
axis_dict = dict(zip(translations_arr, axis_arr))
x_axis = 'iteration'

def plot_model(path):
    # read json file
    info = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=False, header=0)
    info = info[relevant_cols]
    info = info.rename(columns=dict(zip(relevant_cols, translations_arr))) 
    info_cols = info.columns
    print(info_cols)    
    plot_save_path = os.path.join(path, 'plots')
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    for col in info_cols:
        if col == 'iteration':
            continue
        plt.figure(figsize=(4, 4))
        plt.locator_params(nbins=5)
        copy_info = info.copy()
        copy_info = copy_info[[x_axis, col]].dropna(axis=0, how='any')
        if col =='loss_cls':
            copy_info[col] = copy_info[col]
        if 'map' in col.lower():
            copy_info[col] = copy_info[col]/100
        plt.plot(copy_info[x_axis], copy_info[col], linewidth=1)
        plt.title(title_dict[col], fontsize=14, fontweight='bold')
        plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
        plt.ylabel(axis_dict[col], fontsize=14, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim(0, copy_info[x_axis].max()+10)
        # check if mAP is a substring of the column name
        if 'map' in col.lower():
            plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_save_path, col + '.png'), dpi=300)

            

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    plot_model(args.path)
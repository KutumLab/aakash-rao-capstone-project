import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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

model_list = ['faster_rcnn_R_50_C4_1x','faster_rcnn_R_50_C4_3x','faster_rcnn_R_50_DC5_1x','faster_rcnn_R_101_DC5_3x']
name_key = {'faster_rcnn_R_50_C4_1x': 'FaR-CNN R50 C4 1x', 'faster_rcnn_R_50_C4_3x': 'FaR-CNN R50 C4 3x', 'faster_rcnn_R_50_DC5_1x': 'FaR-CNN R50 DC5 1x', 'faster_rcnn_R_101_DC5_3x': 'FaR-CNN R101 DC5 3x'}

def plot_model_individual(src_path, phase):
    if not os.path.exists(src_path):
        print("File not found: ", src_path)
        raise FileNotFoundError(src_path)
    elif len(os.listdir(src_path)) == 0:
        print("Empty folder: ", src_path)
        raise FileNotFoundError(src_path)
    else:
        for folder in model_list:
            model_dict = {}
            for folds in range (1,4):
                fold_path = os.path.join(src_path, folder+f"_fold_{folds}")
                output_path = os.path.join(src_path, folder, "plots")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                metrics = pd.read_csv(os.path.join(fold_path, "metrics.csv"))
                print(fold_path)
                metrics = metrics[relevant_cols]
                metrics = metrics.rename(columns=dict(zip(relevant_cols, translations_arr)))

                for key in translations_arr:
                    if key == 'iteration':
                        continue
                    small_metrics = metrics[['iteration', key]]
                    # drop rows with nan
                    small_metrics = small_metrics.dropna(axis=0, how='any')
                    # print(small_metrics)
                    plt.figure(figsize=(3,3))
                    plt.locator_params(nbins=5)
                    if 'map' in key.lower():
                        plt.plot(small_metrics['iteration'], small_metrics[key]/100, color='#0000FF', linewidth=1)
                    else:
                        plt.plot(small_metrics['iteration'], small_metrics[key], color='#0000FF', linewidth=1)
                    plt.title(f'{title_dict[key]}\nfor {folder}', fontsize=14, fontweight='bold')
                    plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
                    plt.ylabel(axis_dict[key], fontsize=14, fontweight='bold')
                    if 'map' in key.lower():
                        plt.ylim(0, 1)
                    
                    plt.tight_layout()
                    # plt.savefig(os.path.join(output_path, key + f"_fold_{folds}.png"), dpi=300)
                    plt.close()
                
                model_dict[folder+f"_fold_{folds}"] = metrics
                print(metrics.head())
            
            
            model_csv = []
            for key in translations_arr:
                metric_sum = numpy.array([])
                is_nan_list = []
                for model in model_dict.keys():
                    # finding mean and std
                    is_nan_list = model_dict[model][key].isnull()
                    metric_sum = numpy.append(metric_sum, model_dict[model][key].dropna(axis=0, how='any').values)
                    # which indixes were dropped
                metric_sum = metric_sum.reshape(3, -1)
                mean = numpy.mean(metric_sum, axis=0)
                std = numpy.std(metric_sum, axis=0)
                plt.figure(figsize=(3,3))
                plt.locator_params(nbins=5)
                if 'map' in key.lower():
                    plt.plot((numpy.array(range(0, len(mean),1))*100), mean/100, color='#0000FF', linewidth=1)
                    plt.errorbar((numpy.array(range(0, len(mean),1))*100)[::5], mean[::5]/100, yerr=std[::5]/100, capsize=1, capthick=1, elinewidth=1, color='black', linewidth=0)
                else:
                    plt.plot((numpy.array(range(0, len(mean),1))*100), mean, color='#0000FF', linewidth=1)
                    plt.errorbar((numpy.array(range(0, len(mean),1))*100)[::5], mean[::5], yerr=std[::5], capsize=1, capthick=1, elinewidth=1, color='black', linewidth=0)
                plt.title(f'{title_dict[key]}\nfor {name_key[folder]}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
                plt.ylabel(axis_dict[key], fontsize=14, fontweight='bold')
                if 'map' in key.lower():
                    plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, folder + "_" + key + "_mean.png"), dpi=300)
                plt.close()

                col1 = 'mAP'
                col2 = 'mAP50'
                plt.figure(figsize=(3, 3))
                plt.locator_params(nbins=5)
                copy_info = model_dict[model][[x_axis, col1, col2]].dropna(axis=0, how='any')
                plt.plot(copy_info[x_axis], copy_info[col1]/100, linewidth=1, label='mAP')
                plt.plot(copy_info[x_axis], copy_info[col2]/100, linewidth=1, label='mAP50')
                plt.title(f'mAP for \n{name_key[folder]}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
                plt.ylabel('mAP', fontsize=14, fontweight='bold')
                plt.legend()
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, folder+ "_" + 'mAP_vs_mAP50.png'), dpi=300)
                copy_info.to_csv(os.path.join(output_path, 'mAP_vs_mAP50.csv'), index=False)
                plt.close()

                if phase == "testing":
                    break 

            if phase == "testing":
                return

def plot_model_individual_with_collective(path, phase):
    plot_path = os.path.join(path)
    map50_arr = numpy.array([])
    map_arr = numpy.array([])
    plt.figure(figsize=(3,3))
    for folder in model_list:
        src_path = os.path.join(path, folder,"plots")
        map_file = pd.read_csv(os.path.join(src_path, 'mAP_vs_mAP50.csv'))
        print(map_file.head())
        plt.plot(map_file['iteration'], map_file['mAP']/100, linewidth=1, label=name_key[folder])
    plt.title(f'mAP50:95 across\nall RetinaNets', fontsize=14, fontweight='bold')
    plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
    plt.ylabel('mAP', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'mAP50_95.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(3,3))
    for folder in model_list:
        src_path = os.path.join(path, folder,"plots")
        map_file = pd.read_csv(os.path.join(src_path, 'mAP_vs_mAP50.csv'))
        print(map_file.head())
        plt.plot(map_file['iteration'], map_file['mAP50']/100, linewidth=1, label=name_key[folder])
    plt.title(f'mAP50 across\nall RetinaNets', fontsize=14, fontweight='bold')
    plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
    plt.ylabel('mAP', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'mAP50.png'), dpi=300)
    plt.close()


   
        


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    plot_model_individual(args.path, args.phase)
    plot_model_individual_with_collective(args.path, args.phase)
    # give_stats(args.path)

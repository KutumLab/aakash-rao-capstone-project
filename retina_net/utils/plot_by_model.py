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

model_list = ['retinanet_R_50_FPN_1x','retinanet_R_50_FPN_3x','retinanet_R_101_FPN_3x']
name_key = {'retinanet_R_50_FPN_1x': 'RetinaNet R50 1x', 'retinanet_R_50_FPN_3x': 'RetinaNet R50 3x', 'retinanet_R_101_FPN_3x': 'RetinaNet R101 3x'}

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
                    plt.savefig(os.path.join(output_path, key + f"_fold_{folds}.png"), dpi=300)
                
                model_dict[folder+f"_fold_{folds}"] = metrics
                print(metrics.head())
            
            model_csv = []
            for key in translations_arr:
                metric_sum = numpy.array([])
                for model in model_dict.keys():
                    # finding mean and std
                    metric_sum = numpy.append(metric_sum, model_dict[model][key].dropna(axis=0, how='any').values)
                metric_sum = metric_sum.reshape(3, -1)
                mean = numpy.mean(metric_sum, axis=0)
                std = numpy.std(metric_sum, axis=0)
                plt.figure(figsize=(3,3))
                plt.locator_params(nbins=5)
                if 'map' in key.lower():
                    plt.plot(model_dict[model]['iteration'], mean/100, color='#0000FF', linewidth=1)
                    plt.errorbar(model_dict[model]['iteration'][::5], mean[::5]/100, yerr=std[::5]/100, capsize=1, capthick=1, elinewidth=1, color='black', linewidth=0)
                else:
                    plt.plot(model_dict[model]['iteration'], mean, color='#0000FF', linewidth=1)
                    plt.errorbar(model_dict[model]['iteration'][::5], mean[::5], yerr=std[::5], capsize=1, capthick=1, elinewidth=1, color='black', linewidth=0)
                plt.title(f'{title_dict[key]}\nfor {name_key{folder}}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_dict[x_axis], fontsize=14, fontweight='bold')
                plt.ylabel(axis_dict[key], fontsize=14, fontweight='bold')
                if 'map' in key.lower():
                    plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, key + "_mean.png"), dpi=300)
                if phase == "training":
                    break 
                    


            if phase == "testing":
                return


def plot_model_individual_with_collective(src_path, phase):
    if not os.path.exists(src_path):
        print("File not found: ", src_path)
        raise FileNotFoundError(src_path)
    elif len(os.listdir(src_path)) == 0:
        print("Empty folder: ", src_path)
        raise FileNotFoundError(src_path)
    else:
        for folder in model_list:
            output_path = os.path.join(src_path, folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            resultdict = {}
            for dir in os.listdir(src_path):
                if dir==folder:
                    continue
                elif folder not in dir:
                    continue
                else:
                    results = pd.read_csv(os.path.join(src_path,dir, "metrics.csv"))
                    results = results.rename(columns=translation_dict)
                    results = results[relevant_keys]
                    resultdict[dir] = results
            folds = len(resultdict.keys())
            result_df = pd.DataFrame(columns=relevant_keys)
            for key in relevant_keys:
                if key == 'epoch':
                    continue
                plt.figure(figsize=(3,3))
                plt.locator_params(nbins=5)
                metric_sums = []
                for dir in resultdict.keys():
                    results = resultdict[dir]
                    metric_sums.append(results[key])
                metric_sums = numpy.array(metric_sums)
                metric_sums = metric_sums.reshape(folds, -1)
                # print(metric_sums)
                sem = numpy.std(metric_sums, axis=0)#/numpy.sqrt(folds)
                metric_sums = numpy.mean(metric_sums, axis=0)
                result_df[key] = metric_sums
                # print(metric_sums)
                # print(sem)

                plt.plot(metric_sums, color='#0000FF', linewidth=1)
                # errorbars at every 5th point without connecting lines
                plt.errorbar(numpy.arange(0, len(metric_sums), 5), metric_sums[::5], yerr=sem[::5], capsize=1, capthick=1, elinewidth=1, color='black', linewidth=0)
                

                plt.title(f'{plot_titles_dict[key]}\nfor {folder}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_labels_dict[x_key], fontsize=14, fontweight='bold')
                plt.ylabel(axis_labels_dict[key], fontsize=14, fontweight='bold')
                # print basic stats
                if 'mAP' in key:
                    print(f'{folder} {key} mean: {numpy.max(sem)}')
                    print(f'{folder} {key} std: {numpy.max(metric_sums)}')
                if 'loss' in key:
                    plt.ylim(0, max(results[key]))
                else:
                    plt.ylim(0, 1)

                plt.tight_layout()
                plt.savefig(os.path.join(output_path, plot_save_names_dict[key] + "_mean.png"), dpi=300)
                plt.close()
            result_df['epoch'] = results['epoch']
            pd.DataFrame.to_csv(result_df, os.path.join(output_path, "metrics.csv"),index=False)

            col1 = 'metrics_mAP_0.5'
            col2 = 'metrics_mAP_0.5:0.95'
            plt.figure(figsize=(3, 3))
            plt.locator_params(nbins=5)
            copy_info = result_df.copy()
            copy_info = copy_info[[x_key, col1, col2]].dropna(axis=0, how='any')
            plt.plot(copy_info[x_key], copy_info[col1], linewidth=1, label='mAP')   
            plt.plot(copy_info[x_key], copy_info[col2], linewidth=1, label='mAP50')
            plt.title(f'mAP for {folder}', fontsize=14, fontweight='bold')
            plt.xlabel(axis_labels_dict[x_key], fontsize=14, fontweight='bold')
            plt.ylabel('mAP', fontsize=14, fontweight='bold')
            plt.legend()
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'mAP_vs_mAP50.png'), dpi=300)

                
            if phase == "testing":
                return
            
def give_stats(src_path):
    if not os.path.exists(src_path):
        print("File not found: ", src_path)
        raise FileNotFoundError(src_path)
    elif len(os.listdir(src_path)) == 0:
        print("Empty folder: ", src_path)
        raise FileNotFoundError(src_path)
    else:
        master_df = pd.DataFrame(columns=['model', 'metrics_precision', 'metrics_recall', 'metrics_mAP_0.5', 'metrics_mAP_0.5:0.95'])
        model_list = ['yolov5m','yolov5x','yolov5l',]
        prec_list = []
        rec_list = []
        map50_list = []
        map5095_list = []
        for folder in model_list:
            output_path = os.path.join(src_path, folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            resultdict = {}
            for dir in os.listdir(src_path):
                if dir==folder:
                    continue
                elif folder not in dir:
                    continue
                else:
                    results = pd.read_csv(os.path.join(src_path,dir, "metrics.csv"))
                    results = results.rename(columns=translation_dict)
                    results = results[['metrics_precision', 'metrics_recall', 'metrics_mAP_0.5', 'metrics_mAP_0.5:0.95']]
                    max_results = results.max()
                    prec_list.append(max_results['metrics_precision'])
                    rec_list.append(max_results['metrics_recall'])
                    map50_list.append(max_results['metrics_mAP_0.5'])
                    map5095_list.append(max_results['metrics_mAP_0.5:0.95'])
                    print(max_results)
                    break
        master_df['model'] = model_list
        master_df['metrics_precision'] = prec_list
        master_df['metrics_recall'] = rec_list
        master_df['metrics_mAP_0.5'] = map50_list
        master_df['metrics_mAP_0.5:0.95'] = map5095_list
        print(master_df)
        # to latex
        print(master_df.to_latex(index=False))
        # master_df.to_csv(os.path.join(src_path, "stats.csv"),index=False)            
    return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    plot_model_individual(args.path, args.phase)
    # plot_model_individual_with_collective(args.path, args.phase)
    # give_stats(args.path)

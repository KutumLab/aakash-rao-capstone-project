import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import argparse

keys = ['               epoch', '      train/box_loss', '      train/obj_loss',
       '      train/cls_loss', '   metrics/precision', '      metrics/recall',
       '     metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', '        val/box_loss',
       '        val/obj_loss', '        val/cls_loss', '               x/lr0',
       '               x/lr1', '               x/lr2']

translation_array = ['epoch', 'train_box_loss', 'train_obj_loss',
         'train_cls_loss', 'metrics_precision', 'metrics_recall',
         'metrics_mAP_0.5', 'metrics_mAP_0.5:0.95', 'val_box_loss',
         'val_obj_loss', 'val_cls_loss', 'lr0',
         'lr1', 'lr2']
translation_dict = dict(zip(keys, translation_array))
relevant_keys = ['epoch', 'train_box_loss', 'train_obj_loss', 'train_cls_loss', 'metrics_precision', 'metrics_recall', 'metrics_mAP_0.5', 'metrics_mAP_0.5:0.95']
plot_titles = ['Epoch','Box loss', 'Object loss', 'Class loss', 'Precision', 'Recall', 'mAP at IoU=0.5', 'mAP at IoU=0.5:0.95']
axis_labels = ['Epoch', 'Loss', 'Loss', 'Loss', 'Precision', 'Recall', 'mAP', 'mAP']
plot_save_names = ['epoch','box_loss', 'obj_loss', 'cls_loss', 'precision', 'recall', 'mAP_50', 'mAP_50_95']


model_list = ['yolov5m','yolov5x','yolov5l',]

# dictionary to map the keys to the plot titles
plot_titles_dict = dict(zip(relevant_keys, plot_titles))
axis_labels_dict = dict(zip(relevant_keys, axis_labels))
plot_save_names_dict = dict(zip(relevant_keys, plot_save_names))


def plot_model_individual(src_path, phase):
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
                    results = pd.read_csv(os.path.join(src_path,dir, "results.csv"))
                    results = results.rename(columns=translation_dict)
                    results = results[relevant_keys]
                    resultdict[dir] = results
            folds = len(resultdict.keys())
            for key in relevant_keys:
                if key == 'epoch':
                    continue
                plt.figure(figsize=(3,3))
                plt.locator_params(nbins=5)
                x_key = 'epoch'
                for dir in resultdict.keys():
                    results = resultdict[dir]
                    plt.plot(results[x_key], results[key], label=f'fold {dir[-1]}', linewidth=1)
                plt.title(f'{plot_titles_dict[key]}\nfor {folder}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_labels_dict[x_key], fontsize=14, fontweight='bold')
                plt.ylabel(axis_labels_dict[key], fontsize=14, fontweight='bold')
                if 'loss' in key:
                    plt.ylim(0, max(results[key]))
                else:
                    plt.ylim(0, 1)
                plt.legend(loc='best', fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, plot_save_names_dict[key] + ".png"), dpi=300)
                plt.close()
                # break
                # finding mean across three models


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
                    results = pd.read_csv(os.path.join(src_path,dir, "results.csv"))
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
                x_key = 'epoch'
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
            pd.DataFrame.to_csv(result_df, os.path.join(output_path, "results.csv"),index=False)
                
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
                    results = pd.read_csv(os.path.join(src_path,dir, "results.csv"))
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
        # master_df.to_csv(os.path.join(src_path, "stats.csv"),index=False)            
    return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    # plot_model_individual(args.src_path, args.phase)
    # plot_model_individual_with_collective(args.src_path, args.phase)
    give_stats(args.src_path)

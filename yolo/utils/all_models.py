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


def plot(src_path, phase):
    if not os.path.exists(src_path):
        print("File not found: ", src_path)
        raise FileNotFoundError(src_path)
    elif len(os.listdir(src_path)) == 0:
        print("Empty folder: ", src_path)
        raise FileNotFoundError(src_path)
    else:
        output_path = os.path.join(src_path)
        resultdict = {}
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for folder in model_list:
            results = pd.read_csv(os.path.join(src_path, folder, "results.csv"))
            results = results.rename(columns=translation_dict)
            results = results[relevant_keys]
            resultdict[folder] = results
        
        for key in relevant_keys:
            plt.figure(figsize=(3,3))
            plt.locator_params(nbins=5)
            x_key = 'epoch'
            for dir in resultdict.keys():
                plt.plot(resultdict[dir][x_key], resultdict[dir][key], label=dir, linewidth=1)
            plt.title(plot_titles_dict[key], fontsize=14, fontweight='bold')
            plt.xlabel(axis_labels_dict[x_key], fontsize=14, fontweight='bold')
            plt.ylabel(axis_labels_dict[key], fontsize=14, fontweight='bold')
            plt.legend(loc = 'best', fontsize=8)
            if 'loss' in key:
                plt.ylim(0, max(resultdict[dir][key]))
            else:
                plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, plot_save_names_dict[key] + ".png"), dpi=300)
            plt.close()
                
        if phase == "testing":
            return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    plot(args.src_path, args.phase)
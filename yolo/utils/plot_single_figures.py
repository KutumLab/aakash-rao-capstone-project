import os
import numpy
import pandas
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
        for folder in os.listdir(src_path):
            if len(folder) < 7:
                continue
            elif folder[-3:] == 'png':
                continue
            outpath = os.path.join(src_path, folder)
            results = pandas.read_csv(os.path.join(outpath, "results.csv"))
            plot_save_path = os.path.join(outpath, "plot")
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            results = results.rename(columns=translation_dict)
            results = results[relevant_keys]
            for key in relevant_keys:
                plt.figure(figsize=(3,3))
                x_key = 'epoch'
                plt.plot(results[x_key], results[key], color='#0000FF', linewidth=1)
                plt.title(f'{plot_titles_dict[key]} \n for fold {folder[-1]}', fontsize=14, fontweight='bold')
                plt.xlabel(axis_labels_dict[x_key], fontsize=14, fontweight='bold')
                plt.ylabel(axis_labels_dict[key], fontsize=14, fontweight='bold')
                if 'loss' in key:
                    plt.ylim(0, max(results[key]))
                else:
                    plt.ylim(0, 1)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                # plt.grid(True)

                plt.tight_layout()
                plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                plt.savefig(os.path.join(plot_save_path, plot_save_names_dict[key] + ".png"), dpi=300)
                plt.close()

            # if phase == "testing":
            #     print(results.columns)
            #     return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    plot(args.src_path, args.phase)

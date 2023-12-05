import numpy as np
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

cols = ['AP5095', 'AP50', 'AP75', 'cls_accuracy', 'iteration', 'loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
titles = ['mAP@50:95', 'mAP@50', 'mAP@75', 'Class Accuracy', 'Iteration', 'Box Loss', 'Object Loss', 'Total Loss', 'Validation Loss']
axis = ['mAP', 'mAP', 'mAP', 'Accuracy', 'Iteration', 'Loss', 'Loss', 'Loss', 'Loss']
title_dict = dict(zip(cols, titles))
y_axis_dict = dict(zip(cols, axis))
x_axis_dict = "No. of Epochs"
model_list = ['faster_rcnn_R_101_DC5_3x','faster_rcnn_R_50_C4_1x','faster_rcnn_R_50_C4_3x','faster_rcnn_R_50_DC5_1x','faster_rcnn_R_50_DC5_3x']
names = ['ResNet101 with DC5 at 3x', 'ResNet50 with C4 at 1x', 'ResNet50 with C4 at 3x', 'ResNet50 with DC5 at 1x', 'ResNet50 with DC5 at 3x']
model_dict = dict(zip(model_list, names))

def plot_model(data_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for model in os.listdir(data_dir):
        print(model)
        model_plot_dir = os.path.join(plot_dir, model)
        if not os.path.exists(model_plot_dir):
            os.makedirs(model_plot_dir)

        mean_df = pd.read_csv(os.path.join(data_dir, model, f'mean_{model}.csv'), header=0)
        std_df = pd.read_csv(os.path.join(data_dir, model, f'std_{model}.csv'), header=0)

        for column in mean_df.columns:
            if column == 'iteration' or 'unnamed' in column.lower():
                continue
            mean = mean_df[column]
            mean = mean.dropna(how='any')
            # take 150 values at equal intervals
            mean = mean.iloc[::len(mean)//150]
            print(len(mean))
            std = std_df[column]
            std = std.dropna(how='any')
            # take 150 values at equal intervals
            std = std.iloc[::len(std)//150]
            plt.figure(figsize=(4, 4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
            plt.locator_params(axis='x', nbins=5)
            plt.locator_params(axis='y', nbins=5)
            plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
            plt.plot(mean, label='mean',linewidth=0.75)
            plt.fill_between(mean.index, mean - std, mean + std, alpha=0.5, label='std')
            plt.title(title_dict[column], fontsize=12, fontweight='bold')
            plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
            plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
            plt.xticks(np.arange(0, 150, 150//5), fontsize=10)
            plt.yticks(fontsize=10)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(model_plot_dir, f'{column}.png'), dpi=300)
            plt.close()


def plot_metric(data_dir, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for column in cols:
        if column == 'iteration' or 'unnamed' in column.lower():
            continue
        plt.figure(figsize=(6, 4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
        for model in os.listdir(data_dir):
            mean_df = pd.read_csv(os.path.join(data_dir, model, f'mean_{model}.csv'), header=0)
            mean = mean_df[column]
            mean = mean.dropna(how='any')
            plt.plot(mean, label=model,linewidth=0.75)
            std_df = pd.read_csv(os.path.join(data_dir, model, f'std_{model}.csv'), header=0)
            std = std_df[column]
            std = std.dropna(how='any')
            plt.fill_between(mean.index, mean - std, mean + std, alpha=0.25, label=model)
        # custom legend content
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
        plt.legend([handles[idx] for idx in order],[model_dict[labels[idx]] for idx in order], fontsize=8,bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f'{title_dict[column]} for Faster R-CNN', fontsize=12, fontweight='bold')
        plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
        plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{column}.png'), dpi=300)
        plt.close()

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-d','--data_dir', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')

    args = argparse.parse_args()
    plot_model(args.data_dir, args.output_dir)
    plot_metric(args.data_dir, args.output_dir)


import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

cols = ['AP5095', 'AP50', 'AP75', 'iteration', 'loss_box_reg', 'loss_cls',  'total_loss', 'validation_loss']
titles = ['mAP@50:95', 'mAP@50', 'mAP@75',  'Iteration', 'Box Loss', 'Object Loss', 'Total Loss', 'Validation Loss']
axis = ['mAP', 'mAP', 'mAP',  'Iteration', 'Loss', 'Loss', 'Loss', 'Loss']
title_dict = dict(zip(cols, titles))
y_axis_dict = dict(zip(cols, axis))
x_axis_dict = "No. of Epochs"
model_list = ['faster_rcnn_R_50_DC5_3x']
names = ['ResNet50 with FPN at 3x']
model_dict = dict(zip(model_list, names))

archi = {
    'faster_rcnn': 'faster_rcnn_R_50_DC5_3x'
}



def plot_metric(four_class_path, three_class_path, single_path, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    four_class_mean = pd.read_csv(os.path.join(four_class_path, model, 'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
    three_class_mean = pd.read_csv(os.path.join(three_class_path, model, 'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
    single_mean = pd.read_csv(os.path.join(single_path, model,  'csvs',archi[model], f'mean_{archi[model]}.csv'), header=0)
    four_class_std = pd.read_csv(os.path.join(four_class_path, model, 'csvs',archi[model], f'std_{archi[model]}.csv'), header=0)
    three_class_std = pd.read_csv(os.path.join(three_class_path, model, 'csvs',archi[model], f'std_{archi[model]}.csv'), header=0)
    single_std = pd.read_csv(os.path.join(single_path, model,  'csvs',archi[model], f'std_{archi[model]}.csv'), header=0)
    for column in cols:
        if column == 'iteration' or 'unnamed' in column.lower():
            continue
        plt.figure(figsize=(4,4))
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.grid(alpha=0.5, linestyle='--', linewidth=0.75)
        plt.plot(four_class_mean['iteration'], four_class_mean[column], label='4 Class', color='blue')
        plt.plot(three_class_mean['iteration'], three_class_mean[column], label='3 Class', color='orange')
        plt.plot(single_mean['iteration'], single_mean[column], label='Single Class', color='green')
        plt.legend(loc='best', fontsize=10)

        plt.plot(four_class_std['iteration'], four_class_std[column], 'o', color='blue')
        plt.plot(three_class_std['iteration'], three_class_std[column], 'o', color='orange')
        plt.plot(single_std['iteration'], single_std[column], 'o', color='green')


        plt.title(f'{title_dict[column]} for {model_dict[model]}', fontsize=12, fontweight='bold')
        plt.xlabel(x_axis_dict, fontsize=12, fontweight='bold')
        plt.ylabel(y_axis_dict[column], fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{column}.png'), dpi=300)
        plt.close()

        
    pass

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-fcp','--four_class_path', type=str, default='data')
    argparse.add_argument('-tcp','--three_class_path', type=str, default='data')
    argparse.add_argument('-sp','--single_path', type=str, default='data')
    argparse.add_argument('-o','--output_dir', type=str, default='data')
    argparse.add_argument('-m','--model', type=str, default='faster_rcnn_R_50_DC5_3x')

    args = argparse.parse_args()
    plot_metric(args.four_class_path, args.three_class_path, args.single_path, args.output_dir, args.model)
